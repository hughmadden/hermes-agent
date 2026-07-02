#!/usr/bin/env python3
"""moa:auto router mini-bench: classification accuracy + added latency.

Runs a labelled prompt set through the routing classifier only (no MoA turns,
so it costs fractions of a cent) and reports per-class accuracy, latency
percentiles, and the confusion cases. Compares any number of classifier
slots — e.g. Cerebras gemma-4-31b vs the same model on OpenRouter.

Usage (inside the moa-proxy container, keys in env):
  python scripts/moa_router_bench.py --out /out/router-bench.json \
      [--classifiers "custom:cerebras=gemma-4-31b,openrouter=google/gemma-4-31b-it"]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

# Labelled routing cases. Classes match the reference router config below:
# coding / math / general / self. Deliberately includes borderline items —
# the router's job is coarse triage, not perfection.
CASES = [
    # coding
    ("Write a Python function that parses ISO dates from a CSV.", "coding"),
    ("Why does my Rust program panic with 'index out of bounds'?", "coding"),
    ("Refactor this SQL query to use a window function.", "coding"),
    ("Give me a bash one-liner to find files modified in the last hour.", "coding"),
    ("My pytest fixture isn't tearing down — how do I debug it?", "coding"),
    ("Convert this callback-based JS to async/await.", "coding"),
    # math
    ("What is the determinant of [[2,1],[5,3]]?", "math"),
    ("Prove that the sum of two even numbers is even.", "math"),
    ("A train travels 240km in 3 hours. What's its average speed?", "math"),
    ("How many ways can 8 people sit around a round table?", "math"),
    ("Integrate x^2 * e^x dx.", "math"),
    ("What is 15% compound interest on $2000 over 3 years?", "math"),
    # general
    ("Summarize the causes of the French Revolution in a paragraph.", "general"),
    ("Draft a polite email declining a meeting invitation.", "general"),
    ("Compare the pros and cons of remote work for a small team.", "general"),
    ("What are the main differences between Stoicism and Epicureanism?", "general"),
    ("Suggest a week-long itinerary for Japan in autumn.", "general"),
    ("Explain how mRNA vaccines work to a high schooler.", "general"),
    # self (trivial)
    ("hi!", "self"),
    ("thanks, that's perfect", "self"),
    ("good morning :)", "self"),
    ("what's the capital of France?", "self"),
    ("ok sounds good", "self"),
    ("can you say that again more simply?", "self"),
]

ROUTER_PRESETS = {
    "coding": "writing or debugging code, refactors, shell commands, programming questions",
    "math": "calculation, proofs, quantitative puzzles, numeric word problems",
    "general": "general knowledge, writing, analysis — everything that is not code, math, or trivial",
}


def _write_home(home: Path, provider: str, model: str) -> None:
    home.mkdir(parents=True, exist_ok=True)
    providers_block = ""
    if provider.startswith("custom:"):
        name = provider.split(":", 1)[1]
        # Base URL by convention; extend here if more custom hosts join.
        base = {"cerebras": "https://api.cerebras.ai/v1"}[name]
        providers_block = (
            f"providers:\n  {name}:\n    api: {base}\n    name: {name}\n"
            f"    default_model: {model}\n"
        )
    presets = "\n".join(
        f"    {key}:\n"
        f"      route:\n        description: {desc}\n"
        f"      reference_models:\n"
        f"        - provider: openrouter\n          model: unused/ref\n"
        f"      aggregator:\n        provider: openrouter\n        model: unused/agg"
        for key, desc in ROUTER_PRESETS.items()
    )
    (home / "config.yaml").write_text(
        providers_block
        + "moa:\n"
        "  default_preset: general\n"
        "  router:\n"
        "    enabled: true\n"
        f"    classifier:\n      provider: {provider}\n      model: {model}\n"
        "    default: general\n"
        "    timeout_s: 20\n"
        "  presets:\n" + presets + "\n",
        encoding="utf-8",
    )


async def bench_classifier(provider: str, model: str, home: Path, repeats: int) -> dict:
    os.environ["HERMES_HOME"] = str(home)
    _write_home(home, provider, model)

    from hermes_cli.config import load_config
    from hermes_cli.moa_config import normalize_moa_config
    from hermes_cli.proxy.moa_router import route_request, sticky_clear

    # Config module may cache per HERMES_HOME; re-normalize fresh each run.
    cfg = normalize_moa_config((load_config() or {}).get("moa") or {})
    assert cfg["router"]["enabled"], f"router failed to enable for {provider}:{model}"

    rows = []
    for rep in range(repeats):
        for prompt, expected in CASES:
            sticky_clear()  # every case classified fresh
            started = time.time()
            decision = await route_request(
                cfg, [{"role": "user", "content": prompt}]
            )
            rows.append(
                {
                    "prompt": prompt,
                    "expected": expected,
                    "routed": decision.preset_name,
                    "method": decision.method,
                    "latency_ms": int((time.time() - started) * 1000),
                    "rep": rep,
                }
            )
            ok = "ok " if decision.preset_name == expected else "X  "
            print(
                f"  {ok} [{decision.classifier_ms or '----':>4}ms] "
                f"{expected:8s} -> {decision.preset_name:8s} {prompt[:48]!r}",
                flush=True,
            )

    classified = [r for r in rows if r["method"] == "classified"]
    lat = sorted(r["latency_ms"] for r in classified) or [0]
    accuracy = sum(1 for r in rows if r["routed"] == r["expected"]) / len(rows)
    return {
        "classifier": f"{provider}:{model}",
        "cases": len(rows),
        "accuracy": round(accuracy, 3),
        "fallbacks": sum(1 for r in rows if r["method"] == "fallback"),
        "latency_ms": {
            "p50": lat[len(lat) // 2],
            "p90": lat[int(len(lat) * 0.9)],
            "max": lat[-1],
            "mean": int(statistics.mean(lat)),
        },
        "confusions": [
            {"prompt": r["prompt"], "expected": r["expected"], "routed": r["routed"]}
            for r in rows
            if r["routed"] != r["expected"]
        ],
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifiers",
        default="openrouter=google/gemma-4-31b-it",
        help="Comma-separated provider=model pairs",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--out", default="/tmp/moa-router-bench.json")
    parser.add_argument("--home", default="/tmp/moa-router-bench-home")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("CEREBRAS_API_KEY"):
        print("Need OPENROUTER_API_KEY and/or CEREBRAS_API_KEY", file=sys.stderr)
        return 1

    reports = []
    for spec in args.classifiers.split(","):
        provider, _, model = spec.strip().partition("=")
        if not model:
            print(f"Bad classifier spec {spec!r} (want provider=model)", file=sys.stderr)
            return 1
        print(f"\n=== classifier {provider}:{model} ({args.repeats}x{len(CASES)} cases) ===")
        report = asyncio.run(
            bench_classifier(provider, model, Path(args.home), args.repeats)
        )
        reports.append(report)
        print(
            f"  accuracy={report['accuracy']:.1%} fallbacks={report['fallbacks']} "
            f"latency p50={report['latency_ms']['p50']}ms "
            f"p90={report['latency_ms']['p90']}ms"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reports": reports}, indent=2), encoding="utf-8")
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
