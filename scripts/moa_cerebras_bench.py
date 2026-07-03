#!/usr/bin/env python3
"""Cerebras wafer-speed slots in MoA: aggregator, references, full stack.

Answers the backlog item ("Cerebras Gemma 4 as fast classifier/aggregator")
empirically on the 40-task verified set:

- Cerebras solos (gemma-4-31b, gpt-oss-120b, zai-glm-4.7) — baseline quality
  and raw speed of each hosted model through the identical MoA code path.
- ``cere-agg-openrefs``: strong OpenRouter references (the open-moa-heavy
  set) merged by a Cerebras fast aggregator — how much accuracy does the
  fast-merge trade cost against the wall-clock win vs kimi-k2.6 aggregation?
- ``cere-moa``: an all-Cerebras MoA (every slot wafer-served) — the
  lowest-latency composition available.

Requires CEREBRAS_API_KEY (paid tier: the free tier RPM-collapses under
bench concurrency) and OPENROUTER_API_KEY for the openrefs config.

Usage (inside the moa-proxy container):
  python scripts/moa_cerebras_bench.py --out /out/cerebras-bench.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moa_bench import _ANSWER_INSTRUCTION, extract_answer, is_correct  # noqa: E402
from moa_learning_cycle import HELDOUT_TASKS, TRAIN_TASKS  # noqa: E402

TASKS = TRAIN_TASKS + HELDOUT_TASKS

CB = "custom:cerebras"
OR = "openrouter"


def _slot(provider: str, model: str) -> dict:
    return {"provider": provider, "model": model}


def _preset(refs: list[tuple[str, str]] | None, agg: tuple[str, str]) -> dict:
    return {
        "enabled": bool(refs),
        "reference_models": [_slot(p, m) for p, m in (refs or [(OR, "unused/ref")])],
        "aggregator": _slot(*agg),
        "reference_max_tokens": 1500,
    }


HEAVY_REFS = [
    (OR, "deepseek/deepseek-v4-pro"),
    (OR, "qwen/qwen3.7-max"),
    (OR, "z-ai/glm-5.2"),
]

CONFIGS: dict[str, dict] = {
    # Cerebras-hosted solos.
    "cere-gemma31-solo": _preset(None, (CB, "gemma-4-31b")),
    "cere-gptoss-solo": _preset(None, (CB, "gpt-oss-120b")),
    "cere-glm47-solo": _preset(None, (CB, "zai-glm-4.7")),
    # Strong references + wafer-speed aggregator (the backlog experiment).
    "cere-agg-openrefs-gemma": _preset(HEAVY_REFS, (CB, "gemma-4-31b")),
    "cere-agg-openrefs-gptoss": _preset(HEAVY_REFS, (CB, "gpt-oss-120b")),
    # All-Cerebras MoA: every slot wafer-served.
    "cere-moa": _preset(
        [(CB, "gpt-oss-120b"), (CB, "gemma-4-31b")],
        (CB, "zai-glm-4.7"),
    ),
}


def _write_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    blocks = []
    for name, preset in CONFIGS.items():
        refs = "\n".join(
            f"        - provider: {s['provider']}\n          model: {s['model']}"
            for s in preset["reference_models"]
        )
        blocks.append(
            f"    {name}:\n"
            f"      enabled: {str(preset['enabled']).lower()}\n"
            f"      reference_models:\n{refs}\n"
            f"      aggregator:\n"
            f"        provider: {preset['aggregator']['provider']}\n"
            f"        model: {preset['aggregator']['model']}\n"
            f"      reference_max_tokens: {preset['reference_max_tokens']}"
        )
    (home / "config.yaml").write_text(
        "providers:\n"
        "  cerebras:\n"
        "    api: https://api.cerebras.ai/v1\n"
        "    name: cerebras\n"
        "    default_model: gemma-4-31b\n"
        "moa:\n  default_preset: cere-moa\n  presets:\n" + "\n".join(blocks) + "\n",
        encoding="utf-8",
    )


def run_turn(config_name: str, task: dict) -> dict:
    from agent.moa_loop import MoAChatCompletions, _extract_text

    started = time.time()
    result = {"config": config_name, "task": task["id"], "expected": task["a"]}
    try:
        facade = MoAChatCompletions(config_name)
        response = facade.create(
            messages=[{"role": "user", "content": task["q"] + _ANSWER_INSTRUCTION}],
            max_tokens=8000,
            timeout=240,
        )
        text = _extract_text(response)
        ref_usage, _ = facade.consume_reference_usage()
        agg_usage = getattr(response, "usage", None)
        extracted = extract_answer(text)
        result.update(
            {
                "answer": extracted,
                "correct": is_correct(extracted, task["a"], task.get("aliases")),
                "latency_s": round(time.time() - started, 1),
                "ref_tokens": int(getattr(ref_usage, "input_tokens", 0) or 0)
                + int(getattr(ref_usage, "output_tokens", 0) or 0),
                "agg_tokens": int(getattr(agg_usage, "prompt_tokens", 0) or 0)
                + int(getattr(agg_usage, "completion_tokens", 0) or 0),
            }
        )
    except Exception as exc:
        result.update(
            {"answer": None, "correct": False, "error": str(exc)[:300],
             "latency_s": round(time.time() - started, 1)}
        )
    return result


def print_report(results: list[dict]) -> None:
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)
    print(f"\n=== Cerebras MoA slots — {len(TASKS)} tasks ===\n")
    header = f"{'config':28s} {'acc':>7s} {'avg_lat':>8s} {'tok/q':>7s}  wrong/error"
    print(header)
    print("-" * len(header))
    for name in CONFIGS:
        rows = by_config.get(name)
        if not rows:
            continue
        n = len(rows)
        ok = sum(1 for r in rows if r["correct"])
        lat = sum(r.get("latency_s") or 0 for r in rows) / n
        tok = sum((r.get("ref_tokens") or 0) + (r.get("agg_tokens") or 0) for r in rows) / n
        wrong = [r["task"] + ("(err)" if r.get("error") else "") for r in rows if not r["correct"]]
        print(f"{name:28s} {ok:>3d}/{n:<3d} {lat:>7.1f}s {tok:>7.0f}  {', '.join(wrong) or '-'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-cerebras-bench.json")
    parser.add_argument("--home", default="/tmp/moa-cerebras-home")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.report:
        print_report(json.loads(Path(args.report).read_text(encoding="utf-8"))["results"])
        return 0
    if not os.environ.get("CEREBRAS_API_KEY"):
        print("CEREBRAS_API_KEY is required", file=sys.stderr)
        return 1

    home = Path(args.home)
    _write_home(home)
    os.environ["HERMES_HOME"] = str(home)

    config_names = list(CONFIGS)
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    jobs = [(c, t) for c in config_names for t in TASKS]
    print(f"Running {len(jobs)} turns ({len(config_names)} configs x {len(TASKS)} tasks)...")
    results: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_turn, c, t): (c, t["id"]) for c, t in jobs}
        for done, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            status = "ok " if r["correct"] else ("ERR" if r.get("error") else "X  ")
            print(
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:28s} {r['task']:14s} "
                f"-> {str(r.get('answer'))[:32]!r} ({r.get('latency_s')}s)",
                flush=True,
            )
            if done % 20 == 0:
                out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
