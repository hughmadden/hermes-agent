#!/usr/bin/env python3
"""Reference-cap latency/accuracy trade for the heavy fan-out.

Hypothesis (from arXiv:2502.00674 "Self-MoA"): ensembling repeated samples
of the SINGLE BEST model beats mixing different reference models. If it
holds here, preset design simplifies to "pick your best model, sample it
N times" and the reference-selection problem disappears.

Design: aggregator fixed at kimi-k2.6 for every config; only the reference
set varies. Reference temperature (0.6, the preset default) provides sample
diversity for the self configs.

Usage (inside the moa-proxy container, OPENROUTER_API_KEY set):
  python scripts/moa_selfmoa_bench.py --out /out/selfmoa.json
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

AGG = "moonshotai/kimi-k2.6"


def _preset(refs: list[str] | None) -> dict:
    return {
        "enabled": bool(refs),
        "reference_models": [
            {"provider": "openrouter", "model": m} for m in (refs or ["unused/ref"])
        ],
        "aggregator": {"provider": "openrouter", "model": AGG},
        "reference_max_tokens": 1500,
    }

def _preset_cap(refs, cap):
    p = _preset(refs)
    p["reference_max_tokens"] = cap
    return p

HEAVY = ["deepseek/deepseek-v4-pro", "qwen/qwen3.7-max", "z-ai/glm-5.2"]
CONFIGS: dict[str, dict] = {
    "heavy-cap1500": _preset_cap(HEAVY, 1500),
    "heavy-cap600": _preset_cap(HEAVY, 600),
    "heavy-cap300": _preset_cap(HEAVY, 300),
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
            f"        provider: openrouter\n        model: {AGG}\n"
            f"      reference_max_tokens: {preset['reference_max_tokens']}"
        )
    (home / "config.yaml").write_text(
        "moa:\n  default_preset: kimi-selfmoa3\n  presets:\n" + "\n".join(blocks) + "\n",
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
    print(f"\n=== Self-MoA vs mixed refs vs solo (agg={AGG}) — {len(TASKS)} tasks ===\n")
    header = f"{'config':16s} {'acc':>7s} {'avg_lat':>8s} {'tok/q':>7s}  wrong/error"
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
        print(f"{name:16s} {ok:>3d}/{n:<3d} {lat:>7.1f}s {tok:>7.0f}  {', '.join(wrong) or '-'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-selfmoa.json")
    parser.add_argument("--home", default="/tmp/moa-selfmoa-home")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.report:
        print_report(json.loads(Path(args.report).read_text(encoding="utf-8"))["results"])
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    home = Path(args.home)
    _write_home(home)
    os.environ["HERMES_HOME"] = str(home)

    jobs = [(c, t) for c in CONFIGS for t in TASKS]
    print(f"Running {len(jobs)} turns...")
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
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:16s} {r['task']:14s} "
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
