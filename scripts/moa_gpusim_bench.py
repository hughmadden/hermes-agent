#!/usr/bin/env python3
"""Quality simulation for the 4x96GB workstation plan (multi-gpu-moa-plan).

Compares, via OpenRouter stand-ins, the two ways to spend 4 x 96 GB with no
NVLink:

- ``gpu4-composed``: four models that each fit ONE 96 GB card (quantized),
  composed as MoA — three references + an aggregator. No inter-GPU traffic.
- ``tp4-*``: one large model that genuinely needs TP=4 (or TP=2) across the
  cards over PCIe, running solo through the identical code path.

Task set: the 40 verified exact-answer tasks from moa_learning_cycle
(train + held-out combined) — harder than moa_bench's original 16, so big
solos are not all at ceiling. Coding quality is simulated separately via the
aider polyglot runs (see docs/plans/multi-gpu-moa-plan.md).

Usage (inside the moa-proxy container):
  python scripts/moa_gpusim_bench.py --out /out/gpusim.json [--configs a,b]
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

# All slots ride OpenRouter. Sizing evidence: docs/plans/multi-gpu-moa-plan.md.
def _preset(refs: list[str] | None, agg: str) -> dict:
    return {
        "enabled": bool(refs),
        "reference_models": [
            {"provider": "openrouter", "model": m} for m in (refs or ["unused/ref"])
        ],
        "aggregator": {"provider": "openrouter", "model": agg},
        "reference_max_tokens": 1500,
    }

CONFIGS: dict[str, dict] = {
    # Side A: four single-96GB-card models composed (the MoA box).
    "gpu4-composed": _preset(
        [
            "qwen/qwen3-coder-next",            # ~80GB FP8 — coding card
            "openai/gpt-oss-120b",              # ~63GB MXFP4 — reasoning card
            "qwen/qwen3-next-80b-a3b-thinking", # ~80GB FP8 — thinking card
        ],
        "qwen/qwen3.5-122b-a10b",               # ~76GB NVFP4 — aggregator card
    ),
    # Side A': the strongest single-card model alone (1 GPU, no composition).
    "gpu1-solo-qwen35-122b": _preset(None, "qwen/qwen3.5-122b-a10b"),
    "gpu1-solo-gptoss-120b": _preset(None, "openai/gpt-oss-120b"),
    # Side B: models that need multi-GPU tensor parallel over PCIe.
    "tp4-nemotron-ultra-550b": _preset(None, "nvidia/nemotron-3-ultra-550b-a55b"),
    "tp4-qwen35-397b": _preset(None, "qwen/qwen3.5-397b-a17b"),
    "tp2-deepseek-v4-flash": _preset(None, "deepseek/deepseek-v4-flash"),
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
        "moa:\n  default_preset: gpu4-composed\n  presets:\n" + "\n".join(blocks) + "\n",
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
    print(f"\n=== 4x96GB quality simulation — {len(TASKS)} tasks ===\n")
    header = f"{'config':26s} {'acc':>7s} {'avg_lat':>8s} {'tok/q':>7s}  wrong/error"
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
        print(f"{name:26s} {ok:>3d}/{n:<3d} {lat:>7.1f}s {tok:>7.0f}  {', '.join(wrong) or '-'}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-gpusim.json")
    parser.add_argument("--home", default="/tmp/moa-gpusim-home")
    parser.add_argument("--configs", default=None)
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
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:26s} {r['task']:14s} "
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
