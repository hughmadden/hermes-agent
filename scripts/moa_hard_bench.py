#!/usr/bin/env python3
"""Hard quality benchmark: AIME 2024 + 2025 (60 problems, integer answers).

The 40-task exact-answer set saturated at kimi-class strength (three configs
at 40/40), masking quality differences. AIME is the well-known hard
replacement: competition math, exact integer answers (0-999), standard in
every 2025/26 model card. Problems are fetched at runtime from the public
HuggingFace datasets server (math-ai/aime24 + math-ai/aime25) — nothing is
committed to the repo.

Retests the report's config lineup through the identical MoA turn machinery.
Requires OPENROUTER_API_KEY; Cerebras configs also need CEREBRAS_API_KEY.

Usage (inside the moa-proxy container):
  python scripts/moa_hard_bench.py --out /out/aime.json [--configs a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ANSWER_INSTRUCTION = (
    "\n\nSolve carefully. The answer is an integer between 0 and 999. End "
    "your reply with a final line of exactly this form and nothing after "
    "it:\nANSWER: <integer>"
)

CB = "custom:cerebras"
OR = "openrouter"


def _preset(refs: list[tuple[str, str]] | None, agg: tuple[str, str]) -> dict:
    return {
        "enabled": bool(refs),
        "reference_models": [
            {"provider": p, "model": m} for p, m in (refs or [(OR, "unused/ref")])
        ],
        "aggregator": {"provider": agg[0], "model": agg[1]},
        "reference_max_tokens": 1500,
    }


HEAVY_REFS = [
    (OR, "deepseek/deepseek-v4-pro"),
    (OR, "qwen/qwen3.7-max"),
    (OR, "z-ai/glm-5.2"),
]

CONFIGS: dict[str, dict] = {
    # Frontier anchors
    "fable-solo": _preset(None, (OR, "anthropic/claude-fable-5")),
    "gpt55-solo": _preset(None, (OR, "openai/gpt-5.5")),
    # Open solos
    "kimi-solo": _preset(None, (OR, "moonshotai/kimi-k2.6")),
    "v4flash-solo": _preset(None, (OR, "deepseek/deepseek-v4-flash")),
    # Compositions from the report
    "mixed-heavy": _preset(HEAVY_REFS, (OR, "moonshotai/kimi-k2.6")),
    "gpu4-composed": _preset(
        [
            (OR, "qwen/qwen3-coder-next"),
            (OR, "openai/gpt-oss-120b"),
            (OR, "qwen/qwen3-next-80b-a3b-thinking"),
        ],
        (OR, "qwen/qwen3.5-122b-a10b"),
    ),
    "open-moa-nano": _preset(
        [
            (OR, "meta-llama/llama-3.1-8b-instruct"),
            (OR, "mistralai/ministral-8b-2512"),
            (OR, "google/gemma-3-12b-it"),
        ],
        (OR, "google/gemma-3-27b-it"),
    ),
    # Cerebras lanes
    "cere-gemma31-solo": _preset(None, (CB, "gemma-4-31b")),
    "cere-moa": _preset(
        [(CB, "gpt-oss-120b"), (CB, "gemma-4-31b")], (CB, "zai-glm-4.7")
    ),
    "cere-agg-openrefs-gptoss": _preset(HEAVY_REFS, (CB, "gpt-oss-120b")),
}


def fetch_aime() -> list[dict]:
    """AIME 2024 (Maxwell-Jia/AIME_2024) + AIME 2025 (math-ai/aime25)."""
    sources = [
        ("Maxwell-Jia/AIME_2024", "train", "Problem", "Answer", "aime24"),
        ("math-ai/aime25", "test", "problem", "answer", "aime25"),
    ]
    tasks = []
    for name, split, q_key, a_key, tag in sources:
        url = (
            "https://datasets-server.huggingface.co/rows?dataset="
            + urllib.parse.quote(name, safe="")
            + f"&config=default&split={split}&offset=0&length=100"
        )
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = json.load(resp)
        for r in data["rows"]:
            row = r["row"]
            tasks.append(
                {
                    "id": f"{tag}-{r['row_idx']}",
                    "q": str(row[q_key]),
                    "a": str(row[a_key]).strip().split(".")[0],
                }
            )
    if len(tasks) < 55:
        raise RuntimeError(f"expected ~60 AIME problems, got {len(tasks)}")
    return tasks


def extract_answer(text: str) -> str:
    matches = re.findall(r"answer\s*:?\s*(?:is\s*)?\$?\\?(?:boxed\{)?(\d{1,3})\}?", text or "", flags=re.IGNORECASE)
    if matches:
        return matches[-1]
    boxed = re.findall(r"\\boxed\{(\d{1,3})\}", text or "")
    if boxed:
        return boxed[-1]
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if lines:
        nums = re.findall(r"\b(\d{1,3})\b", lines[-1])
        if nums:
            return nums[-1]
    return ""


def is_correct(extracted: str, expected: str) -> bool:
    try:
        return int(extracted) == int(expected)
    except (TypeError, ValueError):
        return False


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
        "moa:\n  default_preset: kimi-solo\n  presets:\n" + "\n".join(blocks) + "\n",
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
            max_tokens=16000,
            timeout=360,
        )
        text = _extract_text(response)
        ref_usage, _ = facade.consume_reference_usage()
        agg_usage = getattr(response, "usage", None)
        extracted = extract_answer(text)
        result.update(
            {
                "answer": extracted,
                "correct": is_correct(extracted, task["a"]),
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


def print_report(results: list[dict], n_tasks: int) -> None:
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)
    print(f"\n=== AIME 2024+2025 — {n_tasks} problems ===\n")
    header = f"{'config':28s} {'acc':>8s} {'avg_lat':>8s} {'tok/q':>7s}  errors"
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
        errs = sum(1 for r in rows if r.get("error"))
        print(f"{name:28s} {ok:>3d}/{n:<4d} {lat:>7.1f}s {tok:>7.0f}  {errs}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-aime.json")
    parser.add_argument("--home", default="/tmp/moa-aime-home")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"], d.get("n_tasks", 60))
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    tasks = fetch_aime()
    print(f"Fetched {len(tasks)} AIME problems")
    home = Path(args.home)
    _write_home(home)
    os.environ["HERMES_HOME"] = str(home)

    config_names = list(CONFIGS)
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",") if c.strip()]

    jobs = [(c, t) for c in config_names for t in tasks]
    print(f"Running {len(jobs)} turns ({len(config_names)} configs x {len(tasks)})...")
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
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:28s} {r['task']:12s} "
                f"-> {str(r.get('answer'))[:8]!r} ({r.get('latency_s')}s)",
                flush=True,
            )
            if done % 25 == 0:
                out.write_text(
                    json.dumps({"n_tasks": len(tasks), "results": results}, indent=2),
                    encoding="utf-8",
                )

    out.write_text(
        json.dumps({"n_tasks": len(tasks), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults written to {out}")
    print_report(results, len(tasks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
