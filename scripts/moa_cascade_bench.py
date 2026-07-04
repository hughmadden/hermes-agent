#!/usr/bin/env python3
"""Cascade-mode bench: HTTP driver over a live `hermes moa serve` endpoint.

Unlike the other moa_*_bench scripts (which drive `agent.moa_loop` directly
in-process), this one hits the OpenAI-compatible proxy over HTTP — it needs
`usage.moa.cascade` from the served response to see which tier a turn
resolved at, so it has to go through the real server path rather than the
facade. Start the server first:

  hermes moa serve --port 8652

Then run this bench against it. Datasets/graders are reused from the
existing hard-mode benches (AIME from moa_hard_bench, HMMT from
moa_mix_bench) so results are directly comparable to the facade-driven
baselines already measured there.

Usage:
  python scripts/moa_cascade_bench.py --models moa:cascade-wafer,moa:cere-moa \
      --dataset aime --out /tmp/cascade-aime.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moa_hard_bench import (  # noqa: E402
    _ANSWER_INSTRUCTION as _AIME_INSTRUCTION,
    extract_answer as _aime_extract_answer,
    fetch_aime,
    is_correct as _aime_is_correct,
)
from moa_mix_bench import (  # noqa: E402
    _ANSWER_INSTRUCTION as _HMMT_INSTRUCTION,
    extract_answer as _hmmt_extract_answer,
    fetch_hmmt,
    is_correct as _hmmt_is_correct,
)

DATASETS: dict[str, dict] = {
    "aime": {
        "fetch": fetch_aime,
        "instruction": _AIME_INSTRUCTION,
        "extract": _aime_extract_answer,
        "is_correct": _aime_is_correct,
    },
    "hmmt": {
        "fetch": fetch_hmmt,
        "instruction": _HMMT_INSTRUCTION,
        "extract": _hmmt_extract_answer,
        "is_correct": _hmmt_is_correct,
    },
}

_REQUEST_TIMEOUT = 420


def post_chat(base: str, model: str, question: str, instruction: str) -> dict:
    url = base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question + instruction}],
        "max_tokens": 16000,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def run_turn(base: str, model: str, task: dict, ds: dict) -> dict:
    started = time.time()
    result = {"model": model, "task": task["id"], "expected": task["a"]}
    try:
        body = post_chat(base, model, task["q"], ds["instruction"])
        latency = round(time.time() - started, 1)
        message = body["choices"][0]["message"]
        text = message.get("content") or ""
        extracted = ds["extract"](text)
        usage = body.get("usage") or {}
        cascade = (usage.get("moa") or {}).get("cascade") or {}
        result.update(
            {
                "answer": extracted,
                "correct": ds["is_correct"](extracted, task["a"]),
                "latency_s": latency,
                "tier": cascade.get("tier"),
                "total_tokens": usage.get("total_tokens"),
            }
        )
    except Exception as exc:
        result.update(
            {
                "answer": None,
                "correct": False,
                "error": str(exc)[:300],
                "latency_s": round(time.time() - started, 1),
            }
        )
    return result


def print_report(results: list[dict], dataset_name: str) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)
    print(f"\n=== Cascade bench — {dataset_name} ({len(results)} turns) ===\n")
    header = f"{'model':28s} {'acc':>8s} {'mean_lat':>9s} {'med_lat':>8s} {'mean_tok':>9s}  tiers"
    print(header)
    print("-" * len(header))
    for name, rows in by_model.items():
        n = len(rows)
        ok = sum(1 for r in rows if r["correct"])
        lats = [r["latency_s"] for r in rows if r.get("latency_s") is not None]
        mean_lat = sum(lats) / len(lats) if lats else 0.0
        med_lat = statistics.median(lats) if lats else 0.0
        toks = [r["total_tokens"] for r in rows if r.get("total_tokens") is not None]
        mean_tok = sum(toks) / len(toks) if toks else 0.0
        tier_counts: dict[str, int] = {}
        for r in rows:
            key = "n/a" if r.get("tier") is None else str(r["tier"])
            tier_counts[key] = tier_counts.get(key, 0) + 1
        tier_str = " ".join(f"{k}:{v}" for k, v in sorted(tier_counts.items()))
        print(
            f"{name:28s} {ok:>3d}/{n:<4d} {mean_lat:>8.1f}s {med_lat:>7.1f}s "
            f"{mean_tok:>9.0f}  {tier_str}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8652/v1")
    parser.add_argument(
        "--models",
        required=True,
        help="comma-separated model ids, e.g. moa:cascade-wafer,moa:cere-moa",
    )
    parser.add_argument("--dataset", choices=list(DATASETS), default="aime")
    parser.add_argument("--out", default="/tmp/moa-cascade-bench.json")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"], d.get("dataset", args.dataset))
        return 0

    ds = DATASETS[args.dataset]
    tasks = ds["fetch"]()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Fetched {len(tasks)} {args.dataset} problems")

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    jobs = [(m, t) for m in models for t in tasks]
    print(f"Running {len(jobs)} turns ({len(models)} models x {len(tasks)}) against {args.base}...")
    results: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_turn, args.base, m, t, ds): (m, t["id"]) for m, t in jobs
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            status = "ok " if r["correct"] else ("ERR" if r.get("error") else "X  ")
            print(
                f"[{done:>3d}/{len(jobs)}] {status} {r['model']:28s} {r['task']:12s} "
                f"tier={r.get('tier')} -> {str(r.get('answer'))[:8]!r} ({r.get('latency_s')}s)",
                flush=True,
            )
            if done % 20 == 0:
                out.write_text(
                    json.dumps(
                        {"dataset": args.dataset, "n_tasks": len(tasks), "results": results},
                        indent=2,
                    ),
                    encoding="utf-8",
                )

    out.write_text(
        json.dumps({"dataset": args.dataset, "n_tasks": len(tasks), "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nResults written to {out}")
    print_report(results, args.dataset)
    return 0


if __name__ == "__main__":
    sys.exit(main())
