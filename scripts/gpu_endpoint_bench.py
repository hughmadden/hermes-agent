#!/usr/bin/env python3
"""Measure decode throughput + inter-token latency of an OpenAI endpoint.

Used for the single-GPU vs TP=2-over-PCIe comparison on pg (RTX 5090 + 4090).
Streams completions and timestamps every chunk: reports single-stream decode
tok/s, mean/p50/p90 inter-token latency, TTFT, and batched aggregate tok/s at
several concurrency levels. Stdlib only (urllib), so it runs anywhere.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request

PROMPT = (
    "Write a detailed, flowing essay (no lists) about the history of "
    "navigation at sea, covering dead reckoning, celestial navigation, the "
    "longitude problem and John Harrison's chronometers, radio navigation, "
    "and GPS. Aim for maximum length and detail."
)


def one_stream(base: str, model: str, max_tokens: int, prompt_extra: str = "") -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT + prompt_extra}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
    )
    started = time.time()
    first = None
    stamps: list[float] = []
    n_chunks = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            if delta.get("content"):
                now = time.time()
                if first is None:
                    first = now
                stamps.append(now)
                n_chunks += 1
    total = time.time() - started
    itls = [b - a for a, b in zip(stamps, stamps[1:])]
    return {
        "chunks": n_chunks,
        "ttft_s": round((first - started), 3) if first else None,
        "decode_s": round(stamps[-1] - first, 3) if len(stamps) > 1 else None,
        "decode_tok_s": round((n_chunks - 1) / (stamps[-1] - first), 1)
        if len(stamps) > 1 and stamps[-1] > first
        else None,
        "itl_ms": {
            "mean": round(1000 * statistics.mean(itls), 2) if itls else None,
            "p50": round(1000 * statistics.median(itls), 2) if itls else None,
            "p90": round(1000 * sorted(itls)[int(len(itls) * 0.9)], 2) if itls else None,
        },
        "wall_s": round(total, 2),
    }


def concurrent_streams(base: str, model: str, max_tokens: int, n: int) -> dict:
    results: list[dict] = []
    lock = threading.Lock()

    def worker(i: int):
        r = one_stream(base, model, max_tokens, prompt_extra=f" (variant {i})")
        with lock:
            results.append(r)

    started = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    total_chunks = sum(r["chunks"] for r in results)
    return {
        "concurrency": n,
        "aggregate_tok_s": round(total_chunks / wall, 1),
        "wall_s": round(wall, 2),
        "per_stream_decode_tok_s": [r["decode_tok_s"] for r in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:8700/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True, help="config label for the report")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--single-runs", type=int, default=3)
    parser.add_argument("--concurrency", default="4,16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print(f"[{args.label}] warmup...", flush=True)
    one_stream(args.base, args.model, 64)

    singles = []
    for i in range(args.single_runs):
        r = one_stream(args.base, args.model, args.max_tokens, prompt_extra=f" ({i})")
        singles.append(r)
        print(f"[{args.label}] single #{i+1}: {r}", flush=True)

    conc = []
    for n in [int(x) for x in args.concurrency.split(",") if x.strip()]:
        r = concurrent_streams(args.base, args.model, args.max_tokens, n)
        conc.append(r)
        print(f"[{args.label}] concurrency {n}: {r}", flush=True)

    report = {
        "label": args.label,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "single_streams": singles,
        "single_decode_tok_s_best": max(
            (r["decode_tok_s"] or 0) for r in singles
        ),
        "concurrent": conc,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
