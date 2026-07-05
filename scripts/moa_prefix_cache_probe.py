#!/usr/bin/env python3
"""Prefix-cache probe for agentic serving: cold vs warm TTFT with a growing
session prefix.

Agentic sessions re-send a large stable prefix (system + history) plus a
small delta every turn. With working prefix caching (vLLM
--enable-prefix-caching, GPU-resident block reuse), warm TTFT should be a
small fraction of cold TTFT. This probe measures exactly that against any
OpenAI-compatible endpoint, and optionally scrapes vLLM /metrics
prefix-cache counters before/after.

Stdlib only. Usage:
  python scripts/moa_prefix_cache_probe.py --base http://host:port/v1 \
      --model NAME --prefix-tokens 60000 [--metrics http://host:port/metrics]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request

_SYS = "You are a terse assistant inside a long-running agent session."
_API_KEY_ENV = None


def _headers() -> dict:
    h = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    key = os.environ.get(_API_KEY_ENV) if _API_KEY_ENV else None
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _build_prefix(target_tokens: int) -> list[dict]:
    """Deterministic fake agent history of ~target_tokens (4 chars/token)."""
    msgs = [{"role": "system", "content": _SYS}]
    per_turn_chars = 2000
    turns = max(1, (target_tokens * 4) // (per_turn_chars * 2))
    for i in range(turns):
        msgs.append(
            {
                "role": "user",
                "content": f"[step {i}] Run diagnostic {i} and summarize.",
            }
        )
        msgs.append(
            {
                "role": "assistant",
                "content": (
                    f"[diagnostic {i} output] "
                    + " ".join(f"metric_{i}_{j}=OK" for j in range(per_turn_chars // 16))
                ),
            }
        )
    return msgs


def _ttft(base: str, model: str, messages: list[dict], timeout: int) -> tuple[float, float, int]:
    """Stream one completion; return (ttft_s, total_s, completion_chars)."""
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "max_tokens": 60,
            "temperature": 0.0,
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers=_headers(),
    )
    t0 = time.time()
    first = None
    chars = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:])
            except ValueError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if delta.get("content"):
                if first is None:
                    first = time.time()
                chars += len(delta["content"])
    return ((first or time.time()) - t0, time.time() - t0, chars)


def _scrape_prefix_metrics(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)[:120]}
    out = {}
    for line in text.splitlines():
        if "prefix_cache" in line and not line.startswith("#"):
            m = re.match(r"(\S+?)(?:\{[^}]*\})?\s+([0-9.e+]+)", line)
            if m:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prefix-tokens", type=int, default=60000)
    ap.add_argument("--warm-calls", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--api-key-env", default=None)
    args = ap.parse_args()
    global _API_KEY_ENV
    _API_KEY_ENV = args.api_key_env

    prefix = _build_prefix(args.prefix_tokens)
    results = {"prefix_tokens_target": args.prefix_tokens, "calls": []}

    if args.metrics:
        results["metrics_before"] = _scrape_prefix_metrics(args.metrics)

    cold_msgs = prefix + [
        {"role": "user", "content": "Question: what is 12*11? Answer with the number only."}
    ]
    ttft, total, chars = _ttft(args.base, args.model, cold_msgs, args.timeout)
    results["calls"].append({"kind": "cold", "ttft_s": ttft, "total_s": total})
    print(f"cold : ttft {ttft:7.2f}s total {total:7.2f}s ({chars} chars)")

    grown = list(prefix)
    for i in range(args.warm_calls):
        grown = grown + [
            {"role": "user", "content": f"[step +{i}] one more small delta turn."},
            {"role": "assistant", "content": f"[delta {i} output] done."},
        ]
        warm_msgs = grown + [
            {"role": "user", "content": f"Question {i}: what is {13 + i}*7? Number only."}
        ]
        ttft, total, chars = _ttft(args.base, args.model, warm_msgs, args.timeout)
        results["calls"].append({"kind": f"warm{i}", "ttft_s": ttft, "total_s": total})
        print(f"warm{i}: ttft {ttft:7.2f}s total {total:7.2f}s ({chars} chars)")

    if args.metrics:
        results["metrics_after"] = _scrape_prefix_metrics(args.metrics)

    cold_ttft = results["calls"][0]["ttft_s"]
    warm_ttfts = [c["ttft_s"] for c in results["calls"][1:]]
    if warm_ttfts:
        ratio = (sum(warm_ttfts) / len(warm_ttfts)) / max(cold_ttft, 1e-9)
        results["warm_over_cold_ttft"] = ratio
        print(f"\nwarm/cold TTFT ratio: {ratio:.2%}  (lower = better caching)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"written {args.out}")


if __name__ == "__main__":
    main()
