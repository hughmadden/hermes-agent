#!/usr/bin/env python3
"""Prompt-caching measurement for agentic workloads, direct against OpenRouter.

Agentic sessions grow a large *stable* prefix (system prompt + conversation
history) with a small delta appended each turn. Provider-side prompt caching
should make repeated prefixes cheaper and faster on turn 2+. This bench
builds a synthetic, byte-stable transcript that mimics that shape and fires
4 sequential calls per model straight at
``https://openrouter.ai/api/v1/chat/completions`` (no hermes/MoA plumbing in
the loop) to see which providers actually discount/speed up the repeat:

  call 1: system + history(~--prefix-tokens) + question A          (cold)
  call 2: system + history + turn_1        + question B            (warm)
  call 3: system + history + turn_1..2     + question C            (warm)
  call 4: system + history + turn_1..3     + question D             (warm)

Each call's prefix is a strict byte-for-byte extension of the previous
call's messages (same system prompt, same history text, same fake tool
outputs) so any provider-side prefix cache has a real matching prefix to
hit. Content is deterministic (numbered fake tool outputs, no randomness)
so reruns are directly comparable.

Usage (inside the moa-proxy test container, OPENROUTER_API_KEY set):
  python scripts/moa_cache_bench.py \\
      --models deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6 \\
      --out /out/or-cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://openrouter.ai/api/v1/chat/completions"
_REQUEST_TIMEOUT_S = 90
_MAX_ATTEMPTS = 2  # 1 retry on 5xx/429

_CHARS_PER_TOKEN = 4  # deterministic estimator, matches task spec

_SYSTEM_PROMPT = (
    "You are an autonomous coding agent working inside a large software "
    "repository. You have access to tools for reading files, searching "
    "code, running shell commands, and editing files. Follow the user's "
    "instructions precisely, prefer minimal surgical changes, match the "
    "existing code style, and never invent APIs that do not exist in the "
    "codebase. When you are not sure about something, say so explicitly "
    "rather than guessing. Always explain your reasoning briefly before "
    "taking an action, and summarize what you changed at the end of a "
    "task. Safety rules: never delete files outside the working "
    "directory, never exfiltrate secrets, never run destructive git "
    "commands without explicit confirmation, and always prefer additive, "
    "reversible changes over irreversible ones. This system prompt "
    "describes your operating context for the remainder of the session; "
    "treat it as a stable contract that does not change between turns. "
) * 2  # ~2k chars


def _fake_tool_turn(index: int) -> str:
    """One deterministic fake history 'turn' (~600 chars): a tool call + output."""
    lines = [f"[turn {index}] tool_call: read_file(path='src/module_{index:03d}.py')"]
    for j in range(8):
        lines.append(
            f"  L{j + 1:03d}: def helper_{index:03d}_{j}(x): return x * {index} + {j}  "
            f"# deterministic filler line {index}.{j}"
        )
    lines.append(
        f"[turn {index}] tool_result: read {len(lines) - 1} lines from module_{index:03d}.py, no errors"
    )
    return "\n".join(lines) + "\n"


def build_history(prefix_tokens: int) -> list[str]:
    """Build a list of deterministic history-turn strings totaling ~prefix_tokens.

    Excludes the system prompt from the token budget (system is additional,
    matching a real agent transcript where system + history both count
    toward context but are tracked/built separately here).
    """
    budget_chars = prefix_tokens * _CHARS_PER_TOKEN
    turns: list[str] = []
    total = 0
    i = 1
    while total < budget_chars:
        t = _fake_tool_turn(i)
        turns.append(t)
        total += len(t)
        i += 1
    return turns


_QUESTIONS = [
    "Based on the files you've read so far, what does module_001 do and is there a bug in it? Answer in 2 sentences.",
    "Given the same context, name one refactor that would reduce duplication across the modules you've seen. 2 sentences.",
    "Still with the same context: which module looks most likely to have an off-by-one error, and why? 2 sentences.",
    "Final question on this context: summarize the overall pattern across all modules read so far in 2 sentences.",
]


def build_messages(history_turns: list[str], n_turns_included: int, question: str) -> list[dict]:
    history_text = "".join(history_turns[:n_turns_included])
    user_content = (
        "Here is the tool-call history so far in this session:\n\n"
        f"{history_text}\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def post_chat(model: str, messages: list[dict], api_key: str, max_tokens: int = 200) -> tuple[dict, float]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        started = time.time()
        req = urllib.request.Request(API_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body, time.time() - started
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", errors="replace")[:500]
            if exc.code == 429 or exc.code >= 500:
                last_exc = RuntimeError(f"HTTP {exc.code}: {body_txt}")
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"HTTP {exc.code}: {body_txt}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            time.sleep(2**attempt)
            continue
    raise RuntimeError(f"request failed after {_MAX_ATTEMPTS} attempts: {last_exc}")


def run_model(model: str, history_turns: list[str], api_key: str, out: Path, all_results: list[dict]) -> list[dict]:
    """Fire the 4-call sequence for one model.

    Call 0 (cold) uses the base history as-is. Calls 1-3 (warm) each append
    exactly one more small, deterministic "extra turn" on top of the
    previous call's full turn list — so call k's prefix is a strict
    byte-for-byte extension of call k-1's prefix, mimicking a session that
    keeps growing by one turn each round.
    """
    calls: list[dict] = []
    running_turns = list(history_turns)
    for call_idx in range(4):
        question = _QUESTIONS[call_idx]
        if call_idx > 0:
            running_turns.append(
                f"[extra-turn appended before call {call_idx + 1}] deterministic filler content {call_idx}.\n"
            )
        messages = build_messages(running_turns, len(running_turns), question)
        record = {
            "model": model,
            "call_index": call_idx,
            "kind": "cold" if call_idx == 0 else "warm",
        }
        try:
            body, latency = post_chat(model, messages, api_key)
            usage = body.get("usage") or {}
            choices = body.get("choices") or [{}]
            message = choices[0].get("message") or {}
            # Reasoning models can spend the whole max_tokens budget on
            # reasoning_details and leave content null (finish_reason
            # "length") — .get(..., "") only supplies a default when the key
            # is *absent*, not when it's present-but-None, so guard with `or`.
            content = message.get("content") or ""
            record.update(
                {
                    "latency_s": round(latency, 3),
                    "usage": usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                    "cache_discount": usage.get("cache_discount"),
                    "answer_preview": content[:120],
                }
            )
        except Exception as exc:
            record.update({"error": str(exc)[:300]})
        calls.append(record)
        all_results.append(record)
        out.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
        status = "ERR" if record.get("error") else "ok "
        print(
            f"  [{model}] call {call_idx + 1}/4 ({record['kind']:4s}) {status} "
            f"lat={record.get('latency_s')}s prompt_tok={record.get('prompt_tokens')} "
            f"cached_tok={record.get('cached_tokens')}",
            flush=True,
        )
    return calls


def print_report(results: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r["model"], []).append(r)

    print("\n=== Prompt-cache bench ===\n")
    header = f"{'model':32s} {'cold_s':>8s} {'warm_s':>8s} {'cached_frac':>12s}  notes"
    print(header)
    print("-" * len(header))
    for model, rows in by_model.items():
        rows = sorted(rows, key=lambda r: r["call_index"])
        cold = next((r for r in rows if r["kind"] == "cold"), None)
        warm = [r for r in rows if r["kind"] == "warm"]
        cold_lat = cold.get("latency_s") if cold else None
        warm_lats = [r["latency_s"] for r in warm if r.get("latency_s") is not None]
        mean_warm_lat = statistics.mean(warm_lats) if warm_lats else None

        fracs = []
        for r in warm:
            pt = r.get("prompt_tokens")
            ct = r.get("cached_tokens")
            if pt and ct is not None:
                fracs.append(ct / pt)
        mean_frac = statistics.mean(fracs) if fracs else None

        # which usage fields actually showed up, across all 4 calls
        fields_seen: set[str] = set()
        for r in rows:
            usage = r.get("usage") or {}
            fields_seen.update(usage.keys())
            details = usage.get("prompt_tokens_details") or {}
            fields_seen.update(f"prompt_tokens_details.{k}" for k in details.keys())
        errors = sum(1 for r in rows if r.get("error"))

        notes_parts = []
        if fields_seen:
            notes_parts.append("fields=" + ",".join(sorted(fields_seen)))
        if errors:
            notes_parts.append(f"{errors}/4 calls errored")
        notes = "; ".join(notes_parts) if notes_parts else "no usage data"

        cold_s = f"{cold_lat:.2f}" if cold_lat is not None else "n/a"
        warm_s = f"{mean_warm_lat:.2f}" if mean_warm_lat is not None else "n/a"
        frac_s = f"{mean_frac * 100:.1f}%" if mean_frac is not None else "n/a"
        print(f"{model:32s} {cold_s:>8s} {warm_s:>8s} {frac_s:>12s}  {notes}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        required=True,
        help="comma-separated OpenRouter model ids, e.g. deepseek/deepseek-v4-pro,moonshotai/kimi-k2.6",
    )
    parser.add_argument("--prefix-tokens", type=int, default=30000, help="target prefix size in tokens (4 chars/token estimate)")
    parser.add_argument("--out", default="/tmp/moa-cache-bench.json")
    parser.add_argument("--report", default=None, help="skip the run; print a report from a saved JSON file")
    args = parser.parse_args()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"])
        return 0

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    history_turns = build_history(args.prefix_tokens)
    approx_prefix_chars = len(_SYSTEM_PROMPT) + sum(len(t) for t in history_turns)
    print(
        f"Built synthetic prefix: {len(history_turns)} history turns, "
        f"~{approx_prefix_chars} chars (~{approx_prefix_chars // _CHARS_PER_TOKEN} tokens incl. system)"
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    print(f"Running {len(models)} models x 4 sequential calls against {API_URL}...")
    for model in models:
        print(f"\n== {model} ==")
        try:
            run_model(model, history_turns, api_key, out, all_results)
        except Exception as exc:
            print(f"  [{model}] FATAL: {exc}", file=sys.stderr)
            all_results.append({"model": model, "call_index": -1, "kind": "fatal", "error": str(exc)[:300]})
            out.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")

    out.write_text(json.dumps({"results": all_results}, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")
    print_report(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
