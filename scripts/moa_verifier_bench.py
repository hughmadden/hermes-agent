#!/usr/bin/env python3
"""Verifier-tool experiment: does a python-exec tool close the residual gap?

Every residual reasoning miss in the earlier benches was tool-shaped (exact
big-number arithmetic, careful counting), and the evolve loop's distilled
skill *instructs* verification the models cannot do by hand. This bench
gives the acting aggregator a `python` tool (executed here in the harness,
sandboxed subprocess, no product change) and measures the lift on AIME
against the no-tool baseline from moa_hard_bench.

Tool loop: up to 4 rounds. Note that for fan-out configs each tool result
re-runs the references (that is exactly how the proxy behaves in real tool
loops), so the cost multiplier is honest.

Usage (inside the moa-proxy container):
  python scripts/moa_verifier_bench.py --out /out/verifier.json \
      [--configs kimi-solo,mixed-heavy] [--limit 60]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moa_hard_bench import (  # noqa: E402
    _ANSWER_INSTRUCTION,
    CONFIGS,
    _write_home,
    extract_answer,
    fetch_aime,
    is_correct,
)

PYTHON_TOOL = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python 3 code and return its stdout. Use it to verify "
            "arithmetic, enumerate cases, or check your candidate answer "
            "before committing to it."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
}

_TOOL_NOTE = (
    "\n\nYou have a `python` tool. VERIFY your answer with it before the "
    "final ANSWER line — run the computation or a brute-force check."
)


def run_python(code: str) -> str:
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        return out.strip()[:4000] or "(no output)"
    except subprocess.TimeoutExpired:
        return "(execution timed out after 15s)"
    except Exception as exc:  # pragma: no cover
        return f"(execution failed: {exc})"


def run_turn(config_name: str, task: dict) -> dict:
    from agent.moa_loop import MoAChatCompletions, _extract_text

    started = time.time()
    result = {"config": config_name, "task": task["id"], "expected": task["a"]}
    messages = [
        {"role": "user", "content": task["q"] + _ANSWER_INSTRUCTION + _TOOL_NOTE}
    ]
    tool_rounds = 0
    total_ref = total_agg = 0
    try:
        facade = MoAChatCompletions(config_name)
        text = ""
        # Up to 8 tool rounds, then one FORCED final round with the tool
        # removed — v1 of this harness capped at 4 and took whatever text was
        # present, and thinking models over-verified straight through the cap
        # (31/36 kimi failures were empty answers at the cap, not wrong math).
        for _round in range(10):
            final_round = _round >= 8
            if final_round and messages[-1]["role"] == "tool":
                messages = messages + [{
                    "role": "user",
                    "content": ("The python tool is no longer available. State "
                                 "your final answer NOW as a line: ANSWER: <integer>"),
                }]
            response = facade.create(
                messages=messages,
                max_tokens=16000,
                timeout=360,
                tools=None if final_round else [PYTHON_TOOL],
            )
            ref_usage, _ = facade.consume_reference_usage()
            agg_usage = getattr(response, "usage", None)
            total_ref += int(getattr(ref_usage, "input_tokens", 0) or 0) + int(
                getattr(ref_usage, "output_tokens", 0) or 0
            )
            total_agg += int(getattr(agg_usage, "prompt_tokens", 0) or 0) + int(
                getattr(agg_usage, "completion_tokens", 0) or 0
            )
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []
            text = _extract_text(response) or ""
            if not tool_calls or final_round:
                break
            tool_rounds += 1
            rendered_calls = []
            tool_msgs = []
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                name = getattr(fn, "name", "") or ""
                args_raw = getattr(fn, "arguments", "") or "{}"
                tc_id = getattr(tc, "id", None) or f"call_{tool_rounds}"
                rendered_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args_raw},
                    }
                )
                try:
                    code = json.loads(args_raw).get("code", "")
                except json.JSONDecodeError:
                    code = args_raw
                output = run_python(code) if name == "python" else f"(unknown tool {name})"
                tool_msgs.append(
                    {"role": "tool", "tool_call_id": tc_id, "content": output}
                )
            messages = messages + [
                {"role": "assistant", "content": None, "tool_calls": rendered_calls},
                *tool_msgs,
            ]
        extracted = extract_answer(text)
        result.update(
            {
                "answer": extracted,
                "correct": is_correct(extracted, task["a"]),
                "tool_rounds": tool_rounds,
                "latency_s": round(time.time() - started, 1),
                "ref_tokens": total_ref,
                "agg_tokens": total_agg,
            }
        )
    except Exception as exc:
        result.update(
            {"answer": None, "correct": False, "error": str(exc)[:300],
             "tool_rounds": tool_rounds, "latency_s": round(time.time() - started, 1)}
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-verifier.json")
    parser.add_argument("--home", default="/tmp/moa-verifier-home")
    parser.add_argument("--configs", default="kimi-solo,v4flash-solo,mixed-heavy,cere-moa")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    tasks = fetch_aime()[: args.limit]
    home = Path(args.home)
    _write_home(home)
    os.environ["HERMES_HOME"] = str(home)

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = set(config_names) - set(CONFIGS)
    if unknown:
        print(f"Unknown configs: {sorted(unknown)}", file=sys.stderr)
        return 1

    jobs = [(c, t) for c in config_names for t in tasks]
    print(f"Running {len(jobs)} verifier-tool turns...")
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
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:16s} {r['task']:12s} "
                f"rounds={r.get('tool_rounds')} -> {str(r.get('answer'))[:8]!r} "
                f"({r.get('latency_s')}s)",
                flush=True,
            )
            if done % 25 == 0:
                out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")

    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)
    print(f"\n=== AIME with python verifier tool ===\n")
    for name in config_names:
        rows = by_config.get(name, [])
        n = len(rows)
        ok = sum(1 for r in rows if r["correct"])
        lat = sum(r.get("latency_s") or 0 for r in rows) / max(n, 1)
        rounds = sum(r.get("tool_rounds") or 0 for r in rows) / max(n, 1)
        print(f"{name:16s} {ok:>3d}/{n:<4d} avg_lat={lat:.1f}s avg_tool_rounds={rounds:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
