#!/usr/bin/env python3
"""Agentic session simulator/probe for ``hermes moa serve``'s cascade mode.

Real agent sessions aren't isolated one-shot calls: they carry a growing
message history, resend the same ``tools`` schema every request, and
alternate between plain user turns and multi-round tool-call threads.
Cascade's session-aware gate should react to that shape: fresh "user" turns
(with ``tools`` present) go through the tier-0 voter gate (cascade), mid
tool-loop turns (last non-system message role "tool"/"assistant") run solo,
and a fresh user turn after a tool thread completes should REVERT to
cascade.

This drives one growing session against an OpenAI-compatible
``{base}/v1/chat/completions`` endpoint and records, per request: HTTP
status, wall latency (and TTFT under ``--stream``), the whole ``usage`` dict
verbatim (incl. ``usage.moa`` and ``usage.prompt_tokens_details.cached_tokens``),
finish_reason, whether tool_calls came back, and a content snippet or error.
It reports mode-selection/revert-check results and tool-thread completion
counts, and writes the full record set to ``--out`` as JSON.

Usage (against a real proxy):
  python scripts/moa_session_sim.py --base http://127.0.0.1:8652/v1 \
      --model moa:cascade-rlm --context-tokens 10000 --turns 8 \
      --tool-threads 2 --out /tmp/moa-session-sim.json

Self-test (no network; spins up a tiny stdlib mock server):
  python scripts/moa_session_sim.py --selftest
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

_SYS = "You are a terse assistant inside a long-running agent session."
_API_KEY_ENV: str | None = None

_TOOL_ASK = "Read the file /tmp/demo.txt and tell me its first line. Use the read_file tool."
_TOOL_RESULT_CONTENT = "line one: hello world"

_READ_FILE_PARAMS = {
    "type": "object",
    "properties": {"path": {"type": "string", "description": "Absolute path to the file to read."}},
    "required": ["path"],
}
TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the local filesystem and return its contents.",
            "parameters": _READ_FILE_PARAMS,
        },
    }
]


def _headers() -> dict:
    h = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    key = os.environ.get(_API_KEY_ENV) if _API_KEY_ENV else None
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _build_filler(target_tokens: int) -> list[dict]:
    """Deterministic fake agent history of ~target_tokens (4 chars/token)."""
    msgs = [{"role": "system", "content": _SYS}]
    per_turn_chars = 2000
    turns = max(1, (target_tokens * 4) // (per_turn_chars * 2))
    for i in range(turns):
        msgs.append(
            {"role": "user", "content": f"[step {i}] Run diagnostic {i} and summarize."}
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


def _question(i: int) -> str:
    a = 12 + (i * 5) % 37
    b = 3 + (i * 7) % 19
    return f"what is {a}*{b}? Answer with the number only."


def _classify_mode(moa: dict | None) -> str:
    """Defensively classify usage.moa as "cascade", "solo", or "unknown".

    Treated as an opaque dict per spec: search its serialized keys/values for
    "solo" vs a cascade indicator ("tier0"/"tier1"/"tier2"/"cascade").
    """
    if not isinstance(moa, dict) or not moa:
        return "unknown"
    try:
        text = json.dumps(moa).lower()
    except (TypeError, ValueError):
        return "unknown"
    if "solo" in text:
        return "solo"
    if "tier0" in text or "tier1" in text or "tier2" in text or "cascade" in text:
        return "cascade"
    return "unknown"


def _post(
    base: str, model: str, messages: list[dict], tools: list[dict], stream: bool, timeout: int
) -> dict:
    """One chat-completions call; returns a raw result dict (uncategorized)."""
    body: dict = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "temperature": 0.0,
        "max_tokens": 300,
    }
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions", data=data, headers=_headers()
    )
    t0 = time.time()
    raw: dict = dict.fromkeys(
        ("http_status", "latency_s", "ttft_s", "finish_reason", "tool_calls", "error")
    )
    raw.update(usage={}, has_tool_calls=False, content="")
    try:
        if stream:
            content_parts: list[str] = []
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None
            usage: dict = {}
            first_content_t: float | None = None
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw["http_status"] = resp.status
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8", "replace").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[5:])
                    except ValueError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                    if delta.get("content"):
                        if first_content_t is None:
                            first_content_t = time.time()
                        content_parts.append(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        blank = {"id": None, "type": "function", "function": {"name": "", "arguments": ""}}
                        slot = tool_calls_acc.setdefault(idx, blank)
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["function"]["arguments"] += fn["arguments"]
            raw["latency_s"] = time.time() - t0
            raw["ttft_s"] = (first_content_t - t0) if first_content_t is not None else None
            raw["usage"] = usage
            raw["finish_reason"] = finish_reason
            raw["content"] = "".join(content_parts)
            tool_calls_list = [tool_calls_acc[i] for i in sorted(tool_calls_acc)]
            raw["tool_calls"] = tool_calls_list or None
            raw["has_tool_calls"] = bool(tool_calls_list)
        else:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw["http_status"] = resp.status
                payload = json.loads(resp.read().decode("utf-8"))
            raw["latency_s"] = time.time() - t0
            choice = (payload.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw["usage"] = payload.get("usage") or {}
            raw["finish_reason"] = choice.get("finish_reason")
            raw["content"] = msg.get("content") or ""
            tool_calls = msg.get("tool_calls")
            raw["tool_calls"] = tool_calls
            raw["has_tool_calls"] = bool(tool_calls)
    except urllib.error.HTTPError as exc:
        raw["http_status"] = exc.code
        raw["latency_s"] = time.time() - t0
        try:
            raw["error"] = exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001
            raw["error"] = str(exc)[:500]
    except Exception as exc:  # noqa: BLE001
        raw["latency_s"] = time.time() - t0
        raw["error"] = str(exc)[:500]
    return raw


def _run_turn(
    base: str,
    model: str,
    messages: list[dict],
    tools: list[dict],
    stream: bool,
    timeout: int,
    phase: str,
) -> tuple[dict, dict]:
    raw = _post(base, model, messages, tools, stream, timeout)
    usage = raw["usage"] or {}
    moa = usage.get("moa") if isinstance(usage, dict) else None
    mode = _classify_mode(moa)
    cached = None
    ptd = usage.get("prompt_tokens_details") if isinstance(usage, dict) else None
    if isinstance(ptd, dict):
        cached = ptd.get("cached_tokens")
    rec = {
        "phase": phase,
        "http_status": raw["http_status"],
        "latency_s": round(raw["latency_s"], 3) if raw["latency_s"] is not None else None,
        "ttft_s": round(raw["ttft_s"], 3) if raw["ttft_s"] is not None else None,
        "usage": usage,
        "finish_reason": raw["finish_reason"],
        "has_tool_calls": raw["has_tool_calls"],
        "content_snippet": (raw["content"] or "")[:120],
        "error": raw["error"],
    }
    status_str = "OK " if rec["error"] is None else "ERR"
    lat = f"{rec['latency_s']:.2f}s" if rec["latency_s"] is not None else "  n/a"
    cached_str = f" cached={cached}" if cached is not None else ""
    print(f"{phase:24s} {status_str} {lat:>7s} mode={mode:8s}{cached_str}")
    return rec, raw


def run_session(
    base: str,
    model: str,
    context_tokens: int,
    turns: int,
    tool_threads: int,
    timeout: int,
    stream: bool,
    turn_delay: float = 0.0,
) -> dict:
    """Drive one growing scripted session; return records + summary."""
    messages = _build_filler(context_tokens)
    records: list[dict] = []
    q_idx = 0

    def _pace() -> None:
        # Real sessions have user/tool think-time between turns; pacing keeps
        # provider TPM-burst limits from masquerading as design limits.
        if turn_delay > 0 and records:
            time.sleep(turn_delay)

    def plain_turn(phase: str) -> dict:
        nonlocal q_idx
        _pace()
        messages.append({"role": "user", "content": _question(q_idx)})
        q_idx += 1
        rec, raw = _run_turn(base, model, messages, TOOLS, stream, timeout, phase)
        records.append(rec)
        messages.append({"role": "assistant", "content": raw["content"] or ""})
        return rec

    for i in range(turns):
        plain_turn(f"plain-init-{i}")

    tool_threads_completed = 0
    revert_results: list[str] = []
    for t in range(tool_threads):
        messages.append({"role": "user", "content": _TOOL_ASK})
        completed = False
        for round_idx in range(1, 5):
            _pace()
            rec, raw = _run_turn(
                base, model, messages, TOOLS, stream, timeout, f"tool-thread-{t}-round-{round_idx}"
            )
            records.append(rec)
            if rec["error"] is not None:
                break
            if raw["has_tool_calls"]:
                messages.append(
                    {
                        "role": "assistant",
                        "content": raw["content"] or None,
                        "tool_calls": raw["tool_calls"],
                    }
                )
                for j, tc in enumerate(raw["tool_calls"]):
                    call_id = tc.get("id") or f"call_{t}_{round_idx}_{j}"
                    messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": _TOOL_RESULT_CONTENT}
                    )
            else:
                messages.append({"role": "assistant", "content": raw["content"] or ""})
                completed = True
                break
        if completed:
            tool_threads_completed += 1

        rec = plain_turn(f"revert-check-{t}")
        moa = (rec.get("usage") or {}).get("moa") if isinstance(rec.get("usage"), dict) else None
        mode = _classify_mode(moa)
        if rec["error"] is not None:
            revert_results.append("errored")
        else:
            revert_results.append({"cascade": "passed", "solo": "failed"}.get(mode, "unknown"))

    for i in range(2):
        plain_turn(f"plain-final-{i}")

    ok = sum(1 for r in records if r["error"] is None)
    summary = {
        "turns_total": len(records),
        "turns_ok": ok,
        "turns_errored": len(records) - ok,
        "tool_threads_run": tool_threads,
        "tool_threads_completed": tool_threads_completed,
        "revert_checks": revert_results,
        "revert_checks_passed": revert_results.count("passed"),
        "revert_checks_failed": revert_results.count("failed"),
        "revert_checks_unknown": revert_results.count("unknown"),
        "revert_checks_errored": revert_results.count("errored"),
    }
    return {
        "base": base,
        "model": model,
        "context_tokens_target": context_tokens,
        "stream": stream,
        "records": records,
        "summary": summary,
    }


def _print_summary(result: dict) -> None:
    s = result["summary"]
    print()
    print(f"turns: {s['turns_ok']} ok / {s['turns_errored']} errored (of {s['turns_total']})")
    print(f"tool threads completed: {s['tool_threads_completed']}/{s['tool_threads_run']}")
    print(
        f"revert checks: {s['revert_checks_passed']} passed, {s['revert_checks_failed']} failed, "
        f"{s['revert_checks_unknown']} unknown, {s['revert_checks_errored']} errored "
        f"(of {len(s['revert_checks'])})"
    )


# --- selftest: tiny stdlib mock server, no network ---------------------


class _MockHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        last = messages[-1] if messages else {}
        last_role = last.get("role")
        last_content = last.get("content") or ""

        if last_role == "user" and "read_file tool" in last_content:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": json.dumps({"path": "/tmp/demo.txt"})},
                    }
                ],
            }
            usage = {
                "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
                "moa": {"mode": "tier0", "cascade_tier": 0, "acting_slot": "mock-voter"},
            }
            finish_reason = "tool_calls"
        elif last_role == "tool":
            msg = {"role": "assistant", "content": "The first line is: line one: hello world"}
            usage = {
                "prompt_tokens": 120, "completion_tokens": 12, "total_tokens": 132,
                "moa": {"mode": "tool-solo", "reason": "mid-loop"},
            }
            finish_reason = "stop"
        else:
            msg = {"role": "assistant", "content": "42"}
            usage = {
                "prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55,
                "moa": {"mode": "cascade", "cascade_tier": 0},
                "prompt_tokens_details": {"cached_tokens": 1234},
            }
            finish_reason = "stop"

        payload = {
            "id": "mock",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": usage,
        }
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _run_selftest() -> int:
    global _API_KEY_ENV
    _API_KEY_ENV = None
    server = http.server.HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = run_session(
            f"http://127.0.0.1:{port}/v1",
            "moa:cascade-rlm",
            context_tokens=200,
            turns=2,
            tool_threads=1,
            timeout=10,
            stream=False,
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)

    s = result["summary"]
    failures: list[str] = []
    if s["turns_total"] != 7:
        failures.append(f"expected 7 turns total (2 init + 2 tool-rounds + 1 revert + 2 final), got {s['turns_total']}")
    if s["turns_errored"] != 0:
        failures.append(f"expected 0 errors, got {s['turns_errored']}")
    if s["tool_threads_completed"] != 1:
        failures.append(f"expected 1 tool thread completed, got {s['tool_threads_completed']}")
    if s["revert_checks_passed"] != 1:
        failures.append(f"expected 1 revert check passed, got {s['revert_checks_passed']}")
    tool_call_rounds = [r for r in result["records"] if r["phase"] == "tool-thread-0-round-1"]
    if not tool_call_rounds or not tool_call_rounds[0]["has_tool_calls"]:
        failures.append("expected tool-thread-0-round-1 to carry has_tool_calls=True")
    cached_hits = [
        r
        for r in result["records"]
        if (r.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens") == 1234
    ]
    if not cached_hits:
        failures.append("expected at least one plain-turn record with cached_tokens=1234 recorded verbatim")

    if failures:
        print("SELFTEST FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print("SELFTEST OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=None, help="OpenAI-compatible API base, e.g. http://127.0.0.1:8652/v1")
    ap.add_argument("--model", default=None, help="model id, e.g. moa:cascade-rlm")
    ap.add_argument("--api-key-env", default=None)
    ap.add_argument("--context-tokens", type=int, default=10000)
    ap.add_argument("--turns", type=int, default=8, help="number of initial plain user turns")
    ap.add_argument("--tool-threads", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--turn-delay", type=float, default=0.0,
                    help="seconds to sleep between turns (real sessions have think-time; avoids conflating provider TPM-burst limits with design limits)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--stream", action="store_true", help="use stream:true and measure TTFT")
    ap.add_argument("--selftest", action="store_true", help="run against a local mock server; no network")
    args = ap.parse_args()

    if args.selftest:
        return _run_selftest()

    if not args.base or not args.model:
        ap.error("--base and --model are required unless --selftest")

    global _API_KEY_ENV
    _API_KEY_ENV = args.api_key_env

    result = run_session(
        args.base, args.model, args.context_tokens, args.turns, args.tool_threads, args.timeout, args.stream, turn_delay=args.turn_delay)
    _print_summary(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"written {args.out}")

    return 0 if result["summary"]["turns_errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
