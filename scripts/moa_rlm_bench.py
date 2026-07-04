#!/usr/bin/env python3
"""Agentic "RLM on wafer" bench: a reasoning-agent loop driven straight
against the Cerebras OpenAI-compatible endpoint (no MoA/aggregation) — how
far does letting a wafer-speed model iterate with a python sandbox (think,
run code, verify, repeat) get it over a single plain-prompt pass?

Unlike the other moa_*_bench scripts, this one talks to Cerebras directly
(``https://api.cerebras.ai/v1/chat/completions``) rather than the MoA facade
or the served proxy — there's no aggregation here, just one model iterating
with itself across up to ``--max-rounds`` turns, optionally emitting a single
```python fence per turn that gets executed in a sandboxed subprocess and fed
back as OUTPUT. Datasets/graders are reused from the existing hard-mode
benches: AIME via ``moa_hard_bench.fetch_aime``, HMMT via
``moa_mix_bench.fetch_hmmt``. Grading uses ``normalize_candidate`` from
``hermes_cli.proxy.moa_cascade`` so answers compare the same way cascade-mode
does (bare ints, ``a/b`` fractions in lowest terms).

For each model this also runs a single-pass ``solo:<model>`` baseline (one
plain call, same temperature, asked to end with ``FINAL: <answer>``) so the
report shows whether the agentic loop actually buys anything over just
asking once.

Requires CEREBRAS_API_KEY. Cerebras 403s the default Python user-agent, so
every request sends ``User-Agent: Mozilla/5.0`` explicitly.

Usage (inside the moa-proxy container):
  python scripts/moa_rlm_bench.py --dataset aime \
      --models gemma-4-31b,gpt-oss-120b --out /tmp/rlm-aime.json

Loop logic can be exercised with no network/API key via:
  python scripts/moa_rlm_bench.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moa_hard_bench import fetch_aime  # noqa: E402
from moa_mix_bench import fetch_hmmt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hermes_cli.proxy.moa_cascade import normalize_candidate  # noqa: E402

DATASETS = {
    "aime": fetch_aime,
    "hmmt": fetch_hmmt,
}

def fetch_gpqa():
    """GPQA Diamond (hendrydong mirror), numeric-gold subset only.

    Free-response science questions whose \\boxed{} gold answer is a pure
    number — the only rows normalize_candidate can grade reliably (unit
    strings like '10^-4 eV' are excluded). n≈25; report with that caveat.
    """
    rows = []
    for off in (0, 100):
        url = (
            "https://datasets-server.huggingface.co/rows?dataset="
            "hendrydong%2Fgpqa_diamond&config=default&split=test"
            f"&offset={off}&length=100"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            rows += json.load(resp)["rows"]
    out = []
    for i, r in enumerate(rows):
        sol = r["row"].get("solution") or ""
        m = re.findall(r"\\boxed\{([^{}]+)\}", sol)
        gold = m[-1].strip() if m else None
        if gold and re.fullmatch(r"-?\d+(\.\d+)?", gold):
            out.append({"id": f"gpqa-{i}", "q": r["row"]["problem"], "a": gold})
    return out

DATASETS["gpqa"] = fetch_gpqa


CEREBRAS_URL = "https://api.cerebras.ai/v1/chat/completions"
_REQUEST_TIMEOUT_S = 120
_MAX_ATTEMPTS = 3  # 1 try + 2 retries
_SANDBOX_TIMEOUT_S = 12
_TAIL_CHARS = 1500

CHARTER = (
    "You are a step-by-step reasoning agent. Each turn, think briefly, then "
    "EITHER emit exactly one ```python code block to compute/verify "
    "something (stdlib only, print your results) OR finish with a line "
    "'FINAL: <answer>'. Prefer computing over guessing; verify before "
    "finishing. Keep each turn short."
)

_NUDGE_NEITHER = (
    "No code and no FINAL detected. Either emit one python block or finish "
    "with FINAL: <answer>."
)
_NUDGE_FORCED_FINAL = (
    "This is your final turn. No more code. Reply with FINAL: <answer> now."
)
_SOLO_SUFFIX = "\n\nEnd with 'FINAL: <answer>'."

_FINAL_RE = re.compile(r"^\s*FINAL:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_PY_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_final(text: str) -> str | None:
    """Last ``FINAL: ...`` line in ``text``, or None if there isn't one."""
    matches = _FINAL_RE.findall(text or "")
    return matches[-1].strip() if matches else None


def run_python(code: str) -> str:
    """Execute model-generated ``code`` in an isolated subprocess.

    Mirrors ``hermes_cli.proxy.moa_cascade.run_verification``'s sandboxing
    (``-I``, empty env, scratch cwd) but returns the raw stdout+stderr tail
    instead of parsing a VERDICT line — this loop feeds output back to the
    model rather than grading it locally.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=_SANDBOX_TIMEOUT_S,
                env={},
                cwd=tmpdir,
            )
        out = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        out = f"[execution timed out after {_SANDBOX_TIMEOUT_S}s]"
    except Exception as exc:  # pragma: no cover - defensive
        out = f"[execution error: {exc}]"
    return out[-_TAIL_CHARS:]


def call_model(
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
    api_key: str | None,
    transport=None,
) -> str:
    """One chat-completions call; returns the assistant message content.

    ``transport(model, messages, max_tokens, temperature) -> str`` bypasses
    HTTP entirely (used by ``--dry-run`` to exercise the loop with a scripted
    fake model). Without a transport, POSTs to Cerebras with retry/backoff on
    429/5xx/network errors.
    """
    if transport is not None:
        return transport(model, messages, max_tokens, temperature)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        # Cerebras 403s the default urllib/python user-agent.
        "User-Agent": "Mozilla/5.0",
    }
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        req = urllib.request.Request(CEREBRAS_URL, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as exc:
            body_txt = exc.read().decode("utf-8", errors="replace")[:300]
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


def run_rlm(
    model: str,
    task: dict,
    max_rounds: int,
    api_key: str | None,
    transport=None,
) -> dict:
    """Drive the reasoning-agent loop for one problem; return a result dict.

    Rounds 1..max_rounds get the normal FINAL/fence/nudge handling. If round
    ``max_rounds`` still hasn't produced FINAL, the nudge that would follow
    it is replaced with a forced no-tool final instruction and one extra
    call is spent to capture the answer (termination-artifact lesson: a
    forced no-tool final round, spent outside the normal per-round code
    budget rather than silently truncating the transcript).
    """
    started = time.time()
    messages: list[dict] = [
        {"role": "system", "content": CHARTER},
        {"role": "user", "content": task["q"]},
    ]
    python_calls = 0
    rounds_used = 0
    final_answer: str | None = None

    for round_idx in range(1, max_rounds + 1):
        rounds_used = round_idx
        reply = call_model(model, messages, 2000, 0.4, api_key, transport=transport)
        messages.append({"role": "assistant", "content": reply})

        final_answer = _extract_final(reply)
        if final_answer is not None:
            break

        is_last = round_idx == max_rounds
        fence_match = None if is_last else _PY_FENCE_RE.search(reply)
        if fence_match:
            python_calls += 1
            output = run_python(fence_match.group(1))
            nudge = f"OUTPUT:\n{output}"
        elif is_last:
            nudge = _NUDGE_FORCED_FINAL
        else:
            nudge = _NUDGE_NEITHER
        messages.append({"role": "user", "content": nudge})

        if is_last:
            # Spend one extra call to redeem the forced no-tool nudge.
            rounds_used += 1
            reply2 = call_model(model, messages, 2000, 0.4, api_key, transport=transport)
            messages.append({"role": "assistant", "content": reply2})
            final_answer = _extract_final(reply2)
            break

    return {
        "final": final_answer,
        "rounds": rounds_used,
        "python_calls": python_calls,
        "wall_s": round(time.time() - started, 2),
    }


def run_solo(model: str, task: dict, api_key: str | None, transport=None) -> dict:
    """Single-pass baseline: one plain call, same temperature, larger budget."""
    started = time.time()
    messages = [{"role": "user", "content": task["q"] + _SOLO_SUFFIX}]
    try:
        reply = call_model(model, messages, 8000, 0.4, api_key, transport=transport)
    except Exception as exc:
        return {
            "final": None,
            "rounds": None,
            "python_calls": None,
            "wall_s": round(time.time() - started, 2),
            "error": str(exc)[:300],
        }
    return {
        "final": _extract_final(reply),
        "rounds": None,
        "python_calls": None,
        "wall_s": round(time.time() - started, 2),
    }


def run_turn(
    mode: str,
    model: str,
    task: dict,
    max_rounds: int,
    api_key: str | None,
    transport=None,
) -> dict:
    result = {"mode": mode, "model": model, "task": task["id"], "expected": task["a"]}
    try:
        r = run_rlm(model, task, max_rounds, api_key, transport=transport) if mode == "rlm" \
            else run_solo(model, task, api_key, transport=transport)
        final = r.get("final")
        result.update(
            {
                "answer": final,
                "correct": final is not None
                and normalize_candidate(final) == normalize_candidate(task["a"]),
                "wall_s": r.get("wall_s"),
                "rounds": r.get("rounds"),
                "python_calls": r.get("python_calls"),
            }
        )
        if r.get("error"):
            result["error"] = r["error"]
    except Exception as exc:
        result.update(
            {
                "answer": None,
                "correct": False,
                "error": str(exc)[:300],
                "wall_s": None,
                "rounds": None,
                "python_calls": None,
            }
        )
    return result


def print_report(results: list[dict], dataset_name: str) -> None:
    by_row: dict[str, list[dict]] = {}
    for r in results:
        by_row.setdefault(f"{r['mode']}:{r['model']}", []).append(r)
    print(f"\n=== RLM-on-wafer bench — {dataset_name} ({len(results)} turns) ===\n")
    header = f"{'row':28s} {'acc':>8s} {'mean_s':>8s} {'med_s':>8s} {'med_rounds':>10s}  py_rate"
    print(header)
    print("-" * len(header))
    for name, rows in by_row.items():
        n = len(rows)
        ok = sum(1 for r in rows if r["correct"])
        times = [r["wall_s"] for r in rows if r.get("wall_s") is not None]
        mean_t = sum(times) / len(times) if times else 0.0
        med_t = statistics.median(times) if times else 0.0
        rounds = [r["rounds"] for r in rows if r.get("rounds") is not None]
        med_rounds_str = f"{statistics.median(rounds):.1f}" if rounds else "n/a"
        py_calls = [r["python_calls"] for r in rows if r.get("python_calls") is not None]
        py_rate_str = (
            f"{100 * sum(1 for c in py_calls if c) / len(py_calls):.0f}%" if py_calls else "n/a"
        )
        print(
            f"{name:28s} {ok:>3d}/{n:<4d} {mean_t:>7.1f}s {med_t:>7.1f}s "
            f"{med_rounds_str:>10s}  {py_rate_str}"
        )
    print()


def _dry_run() -> int:
    """Exercise the loop against scripted fake transports — no network, no
    CEREBRAS_API_KEY needed. Covers: python-fence execution + OUTPUT
    continuation, the plain no-code/no-FINAL nudge, the forced no-tool final
    round, and normalize_candidate-based grading (int + fraction forms).
    """
    failures: list[str] = []

    # 1. Fence execution: round 1 emits code, round 2 sees OUTPUT and finishes.
    calls = {"n": 0}

    def transport_fence_then_final(model, messages, max_tokens, temperature):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Let's compute.\n```python\nprint(2 + 2)\n```"
        last_user = messages[-1]["content"]
        if "OUTPUT" not in last_user or "4" not in last_user:
            raise AssertionError(f"expected OUTPUT with 4, got: {last_user!r}")
        return "Verified.\nFINAL: 4"

    r1 = run_rlm(
        "fake-model", {"id": "dry-1", "q": "2+2?", "a": "4"}, max_rounds=12,
        api_key=None, transport=transport_fence_then_final,
    )
    if r1["final"] != "4" or r1["rounds"] != 2 or r1["python_calls"] != 1:
        failures.append(f"fence+final case: unexpected result {r1}")

    # 2. Neither fence nor FINAL -> plain nudge, then FINAL next round.
    calls2 = {"n": 0}

    def transport_neither_then_final(model, messages, max_tokens, temperature):
        calls2["n"] += 1
        if calls2["n"] == 1:
            return "I am thinking about this without doing anything concrete."
        last_user = messages[-1]["content"]
        if last_user != _NUDGE_NEITHER:
            raise AssertionError(f"expected the plain nudge, got: {last_user!r}")
        return "FINAL: 7"

    r2 = run_rlm(
        "fake-model", {"id": "dry-2", "q": "3+4?", "a": "7"}, max_rounds=12,
        api_key=None, transport=transport_neither_then_final,
    )
    if r2["final"] != "7" or r2["rounds"] != 2 or r2["python_calls"] != 0:
        failures.append(f"nudge case: unexpected result {r2}")

    # 3. Forced no-tool final round: model stalls past max_rounds=2, so the
    # extra forced round (round 3) must fire with the exact forced nudge.
    calls3 = {"n": 0}

    def transport_never_final(model, messages, max_tokens, temperature):
        calls3["n"] += 1
        if calls3["n"] < 3:
            return "Still thinking, no answer yet."
        last_user = messages[-1]["content"]
        if last_user != _NUDGE_FORCED_FINAL:
            raise AssertionError(f"expected forced-final nudge, got: {last_user!r}")
        return "FINAL: 9"

    r3 = run_rlm(
        "fake-model", {"id": "dry-3", "q": "4+5?", "a": "9"}, max_rounds=2,
        api_key=None, transport=transport_never_final,
    )
    if r3["final"] != "9" or r3["rounds"] != 3:
        failures.append(f"forced-final case: unexpected result {r3}")

    # 3b. Forced round still finds no FINAL -> answer stays None, no crash.
    def transport_never_ever(model, messages, max_tokens, temperature):
        return "I have no idea."

    r3b = run_rlm(
        "fake-model", {"id": "dry-3b", "q": "??", "a": "1"}, max_rounds=1,
        api_key=None, transport=transport_never_ever,
    )
    if r3b["final"] is not None or r3b["rounds"] != 2:
        failures.append(f"unresolved forced-final case: unexpected result {r3b}")

    # 4. Grading via run_turn: fraction form normalizes the same as gold.
    rt4 = run_turn(
        "rlm", "fake-model", {"id": "dry-4", "q": "1/2 in lowest terms?", "a": "1/2"},
        max_rounds=3, api_key=None, transport=lambda *a: "FINAL: 2/4",
    )
    if not rt4["correct"]:
        failures.append(f"grading case (fraction reduction): expected correct, got {rt4}")

    # 5. Solo baseline path: single call with the larger token budget.
    def transport_solo(model, messages, max_tokens, temperature):
        if max_tokens != 8000:
            raise AssertionError(f"expected solo max_tokens=8000, got {max_tokens}")
        return "Reasoning...\nFINAL: 42"

    rt5 = run_turn(
        "solo", "fake-model", {"id": "dry-5", "q": "The answer to everything.", "a": "42"},
        max_rounds=3, api_key=None, transport=transport_solo,
    )
    if not rt5["correct"] or rt5["rounds"] is not None:
        failures.append(f"solo case: expected correct with rounds=None, got {rt5}")

    if failures:
        print("DRY RUN FAILED:")
        for f in failures:
            print(f" - {f}")
        return 1
    print(
        "DRY RUN OK: fence execution, plain nudge, forced final round "
        "(resolved + unresolved), fraction grading, and solo baseline all verified."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASETS), default="aime")
    parser.add_argument(
        "--models", default="gemma-4-31b,gpt-oss-120b",
        help="comma-separated Cerebras model ids",
    )
    parser.add_argument(
        "--baseline", dest="baseline", action="store_true", default=True,
        help="also run a single-pass solo:<model> baseline per model (default on)",
    )
    parser.add_argument("--no-baseline", dest="baseline", action="store_false")
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default="/tmp/moa-rlm-bench.json")
    parser.add_argument("--report", default=None, help="re-print a report from a saved --out file")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="exercise the loop against a fake transport; no network/API key needed",
    )
    args = parser.parse_args()

    if args.dry_run:
        return _dry_run()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"], d.get("dataset", args.dataset))
        return 0

    api_key = os.environ.get("CEREBRAS_API_KEY")
    if not api_key:
        print("CEREBRAS_API_KEY is required", file=sys.stderr)
        return 1

    tasks = DATASETS[args.dataset]()
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"Fetched {len(tasks)} {args.dataset} problems")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = ["rlm", "solo"] if args.baseline else ["rlm"]

    jobs = [(mode, m, t) for m in models for mode in modes for t in tasks]
    print(
        f"Running {len(jobs)} turns ({len(models)} models x {len(modes)} modes x "
        f"{len(tasks)} tasks) against Cerebras..."
    )
    results: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_turn, mode, m, t, args.max_rounds, api_key): (mode, m, t["id"])
            for mode, m, t in jobs
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            status = "ok " if r["correct"] else ("ERR" if r.get("error") else "X  ")
            print(
                f"[{done:>3d}/{len(jobs)}] {status} {r['mode']:4s} {r['model']:16s} "
                f"{r['task']:12s} rounds={r.get('rounds')} py={r.get('python_calls')} "
                f"-> {str(r.get('answer'))[:12]!r} ({r.get('wall_s')}s)",
                flush=True,
            )
            if done % 5 == 0:
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
