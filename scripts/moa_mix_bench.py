#!/usr/bin/env python3
"""Frontier/wafer/local MoA mixes on HMMT Feb 2025 (harder than AIME).

The batch's questions, one config per hypothesis:

- fable-solo / gpt55-plan-solo ......... frontier baselines (GPT-5.5 rides the
  ChatGPT plan via openai-codex — zero marginal cost with the live hermes
  auth store mounted into the container)
- council-fable ........................ do two frontier advisors (plan GPT-5.5
  + Opus) lift Fable above its solo score on problems hard enough to miss?
- wafer-fable / wafer-gpt55plan ........ wafer-speed advisors (Cerebras
  gpt-oss-120b + gemma-4-31b, ~2 s) briefing a frontier judge — "fast smart":
  near-free advice, frontier synthesis
- cere-moa ............................. all-Cerebras MoA on the harder set
- step35-solo / qwen27-solo ............ locally-runnable-class solos
  (Step-3.5-Flash claims ~97 AIME; qwen3.6-27b is a 1-GPU model)
- local-trio-stepagg ................... fully local-class MoA: v4-flash +
  qwen3.6-27b + glm-4.7-flash refs → Step-3.5-Flash aggregator
- local-trio-glm52agg .................. same refs + step as ref → GLM-5.2 agg

Task set: HMMT Feb 2025 problems whose answers are integers or simple
fractions (auto-gradable; ~20 of 30). Much harder than AIME — Fable is not
expected to saturate it.

Usage (inside the moa-proxy container; needs OPENROUTER_API_KEY,
CEREBRAS_API_KEY, and the hermes auth.json mounted for openai-codex):
  python scripts/moa_mix_bench.py --out /out/mix.json [--configs a,b]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

_ANSWER_INSTRUCTION = (
    "\n\nSolve carefully. End your reply with a final line of exactly this "
    "form and nothing after it:\nANSWER: <answer>\n"
    "where <answer> is an integer, or a fraction a/b in lowest terms."
)

CB = "custom:cerebras"
OR = "openrouter"
CODEX = "openai-codex"


def _preset(refs, agg, ref_cap=600):
    return {
        "enabled": bool(refs),
        "reference_models": [
            {"provider": p, "model": m} for p, m in (refs or [(OR, "unused/ref")])
        ],
        "aggregator": {"provider": agg[0], "model": agg[1]},
        "reference_max_tokens": ref_cap,
    }


WAFER_REFS = [(CB, "gpt-oss-120b"), (CB, "gemma-4-31b")]
LOCAL_REFS = [
    (OR, "deepseek/deepseek-v4-flash"),
    (OR, "qwen/qwen3.6-27b"),
    (OR, "z-ai/glm-4.7-flash"),
]

CONFIGS: dict[str, dict] = {
    # Frontier tier
    "fable-solo": _preset(None, (OR, "anthropic/claude-fable-5")),
    "gpt55-plan-solo": _preset(None, (CODEX, "gpt-5.5")),
    "council-fable": _preset(
        [(CODEX, "gpt-5.5"), (OR, "anthropic/claude-opus-4.8")],
        (OR, "anthropic/claude-fable-5"),
        ref_cap=2000,
    ),
    "wafer-fable": _preset(WAFER_REFS, (OR, "anthropic/claude-fable-5"), ref_cap=2000),
    "wafer-gpt55plan": _preset(WAFER_REFS, (CODEX, "gpt-5.5"), ref_cap=2000),
    "cere-moa": _preset(
        [(CB, "gpt-oss-120b"), (CB, "gemma-4-31b")], (CB, "zai-glm-4.7"), ref_cap=1500
    ),
    # Local-class tier
    "step35-solo": _preset(None, (OR, "stepfun/step-3.5-flash")),
    "qwen27-solo": _preset(None, (OR, "qwen/qwen3.6-27b")),
    "local-trio-stepagg": _preset(LOCAL_REFS, (OR, "stepfun/step-3.5-flash"), ref_cap=1500),
    "local-trio-glm52agg": _preset(
        LOCAL_REFS[:2] + [(OR, "stepfun/step-3.5-flash")],
        (OR, "z-ai/glm-5.2"),
        ref_cap=1500,
    ),
}


def fetch_hmmt() -> list[dict]:
    url = (
        "https://datasets-server.huggingface.co/rows?dataset="
        + urllib.parse.quote("MathArena/hmmt_feb_2025", safe="")
        + "&config=default&split=train&offset=0&length=100"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    tasks = []
    for r in data["rows"]:
        row = r["row"]
        ans = _normalize_answer(str(row["answer"]))
        if ans is None:
            continue  # expression answers are not auto-gradable
        tasks.append(
            {"id": f"hmmt-{row['problem_idx']}", "q": row["problem"], "a": ans}
        )
    if len(tasks) < 15:
        raise RuntimeError(f"only {len(tasks)} gradable HMMT problems")
    return tasks


def fetch_matharena(dataset: str, prefix: str, min_rows: int = 20) -> list[dict]:
    """Generic MathArena competition loader (gradable int/fraction subset)."""
    url = (
        "https://datasets-server.huggingface.co/rows?dataset="
        + urllib.parse.quote(dataset, safe="")
        + "&config=default&split=train&offset=0&length=100"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    tasks = []
    for r in data["rows"]:
        row = r["row"]
        ans = _normalize_answer(str(row["answer"]))
        if ans is None:
            continue
        tasks.append(
            {"id": f"{prefix}-{row['problem_idx']}", "q": row["problem"], "a": ans}
        )
    if len(tasks) < min_rows:
        raise RuntimeError(f"only {len(tasks)} gradable problems in {dataset}")
    return tasks


def fetch_aime26() -> list[dict]:
    """AIME 2026 (Feb 2026 — post-cutoff for all campaign models)."""
    return fetch_matharena("MathArena/aime_2026", "aime26", min_rows=25)


def fetch_hmmt26() -> list[dict]:
    """HMMT Feb 2026 (post-cutoff)."""
    return fetch_matharena("MathArena/hmmt_feb_2026", "hmmt26", min_rows=20)


def _normalize_answer(raw: str):
    """Integer or fraction → canonical string; None when not gradable."""
    s = raw.strip().strip("$").replace(" ", "").replace("\\dfrac", "\\frac")
    m = re.fullmatch(r"-?\d+", s)
    if m:
        return str(int(s))
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m:
        return str(Fraction(int(m.group(1)), int(m.group(2))))
    m = re.fullmatch(r"(-?\d+)/(\d+)", s)
    if m:
        return str(Fraction(int(m.group(1)), int(m.group(2))))
    return None


def extract_answer(text: str):
    matches = re.findall(r"(?:ANSWER|FINAL)\s*:\s*(.+)", text or "", flags=re.IGNORECASE)
    cand = matches[-1].splitlines()[0].strip() if matches else ""
    # RLM-style responses may stack terminators ("FINAL: ANSWER: 191");
    # peel any leading prefixes so grading matches the proxy's candidate
    # extraction (see hermes_cli/proxy/moa_cascade.py).
    cand = re.sub(r"^(?:(?:ANSWER|FINAL)\s*:\s*)+", "", cand, flags=re.IGNORECASE)
    if not cand:
        boxed = re.findall(r"\\boxed\{([^}]+)\}", text or "")
        cand = boxed[-1] if boxed else ""
    return _normalize_answer(cand) if cand else None


def is_correct(extracted, expected) -> bool:
    return extracted is not None and str(extracted) == str(expected)


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
            f"      reference_max_tokens: {preset['reference_max_tokens']}\n"
            f"      reference_quorum_grace: 0.5"
        )
    (home / "config.yaml").write_text(
        "providers:\n"
        "  cerebras:\n"
        "    api: https://api.cerebras.ai/v1\n"
        "    name: cerebras\n"
        "    default_model: gemma-4-31b\n"
        "moa:\n  default_preset: cere-moa\n  presets:\n" + "\n".join(blocks) + "\n",
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
            max_tokens=20000,
            timeout=int(os.environ.get("MIX_TIMEOUT", 420)),
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
    print(f"\n=== HMMT Feb 2025 (gradable subset, {n_tasks} problems) ===\n")
    header = f"{'config':24s} {'acc':>8s} {'avg_lat':>8s} {'tok/q':>7s}  errors"
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
        print(f"{name:24s} {ok:>3d}/{n:<4d} {lat:>7.1f}s {tok:>7.0f}  {errs}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-mix.json")
    parser.add_argument("--home", default="/tmp/moa-mix-home")
    parser.add_argument("--configs", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"], d.get("n_tasks", 0))
        return 0
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    tasks = fetch_hmmt()
    print(f"Fetched {len(tasks)} gradable HMMT problems")
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
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:24s} {r['task']:10s} "
                f"-> {str(r.get('answer'))[:12]!r} ({r.get('latency_s')}s)",
                flush=True,
            )
            if done % 20 == 0:
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
