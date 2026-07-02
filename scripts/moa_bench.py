#!/usr/bin/env python3
"""MoA composition benchmark: open-weight combos vs frontier solo models.

Runs an exact-answer question set through a matrix of MoA presets (solo
baselines use a disabled preset, so the aggregator acts alone through the
identical code path) and reports per-config accuracy, latency, and token
spend. Everything goes through MoAChatCompletions — the same turn machinery
`hermes moa serve` uses — so results transfer to the proxy.

Usage (typically inside the moa-proxy test container, with OPENROUTER_API_KEY):
  python scripts/moa_bench.py --out /out/results.json [--configs a,b] [--questions 1,2,3]
  python scripts/moa_bench.py --report /out/results.json   # re-print table

The bench HERMES_HOME (default /tmp/moa-bench-home) has moa.save_traces on,
so a completed run leaves real traces for `hermes moa evolve`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Question set — exact final answers, mixed difficulty. Each item: id,
# question, expected answer, optional accepted aliases (normalized compare).
# ---------------------------------------------------------------------------

QUESTIONS = [
    {"id": "crt", "q": "A positive integer n leaves remainder 2 when divided by 3, remainder 3 when divided by 5, and remainder 2 when divided by 7. What is the smallest such n?", "a": "23"},
    {"id": "digitsum", "q": "What is the sum of the decimal digits of 2^20?", "a": "31"},
    {"id": "zeros", "q": "How many trailing zeros does 100! (100 factorial) have?", "a": "24"},
    {"id": "divisors", "q": "How many positive divisors does 360 have?", "a": "24"},
    {"id": "rcount", "q": "How many times does the letter 'r' appear in the phrase: strawberry raspberry", "a": "6"},
    {"id": "pycode", "q": "What does this Python 3 expression print?  print(sorted({'b':2,'a':1,'c':3}.items(), key=lambda kv: -kv[1])[1][0])", "a": "b"},
    {"id": "snail", "q": "A snail climbs 3 meters each day and slips back 2 meters each night. The well is 10 meters deep. On which day (day number) does it first reach the top?", "a": "8", "aliases": ["day 8"]},
    {"id": "hex", "q": "Convert the hexadecimal number 0x2F3 to decimal.", "a": "755"},
    {"id": "weekday", "q": "Today is Wednesday. What day of the week will it be exactly 100 days from now?", "a": "friday"},
    {"id": "dice", "q": "Two fair six-sided dice are rolled. Given that at least one die shows a 6, what is the probability the sum is 9? Answer as a fraction in lowest terms.", "a": "2/11"},
    {"id": "prime10", "q": "What is the 10th prime number?", "a": "29"},
    {"id": "count3", "q": "How many positive integers n with n <= 1000 are divisible by 3 but divisible by neither 2 nor 5?", "a": "134"},
    {"id": "det", "q": "Compute the determinant of the matrix [[2,1,3],[0,4,5],[1,0,6]].", "a": "41"},
    {"id": "bigmul", "q": "Compute exactly: 987654321 * 123456789", "a": "121932631112635269"},
    {"id": "anagram", "q": "How many distinct arrangements (anagrams) are there of the letters of MISSISSIPPI?", "a": "34650"},
    {"id": "coin", "q": "A fair coin is flipped 10 times. What is the probability of getting exactly 3 heads? Answer as a fraction in lowest terms.", "a": "15/128"},
]

_ANSWER_INSTRUCTION = (
    "\n\nSolve carefully. End your reply with a final line of exactly this "
    "form and nothing after it:\nANSWER: <your final answer>"
)

# ---------------------------------------------------------------------------
# Config matrix. Solo baselines are MoA presets with enabled:false — the
# aggregator acts alone through the same code path. All slots ride OpenRouter.
# ---------------------------------------------------------------------------

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
    # Frontier commercial baselines (solo)
    "frontier-opus-solo": _preset(None, "anthropic/claude-opus-4.8"),
    "frontier-gpt55-solo": _preset(None, "openai/gpt-5.5"),
    # Open-weight solo baselines
    "open-deepseek-solo": _preset(None, "deepseek/deepseek-v4-pro"),
    "open-kimi-solo": _preset(None, "moonshotai/kimi-k2.6"),
    # Open-weight MoA combinations
    "open-moa-heavy": _preset(
        ["deepseek/deepseek-v4-pro", "qwen/qwen3.7-max", "z-ai/glm-5.2"],
        "moonshotai/kimi-k2.6",
    ),
    "open-moa-alt": _preset(
        ["moonshotai/kimi-k2.6", "minimax/minimax-m3", "deepseek/deepseek-v4-flash"],
        "deepseek/deepseek-v4-pro",
    ),
    "open-moa-flash": _preset(
        ["deepseek/deepseek-v4-flash", "qwen/qwen3.6-35b-a3b", "z-ai/glm-5.1"],
        "deepseek/deepseek-v4-flash",
    ),
    # Deliberately small/fallible combo (single-consumer-GPU-sized models):
    # headroom for the learning loop to show quality lift, and a proxy for the
    # "independent per-GPU small models" composition studied in
    # docs/plans/multi-gpu-moa-plan.md.
    "open-moa-nano": _preset(
        ["meta-llama/llama-3.1-8b-instruct", "mistralai/ministral-8b-2512", "google/gemma-3-12b-it"],
        "google/gemma-3-27b-it",
    ),
    "open-nano-solo": _preset(None, "google/gemma-3-27b-it"),
}


def _write_bench_home(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    presets_yaml = []
    for name, preset in CONFIGS.items():
        refs = "\n".join(
            f"        - provider: {s['provider']}\n          model: {s['model']}"
            for s in preset["reference_models"]
        )
        presets_yaml.append(
            f"    {name}:\n"
            f"      enabled: {str(preset['enabled']).lower()}\n"
            f"      reference_models:\n{refs}\n"
            f"      aggregator:\n"
            f"        provider: {preset['aggregator']['provider']}\n"
            f"        model: {preset['aggregator']['model']}\n"
            f"      reference_max_tokens: {preset['reference_max_tokens']}"
        )
    (home / "config.yaml").write_text(
        "moa:\n"
        "  save_traces: true\n"
        "  default_preset: open-moa-heavy\n"
        "  presets:\n" + "\n".join(presets_yaml) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Answer extraction / grading
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    s = str(text or "").strip().lower()
    s = s.strip(" .!*`'\"()[]")
    s = s.replace(",", "").replace("$", "").replace("\\", "")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_answer(text: str) -> str:
    matches = re.findall(r"answer\s*:\s*(.+)", text or "", flags=re.IGNORECASE)
    if matches:
        return matches[-1].splitlines()[0]
    # Fallback: last non-empty line.
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def is_correct(extracted: str, expected: str, aliases: list[str] | None = None) -> bool:
    got = _normalize(extracted)
    want_all = [_normalize(expected)] + [_normalize(a) for a in (aliases or [])]
    for want in want_all:
        if got == want:
            return True
        # Numeric equivalence (e.g. "8" vs "day 8" handled via aliases; "8.0").
        try:
            if float(got) == float(want):
                return True
        except ValueError:
            pass
        # The wanted token appearing as the whole trailing value ("friday" in
        # "friday.") is already handled by normalization; also accept the
        # expected answer as a standalone word inside a short extraction.
        if len(got) <= 40 and re.search(rf"(?<![\w/]){re.escape(want)}(?![\w/])", got):
            return True
    return False


# ---------------------------------------------------------------------------
# Turn runner
# ---------------------------------------------------------------------------

def run_turn(config_name: str, question: dict) -> dict:
    from agent.moa_loop import MoAChatCompletions, _extract_text

    started = time.time()
    result = {
        "config": config_name,
        "question": question["id"],
        "expected": question["a"],
    }
    try:
        facade = MoAChatCompletions(config_name)
        response = facade.create(
            messages=[{"role": "user", "content": question["q"] + _ANSWER_INSTRUCTION}],
            max_tokens=8000,
        )
        text = _extract_text(response)
        ref_usage, _cost = facade.consume_reference_usage()
        agg_usage = getattr(response, "usage", None)
        facade.consume_and_save_trace(session_id=f"bench-{config_name}")
        extracted = extract_answer(text)
        result.update(
            {
                "answer": extracted,
                "correct": is_correct(extracted, question["a"], question.get("aliases")),
                "latency_s": round(time.time() - started, 1),
                "ref_tokens": int(getattr(ref_usage, "input_tokens", 0) or 0)
                + int(getattr(ref_usage, "output_tokens", 0) or 0),
                "agg_tokens": int(getattr(agg_usage, "prompt_tokens", 0) or 0)
                + int(getattr(agg_usage, "completion_tokens", 0) or 0),
                "raw_tail": (text or "")[-200:],
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


def print_report(results: list[dict]) -> None:
    by_config: dict[str, list[dict]] = {}
    for r in results:
        by_config.setdefault(r["config"], []).append(r)
    total_q = len({r["question"] for r in results})
    print(f"\n=== MoA composition benchmark — {total_q} questions ===\n")
    header = f"{'config':24s} {'acc':>7s} {'avg_lat':>8s} {'tokens/q':>9s}  wrong/error"
    print(header)
    print("-" * len(header))
    for name in CONFIGS:
        rows = by_config.get(name)
        if not rows:
            continue
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        lat = sum(r.get("latency_s") or 0 for r in rows) / n
        toks = sum((r.get("ref_tokens") or 0) + (r.get("agg_tokens") or 0) for r in rows) / n
        wrong = [
            r["question"] + ("(err)" if r.get("error") else "")
            for r in rows
            if not r["correct"]
        ]
        print(
            f"{name:24s} {correct:>3d}/{n:<3d} {lat:>7.1f}s {toks:>9.0f}  {', '.join(wrong) or '-'}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/tmp/moa-bench-results.json")
    parser.add_argument("--home", default="/tmp/moa-bench-home")
    parser.add_argument("--configs", default=None, help="Comma-separated config subset")
    parser.add_argument("--questions", default=None, help="Comma-separated question-id subset")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--report", default=None, help="Just re-print the table from a results JSON")
    args = parser.parse_args()

    if args.report:
        print_report(json.loads(Path(args.report).read_text(encoding="utf-8"))["results"])
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    home = Path(args.home)
    _write_bench_home(home)
    os.environ["HERMES_HOME"] = str(home)

    config_names = list(CONFIGS)
    if args.configs:
        config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = set(config_names) - set(CONFIGS)
        if unknown:
            print(f"Unknown configs: {sorted(unknown)}", file=sys.stderr)
            return 1
    questions = QUESTIONS
    if args.questions:
        wanted = {q.strip() for q in args.questions.split(",")}
        questions = [q for q in QUESTIONS if q["id"] in wanted]

    jobs = [(c, q) for c in config_names for q in questions]
    print(
        f"Running {len(jobs)} turns ({len(config_names)} configs x "
        f"{len(questions)} questions, workers={args.workers})..."
    )
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_turn, c, q): (c, q["id"]) for c, q in jobs}
        for done, future in enumerate(as_completed(futures), start=1):
            r = future.result()
            results.append(r)
            status = "ok " if r["correct"] else ("ERR" if r.get("error") else "X  ")
            print(
                f"[{done:>3d}/{len(jobs)}] {status} {r['config']:24s} "
                f"{r['question']:10s} -> {str(r.get('answer'))[:40]!r} "
                f"({r.get('latency_s')}s)"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"questions": len(questions), "configs": config_names, "results": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\nResults written to {out}")
    print_report(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
