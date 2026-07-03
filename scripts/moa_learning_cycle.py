#!/usr/bin/env python3
"""MoA learning-loop benchmark: does `hermes moa evolve` improve aggregation?

Methodology (train/held-out split is the point — see
docs/plans/moa-proxy-backlog.md item 3):

- Two disjoint task sets, category-matched (same 20 skill categories, different
  instances): TRAIN drives the evolve loop, HELDOUT is evaluation-only and its
  turns are NEVER traced, so the distilled skill cannot see them.
- Cycle protocol: baseline held-out eval with learning reset (no skill), then
  per cycle: run the train set with graded outcomes recorded into traces →
  `hermes moa evolve` (supervised digest) → snapshot the skill → paired
  held-out evals WITH the skill and WITHOUT it (file temporarily suspended).
- Leakage guard: after every distillation the skill body is scanned for
  held-out answers/task fingerprints; hits are recorded in the results JSON.

Usage (inside the moa-proxy test container, OPENROUTER_API_KEY set):
  python scripts/moa_learning_cycle.py --cycles 3 --config open-moa-flash \
      --out-dir /out/learning [--reset-learning]
  python scripts/moa_learning_cycle.py --report /out/learning/results.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moa_bench import (  # noqa: E402
    _ANSWER_INSTRUCTION,
    CONFIGS,
    _write_bench_home,
    extract_answer,
    is_correct,
)

# ---------------------------------------------------------------------------
# Task sets. Answers are ground truth, verified programmatically (see
# docs/plans/moa-learning-results-*.md for the verification script). The two
# sets share categories 1:1 but no instances — held-out measures transfer of
# aggregation heuristics, not memorization.
# ---------------------------------------------------------------------------

TRAIN_TASKS = [
    {"id": "t-bigmul", "q": "Compute exactly: 87654321 * 12345678", "a": "1082152022374638"},
    {"id": "t-digitsum", "q": "What is the sum of the decimal digits of 3^50?", "a": "144"},
    {"id": "t-zeros", "q": "How many trailing zeros does 2026! (2026 factorial) have?", "a": "505"},
    {"id": "t-divisors", "q": "How many positive divisors does 20790 have?", "a": "64"},
    {"id": "t-lettercount", "q": "How many times does the letter 's' appear in this text: she sells seashells by the seashore, and the shells she sells are surely seashells", "a": "17"},
    {"id": "t-codetrace", "q": "What does this Python 3 program print?\nfuncs = []\nfor i in range(5):\n    funcs.append(lambda x: x + i)\nprint(sum(f(10) for f in funcs))", "a": "70"},
    {"id": "t-crt", "q": "Find the smallest positive integer n with n mod 7 = 3, n mod 11 = 5, and n mod 13 = 8.", "a": "346"},
    {"id": "t-prob", "q": "Two fair six-sided dice are rolled. Given that the sum is at least 10, what is the probability that at least one die shows a 5? Answer as a fraction in lowest terms.", "a": "1/2"},
    {"id": "t-comb", "q": "How many distinct arrangements of the letters of BANANA have no two N's adjacent?", "a": "40"},
    {"id": "t-det", "q": "Compute the determinant of the matrix [[2,0,1,3],[1,4,0,2],[3,1,5,0],[0,2,1,4]].", "a": "155"},
    {"id": "t-weekday", "q": "Today is Monday. What day of the week will it be exactly 1000 days from now?", "a": "sunday"},
    {"id": "t-base", "q": "Convert the hexadecimal number 0xABCDE to decimal.", "a": "703710"},
    {"id": "t-recur", "q": "A sequence has a(1)=2, a(2)=5, and a(n)=3*a(n-1)-2*a(n-2) for n>=3. What is a(12)?", "a": "6143"},
    {"id": "t-gcdlcm", "q": "Compute lcm(84, 990) divided by gcd(84, 990).", "a": "2310"},
    {"id": "t-floorsum", "q": "Compute the sum of floor(100/k) for k = 1 to 100.", "a": "482"},
    {"id": "t-geom", "q": "How many lattice points lie strictly between (0,0) and (84,60) on the straight line segment joining them?", "a": "11"},
    {"id": "t-logic", "q": "Find the smallest three-digit number whose digits are strictly increasing and multiply to 126.", "a": "279"},
    {"id": "t-string", "q": "What does this Python 3 expression evaluate to?  ''.join(sorted('mississippi'))[4:8]", "a": "mpps"},
    {"id": "t-modexp", "q": "Compute 7^222 mod 1000.", "a": "49"},
    {"id": "t-count", "q": "How many positive integers n <= 2000 are divisible by 7 but by neither 5 nor 3?", "a": "152"},
]

HELDOUT_TASKS = [
    {"id": "h-bigmul", "q": "Compute exactly: 234567891 * 87654321", "a": "20560889214007011"},
    {"id": "h-digitsum", "q": "What is the sum of the decimal digits of 7^40?", "a": "142"},
    {"id": "h-zeros", "q": "How many trailing zeros does 555! (555 factorial) have?", "a": "137"},
    {"id": "h-divisors", "q": "How many positive divisors does 87360 have?", "a": "112"},
    {"id": "h-lettercount", "q": "How many times does the letter 'e' appear in this text: peter piper picked a peck of pickled peppers where he entered the empty theatre", "a": "18"},
    {"id": "h-codetrace", "q": "What does this Python 3 expression evaluate to?  'abcdefghij'[8:1:-2]", "a": "igec"},
    {"id": "h-crt", "q": "Find the smallest positive integer n with n mod 5 = 4, n mod 9 = 2, and n mod 11 = 7.", "a": "29"},
    {"id": "h-prob", "q": "An urn contains 5 red and 4 blue balls. Three balls are drawn without replacement. Given that at least two of the drawn balls are red, what is the probability that all three are red? Answer as a fraction in lowest terms.", "a": "1/5"},
    {"id": "h-comb", "q": "How many distinct arrangements of the letters of COFFEE have no two E's adjacent?", "a": "120"},
    {"id": "h-det", "q": "Compute the determinant of the matrix [[1,2,0,1],[0,3,1,2],[2,1,4,0],[1,0,2,5]].", "a": "80"},
    {"id": "h-weekday", "q": "Today is Saturday. What day of the week will it be exactly 500 days from now?", "a": "tuesday"},
    {"id": "h-base", "q": "Convert the decimal number 100000 to base 7.", "a": "564355"},
    {"id": "h-recur", "q": "A sequence has b(1)=1, b(2)=4, and b(n)=2*b(n-1)+3*b(n-2) for n>=3. What is b(11)?", "a": "73811"},
    {"id": "h-gcdlcm", "q": "Compute lcm(126, 168) plus gcd(126, 168).", "a": "546"},
    {"id": "h-floorsum", "q": "Compute the sum of floor(200/k) for k = 1 to 200.", "a": "1098"},
    {"id": "h-geom", "q": "How many right triangles with positive integer legs (a, b) with a <= b have hypotenuse exactly 65?", "a": "4"},
    {"id": "h-logic", "q": "What is the smallest positive multiple of 20 whose decimal digits sum to 20?", "a": "3980"},
    {"id": "h-string", "q": "What does this Python 3 expression evaluate to?  'the quick brown fox'.replace(' ', '')[::4]", "a": "tubn"},
    {"id": "h-modexp", "q": "Compute 3^2026 mod 100.", "a": "29"},
    {"id": "h-count", "q": "How many positive integers n <= 5000 are divisible by 11 but by neither 2 nor 7?", "a": "195"},
]


# ---------------------------------------------------------------------------
# Learning-state management inside the bench HERMES_HOME
# ---------------------------------------------------------------------------

def _skill_file(home: Path) -> Path:
    return home / "skills" / "moa-aggregation" / "SKILL.md"


def _trace_dir(home: Path) -> Path:
    return home / "moa-traces"


def reset_learning(home: Path) -> None:
    """Clear distilled skill + traces — the documented learning reset."""
    shutil.rmtree(home / "skills" / "moa-aggregation", ignore_errors=True)
    shutil.rmtree(_trace_dir(home), ignore_errors=True)


class suspended_skill:
    """Temporarily hide the skill file (paired without-skill eval)."""

    def __init__(self, home: Path):
        self.path = _skill_file(home)
        self.hidden = self.path.with_suffix(".md.suspended")
        self.active = False

    def __enter__(self):
        if self.path.exists():
            self.path.rename(self.hidden)
            self.active = True
        return self

    def __exit__(self, *exc):
        if self.active:
            self.hidden.rename(self.path)
        return False


# ---------------------------------------------------------------------------
# Turn runner (same machinery as moa_bench, plus graded-outcome tracing)
# ---------------------------------------------------------------------------

def run_turn(config_name: str, task: dict, *, trace: bool, session_id: str) -> dict:
    from agent.moa_loop import MoAChatCompletions, _extract_text

    started = time.time()
    result = {"config": config_name, "task": task["id"], "expected": task["a"]}
    try:
        facade = MoAChatCompletions(config_name)
        response = facade.create(
            messages=[{"role": "user", "content": task["q"] + _ANSWER_INSTRUCTION}],
            max_tokens=8000,
            # Bound each upstream call: without this a hung provider ties up a
            # worker for the SDK's 600s default x retries and stalls the phase.
            timeout=240,
        )
        text = _extract_text(response)
        ref_usage, _cost = facade.consume_reference_usage()
        agg_usage = getattr(response, "usage", None)
        extracted = extract_answer(text)
        correct = is_correct(extracted, task["a"], task.get("aliases"))
        if trace:
            # Train turns feed evolve WITH the graded outcome. Held-out turns
            # must never reach the trace dir (leakage), so the pending trace
            # is simply dropped when trace=False.
            facade.consume_and_save_trace(
                session_id=session_id,
                outcome={
                    "correct": correct,
                    "expected": task["a"],
                    "extracted": extracted,
                    "grader": "moa_learning_cycle",
                },
            )
        result.update(
            {
                "answer": extracted,
                "correct": correct,
                "latency_s": round(time.time() - started, 1),
                "ref_tokens": int(getattr(ref_usage, "input_tokens", 0) or 0)
                + int(getattr(ref_usage, "output_tokens", 0) or 0),
                "agg_tokens": int(getattr(agg_usage, "prompt_tokens", 0) or 0)
                + int(getattr(agg_usage, "completion_tokens", 0) or 0),
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


def run_set(
    config_name: str,
    tasks: list[dict],
    *,
    label: str,
    trace: bool,
    workers: int,
) -> list[dict]:
    results = []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(
            run_turn, config_name, t, trace=trace, session_id=f"learn-{label}"
        ): t["id"]
        for t in tasks
    }
    try:
        # Phase deadline: a single wedged upstream call must not stall the
        # whole run. Stragglers are recorded as phase-timeout failures; their
        # threads finish (and are discarded) in the background.
        for done, fut in enumerate(as_completed(futures, timeout=900), start=1):
            r = fut.result()
            results.append(r)
            status = "ok " if r["correct"] else ("ERR" if r.get("error") else "X  ")
            print(
                f"  [{label} {done:>2d}/{len(futures)}] {status} {r['task']:14s} "
                f"-> {str(r.get('answer'))[:32]!r} ({r.get('latency_s')}s)",
                flush=True,
            )
    except TimeoutError:
        finished_ids = {r["task"] for r in results}
        for task_id in futures.values():
            if task_id not in finished_ids:
                print(f"  [{label}] TIMEOUT {task_id} (phase deadline)", flush=True)
                results.append(
                    {
                        "config": config_name,
                        "task": task_id,
                        "expected": next(
                            t["a"] for t in tasks if t["id"] == task_id
                        ),
                        "answer": None,
                        "correct": False,
                        "error": "phase deadline exceeded",
                        "latency_s": None,
                    }
                )
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return results


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = sum(1 for r in rows if r["correct"])
    return {
        "n": n,
        "correct": ok,
        "accuracy": round(ok / n, 3) if n else None,
        "avg_latency_s": round(sum(r.get("latency_s") or 0 for r in rows) / n, 1) if n else None,
        "avg_tokens": round(
            sum((r.get("ref_tokens") or 0) + (r.get("agg_tokens") or 0) for r in rows) / n
        )
        if n
        else None,
        "errors": sum(1 for r in rows if r.get("error")),
        "wrong": sorted(r["task"] for r in rows if not r["correct"]),
    }


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def leakage_scan(skill_body: str) -> list[str]:
    """Flag held-out answers or task fingerprints appearing in the skill."""
    hits = []
    body = skill_body.lower()
    for task in HELDOUT_TASKS:
        ans = str(task["a"]).lower()
        # Standalone-token answer match (skip tiny/ambiguous answers that
        # collide with ordinary prose numbers).
        if len(ans) >= 3 and re.search(rf"(?<![\w/.]){re.escape(ans)}(?![\w/.])", body):
            hits.append(f"{task['id']}: answer {task['a']!r} appears in skill")
        fingerprint = " ".join(task["q"].lower().split()[:8])
        if len(fingerprint) > 20 and fingerprint in body:
            hits.append(f"{task['id']}: question text appears in skill")
    return hits


# ---------------------------------------------------------------------------
# Evolve invocation (in-process, same interpreter/config)
# ---------------------------------------------------------------------------

def run_evolve(home: Path, distiller: str, max_turns: int) -> None:
    from types import SimpleNamespace

    from hermes_cli.moa_evolve import cmd_moa_evolve

    rc = cmd_moa_evolve(
        SimpleNamespace(
            trace_dir=str(_trace_dir(home)),
            max_turns=max_turns,
            dry_run=False,
            model=distiller,
        )
    )
    if rc != 0:
        raise RuntimeError(f"hermes moa evolve failed (rc={rc})")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(data: dict) -> None:
    cfg = data.get("config")
    print(f"\n=== MoA learning cycles — config {cfg} ===\n")
    header = (
        f"{'phase':26s} {'acc':>7s} {'avg_lat':>8s} {'tok/q':>7s}  wrong"
    )
    print(header)
    print("-" * len(header))
    for row in data["timeline"]:
        s = row["summary"]
        print(
            f"{row['phase']:26s} {s['correct']:>3d}/{s['n']:<3d} "
            f"{s['avg_latency_s']:>7.1f}s {s['avg_tokens']:>7d}  "
            f"{', '.join(s['wrong']) or '-'}"
        )
    leaks = [l for row in data["timeline"] for l in row.get("leakage", [])]
    print(f"\nLeakage scan: {len(leaks)} hit(s)" + (":" if leaks else ""))
    for l in leaks:
        print(f"  ! {l}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--config", default="open-moa-flash", choices=list(CONFIGS))
    parser.add_argument("--home", default="/tmp/moa-learning-home")
    parser.add_argument("--out-dir", default="/tmp/moa-learning-out")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--distiller",
        default="openrouter:deepseek/deepseek-v4-pro",
        help="provider:model for the evolve distillation call",
    )
    parser.add_argument(
        "--reset-learning",
        action="store_true",
        help="Clear the distilled skill + traces in --home before starting",
    )
    parser.add_argument("--report", default=None, help="Re-print a results JSON")
    args = parser.parse_args()

    if args.report:
        print_report(json.loads(Path(args.report).read_text(encoding="utf-8")))
        return 0

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is required", file=sys.stderr)
        return 1

    home = Path(args.home)
    _write_bench_home(home)
    os.environ["HERMES_HOME"] = str(home)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.reset_learning:
        reset_learning(home)
        print(f"Learning state reset in {home}")

    timeline: list[dict] = []

    def record(phase: str, rows: list[dict], leakage: list[str] | None = None) -> None:
        timeline.append(
            {
                "phase": phase,
                "summary": summarize(rows),
                "results": rows,
                "leakage": leakage or [],
            }
        )
        (out_dir / "results.json").write_text(
            json.dumps(
                {
                    "config": args.config,
                    "distiller": args.distiller,
                    "cycles": args.cycles,
                    "train_ids": [t["id"] for t in TRAIN_TASKS],
                    "heldout_ids": [t["id"] for t in HELDOUT_TASKS],
                    "timeline": timeline,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print(f"[cycle 0] baseline held-out eval (no skill), config={args.config}")
    assert not _skill_file(home).exists(), (
        "skill file present at baseline — pass --reset-learning or clear "
        f"{_skill_file(home)}"
    )
    record("c0-heldout-baseline", run_set(
        args.config, HELDOUT_TASKS, label="c0-base", trace=False, workers=args.workers
    ))

    for cycle in range(1, args.cycles + 1):
        print(f"\n[cycle {cycle}] train pass ({len(TRAIN_TASKS)} tasks, traced+graded)")
        train_rows = run_set(
            args.config,
            TRAIN_TASKS,
            label=f"c{cycle}-train",
            trace=True,
            workers=args.workers,
        )
        record(f"c{cycle}-train", train_rows)

        print(f"[cycle {cycle}] evolve (distiller={args.distiller})")
        run_evolve(home, args.distiller, max_turns=40)
        skill_body = _skill_file(home).read_text(encoding="utf-8")
        (out_dir / f"skill-cycle{cycle}.md").write_text(skill_body, encoding="utf-8")
        leaks = leakage_scan(skill_body)
        if leaks:
            print(f"  LEAKAGE WARNINGS ({len(leaks)}):")
            for l in leaks:
                print(f"    ! {l}")

        print(f"[cycle {cycle}] held-out eval WITH skill")
        with_rows = run_set(
            args.config,
            HELDOUT_TASKS,
            label=f"c{cycle}-with",
            trace=False,
            workers=args.workers,
        )
        record(f"c{cycle}-heldout-with-skill", with_rows, leakage=leaks)

        print(f"[cycle {cycle}] held-out eval WITHOUT skill (paired)")
        with suspended_skill(home):
            without_rows = run_set(
                args.config,
                HELDOUT_TASKS,
                label=f"c{cycle}-without",
                trace=False,
                workers=args.workers,
            )
        record(f"c{cycle}-heldout-no-skill", without_rows)

    print_report(
        {
            "config": args.config,
            "timeline": timeline,
        }
    )
    print(f"Raw results: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
