#!/usr/bin/env python3
"""Judge-gate freeform eval: blind pairwise A/B over a live `hermes moa serve`.

Addendum v1.1 (docs/plans/moa-cascade-spec.md) adds `cascade.gate: "judge"` so
freeform (non-exact-match) traffic can still resolve at tier 0 when voters
substantively agree. There is no gold answer for freeform prompts, so this
eval doesn't grade correctness — it drives the judge-gated cascade
(`--model`) and a comparison model (`--baseline`) over the SAME 30 embedded
prompts, then has a third model (`--grader`) blind-judge each pair (shuffled
A/B, no model names) as "A" / "B" / "TIE".

Reports: win/tie/loss (from --model's perspective), tier-0 rate + gate_used
histogram for both sides, median latency for both sides.

Start the server first:
  hermes moa serve --port 8652

Usage:
  python scripts/moa_judge_gate_eval.py --model moa:cascade-judge \
      --baseline moa:cere-moa --grader moa:fable-lane \
      --out /tmp/judge-gate-eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_REQUEST_TIMEOUT = 180

# 30 freeform prompts: general knowledge, explanations, comparisons, how-tos.
# Mixed difficulty on purpose; no exact-answer format (this is a quality A/B,
# not a correctness bench).
PROMPTS: list[dict] = [
    {"id": "p01", "q": "Explain why the sky is blue in a way a curious 10-year-old would understand."},
    {"id": "p02", "q": "What's the difference between TCP and UDP, and when would you pick one over the other?"},
    {"id": "p03", "q": "How do I make a good pour-over coffee at home? Walk me through it."},
    {"id": "p04", "q": "Compare Python and Go for building a small backend service. What tradeoffs matter most?"},
    {"id": "p05", "q": "What caused the fall of the Roman Empire? Give the main factors historians point to."},
    {"id": "p06", "q": "Explain the difference between weather and climate."},
    {"id": "p07", "q": "How does a car's internal combustion engine actually convert fuel into motion?"},
    {"id": "p08", "q": "What's the difference between a virus and a bacterium, medically speaking?"},
    {"id": "p09", "q": "How should I go about learning to play the guitar as a complete beginner?"},
    {"id": "p10", "q": "Explain how vaccines train the immune system."},
    {"id": "p11", "q": "What are the key differences between a stock and a bond as investments?"},
    {"id": "p12", "q": "Why do we have leap years, and why isn't it just every 4 years without exception?"},
    {"id": "p13", "q": "How does compound interest work, and why does starting early matter so much?"},
    {"id": "p14", "q": "Explain the difference between machine learning and traditional rule-based programming."},
    {"id": "p15", "q": "What's the best way to organize a home network with multiple devices and a NAS?"},
    {"id": "p16", "q": "Compare renewable energy sources (solar, wind, hydro) in terms of cost and reliability."},
    {"id": "p17", "q": "How do I choose a good running shoe if I'm training for a first 10k?"},
    {"id": "p18", "q": "Explain what a black hole is and how we know they exist if light can't escape them."},
    {"id": "p19", "q": "What's the difference between civil law and common law legal systems?"},
    {"id": "p20", "q": "How does noise-cancelling headphone technology actually work?"},
    {"id": "p21", "q": "Explain the causes of inflation in simple terms, and what a central bank can do about it."},
    {"id": "p22", "q": "What's the difference between a firm's revenue, profit, and cash flow?"},
    {"id": "p23", "q": "How do I properly season and care for a cast iron skillet?"},
    {"id": "p24", "q": "Explain how GPS determines your location on Earth."},
    {"id": "p25", "q": "Compare renting versus buying a home financially, in general terms."},
    {"id": "p26", "q": "What's the difference between a nutritionist, a dietitian, and a personal trainer?"},
    {"id": "p27", "q": "Explain why glass is transparent but most solids are not."},
    {"id": "p28", "q": "How should a beginner start learning to cook Thai food at home?"},
    {"id": "p29", "q": "What's the difference between DNA and RNA, and why does it matter biologically?"},
    {"id": "p30", "q": "Explain the basic idea behind how encryption keeps data private online."},
]

_GRADER_SYSTEM = (
    "You are a strict, impartial judge comparing two candidate answers to the "
    "same question. Judge on correctness, clarity, and helpfulness. Reply "
    "with exactly one word: A if Answer A is better, B if Answer B is "
    "better, or TIE if they are of roughly equal quality. Do not explain."
)


def post_chat(base: str, model: str, messages: list[dict], max_tokens: int, temperature: float | None = None) -> dict:
    url = base.rstrip("/") + "/chat/completions"
    payload: dict = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    if temperature is not None:
        payload["temperature"] = temperature
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


def _timed_answer(base: str, model: str, question: str, max_tokens: int) -> dict:
    started = time.time()
    body = post_chat(base, model, [{"role": "user", "content": question}], max_tokens)
    latency = round(time.time() - started, 1)
    message = body["choices"][0]["message"]
    text = message.get("content") or ""
    usage = body.get("usage") or {}
    cascade = (usage.get("moa") or {}).get("cascade") or {}
    return {
        "text": text,
        "latency_s": latency,
        "total_tokens": usage.get("total_tokens"),
        "tier": cascade.get("tier"),
        "gate_used": cascade.get("gate_used"),
    }


def parse_verdict(raw: str) -> str | None:
    """Tolerantly parse a grader reply into "A" / "B" / "TIE" / None."""
    text = (raw or "").strip()
    if not text:
        return None
    upper = text.upper()
    if re.search(r"\bTIE\b", upper):
        return "TIE"
    m_a = re.search(r"\bA\b", upper)
    m_b = re.search(r"\bB\b", upper)
    if m_a and not m_b:
        return "A"
    if m_b and not m_a:
        return "B"
    if m_a and m_b:
        return "A" if m_a.start() < m_b.start() else "B"
    if upper[0] == "A":
        return "A"
    if upper[0] == "B":
        return "B"
    return None


def run_prompt(
    base: str,
    model: str,
    baseline: str,
    grader: str,
    prompt: dict,
    answer_max_tokens: int,
    seed: int,
) -> dict:
    result: dict = {"id": prompt["id"], "question": prompt["q"]}
    try:
        result["model"] = _timed_answer(base, model, prompt["q"], answer_max_tokens)
    except Exception as exc:
        result["model"] = {"error": str(exc)[:300]}
    try:
        result["baseline"] = _timed_answer(base, baseline, prompt["q"], answer_max_tokens)
    except Exception as exc:
        result["baseline"] = {"error": str(exc)[:300]}

    model_text = result["model"].get("text")
    baseline_text = result["baseline"].get("text")
    if not model_text or not baseline_text:
        result["verdict"] = None
        result["grader_error"] = "missing answer(s), skipped grading"
        return result

    rng = random.Random(f"{seed}:{prompt['id']}")
    model_is_a = rng.random() < 0.5
    answer_a = model_text if model_is_a else baseline_text
    answer_b = baseline_text if model_is_a else model_text
    result["shuffle"] = {"model_slot": "A" if model_is_a else "B"}

    user_msg = (
        f"Question:\n{prompt['q']}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}"
    )
    try:
        grader_body = post_chat(
            base,
            grader,
            [
                {"role": "system", "content": _GRADER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        grader_raw = grader_body["choices"][0]["message"].get("content") or ""
        result["grader_raw"] = grader_raw
        slot_verdict = parse_verdict(grader_raw)
        if slot_verdict is None:
            result["verdict"] = None
        elif slot_verdict == "TIE":
            result["verdict"] = "TIE"
        else:
            picked_model = model_is_a if slot_verdict == "A" else (not model_is_a)
            result["verdict"] = "model" if picked_model else "baseline"
    except Exception as exc:
        result["verdict"] = None
        result["grader_error"] = str(exc)[:300]
    return result


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def print_report(results: list[dict], model: str, baseline: str) -> None:
    print(f"\n=== Judge-gate freeform eval ({len(results)} prompts) ===")
    print(f"model:    {model}")
    print(f"baseline: {baseline}\n")

    wins = sum(1 for r in results if r.get("verdict") == "model")
    losses = sum(1 for r in results if r.get("verdict") == "baseline")
    ties = sum(1 for r in results if r.get("verdict") == "TIE")
    unknown = sum(1 for r in results if r.get("verdict") is None)
    print(f"win/tie/loss (model vs baseline): {wins}/{ties}/{losses}  (undetermined: {unknown})")

    for label in ("model", "baseline"):
        side = [r[label] for r in results if isinstance(r.get(label), dict) and "error" not in r[label]]
        n = len(side)
        tier0 = sum(1 for s in side if s.get("tier") == 0)
        gate_hist: dict[str, int] = {}
        for s in side:
            key = "n/a" if s.get("gate_used") is None else str(s["gate_used"])
            gate_hist[key] = gate_hist.get(key, 0) + 1
        lats = [s["latency_s"] for s in side if s.get("latency_s") is not None]
        print(
            f"  {label:8s} n={n:<3d} tier0={tier0}/{n} "
            f"median_lat={_median(lats):.1f}s  gate_used={gate_hist}"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8652/v1")
    parser.add_argument("--model", required=True, help="judge-gated cascade model id, e.g. moa:cascade-judge")
    parser.add_argument("--baseline", default="moa:cere-moa", help="comparison model id")
    parser.add_argument(
        "--grader",
        default="moa:fable-lane",
        help="blind pairwise grader model id, reachable at --base (must be a serve preset)",
    )
    parser.add_argument("--max-tokens", type=int, default=1500, help="max_tokens for model/baseline answers")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0, help="shuffle seed for blind A/B ordering")
    parser.add_argument("--out", default="/tmp/moa-judge-gate-eval.json")
    parser.add_argument("--report", default=None, help="print a report from a previously written --out file")
    args = parser.parse_args()

    if args.report:
        d = json.loads(Path(args.report).read_text(encoding="utf-8"))
        print_report(d["results"], d.get("model", args.model), d.get("baseline", args.baseline))
        return 0

    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS
    print(f"Running {len(prompts)} prompts against {args.base} (model={args.model}, baseline={args.baseline}, grader={args.grader})...")

    results: list[dict] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def _write() -> None:
        out.write_text(
            json.dumps(
                {"model": args.model, "baseline": args.baseline, "grader": args.grader, "results": results},
                indent=2,
            ),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_prompt, args.base, args.model, args.baseline, args.grader, p, args.max_tokens, args.seed
            ): p["id"]
            for p in prompts
        }
        for done, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            results.append(r)
            verdict = r.get("verdict") or "?"
            m_tier = r.get("model", {}).get("tier")
            m_gate = r.get("model", {}).get("gate_used")
            print(
                f"[{done:>3d}/{len(prompts)}] {r['id']:5s} verdict={verdict:8s} "
                f"model_tier={m_tier} model_gate={m_gate}",
                flush=True,
            )
            if done % 5 == 0:
                _write()

    _write()
    print(f"\nResults written to {out}")
    print_report(results, args.model, args.baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
