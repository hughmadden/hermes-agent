# Tier-1 Verifier Scoring Prototype — plan (2026-08-18)

Replace/augment the cascade tier-1 path (aggregator with voter context)
with LLM-as-a-Verifier-style continuous scoring of voter candidates.
Design doc: turq-canon `docs/projects/moa-vs-llm-as-a-verifier-2026-08.md`.

## Why

- Tier-1 aggregator is the cascade's known weak link (anchored by voter
  context — architecture finding #3).
- Tier-0's discrete quorum cannot rank near-ties; today a 2-2 or 3-1
  split with a weak margin either returns a marginal winner or escalates.
- LaV evidence: continuous logprob-expectation scoring beats discrete
  judging (zero ties vs 27%; pairwise acc 73.1→77.5% at G=20).

## Verified constraints (probe 2026-08-18)

- Verifier must run on **Cerebras** (`gemma-4-31b` preferred; `gpt-oss-120b`
  works with `reasoning_effort:"low"` + ≥1k max tokens). Both return 20
  top logprobs.
- z.ai glm-5.3 strips logprobs → excluded as verifier.
- Use `-A 'Mozilla/5.0'` (Cloudflare 1010 on default Python UA).
- Scoring prompt: letter scale A–T (letters, not digits, per paper note —
  digit-adjacent tokens pollute top-logprob extraction).

## Scope (prototype)

1. **New module** `moa/verifier_scoring.py`:
   - `score_candidate(task, trajectory, criterion, model) -> float`
     — one chat call, `max_completion_tokens≈4` (gemma) / 1024 low-effort
     (gpt-oss), `logprobs=true, top_logprobs=20`, extract letter
     distribution at first content position, expectation → [0,1].
   - `rank_candidates(task, candidates, C=3, K=4) -> list[(idx, score)]`
     — mean over criteria × repeats. No pairwise tournament at N=4;
     direct absolute scores (paper's Eq. 3.1), pairwise only if scores
     tie within epsilon.
   - Criteria set for code/tool lanes: Specification / Output / Errors
     (paper §4.3). Math lane keeps existing cascade (residue already
     tiny).
2. **Cascade hook**: new preset `cascade-verify` —
   tier-0 unchanged; on no-consensus call `rank_candidates` on the ≤4
   voter candidates; return top scorer if gap-to-second ≥ `verify_margin`
   (start 0.05); else fall through to existing tier-1/tier-2.
3. **Tracing**: record per-candidate scores, criteria, K, verifier model
   in the MoA trace (`save_traces: true` for the dev container).
4. **Tests**: unit (letter extraction, expectation math, stacked-prefix
   robustness — lesson from the `FINAL:ANSWER:` bug), integration
   against a mock logprob fixture, live smoke vs moa-serve-worker (8652).

## Evaluation (order)

1. Offline replay: cached tier-1 escalation traces scored by the new
   module; compare chosen candidate vs ground-truth-correct where known.
2. Bench lane: the code/tooling track of the local benchmark repo v2
   runner — the cascade's weakest category (prior "MoA hurts code
   editing" finding) — `cascade-live` vs `cascade-verify`, ≥3 runs,
   publish bands not points.
3. If ≥ +2pp band improvement at ≤ current tier-1 cost: promote to
   moa-stable behind a config flag; keep `cascade-live` as rollback.

## Non-goals

- No pivot tournament (N=4 too small to matter).
- No changes to tier-0 or tier-2.
- No glm/z.ai verifier path.
- No RL/progress-signal work (interesting, separate).
