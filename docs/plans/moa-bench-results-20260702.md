# MoA composition benchmark — 2026-07-02 run

Question: can combinations of open-weight models, composed as Hermes MoA
presets, match frontier commercial models on accuracy? Run via
`scripts/moa_bench.py` (16 exact-answer questions: arithmetic, number
theory, probability, code-output prediction, letter counting, exact
long multiplication) inside the moa-proxy test container over OpenRouter.
Raw per-turn data: `moa-bench-results-20260702.json`.

| config | composition | accuracy | avg latency | tokens/q |
|---|---|---|---|---|
| frontier-opus-solo | claude-opus-4.8 alone | **16/16** | 4.4 s | 283 |
| frontier-gpt55-solo | gpt-5.5 alone | **16/16** | 4.6 s | 215 |
| open-deepseek-solo | deepseek-v4-pro alone | 15/16 | 15.1 s | 992 |
| open-kimi-solo | kimi-k2.6 alone | 14/16 | 60.5 s | 1 678 |
| open-moa-heavy | refs: deepseek-v4-pro, qwen3.7-max, glm-5.2 → agg: kimi-k2.6 | **16/16** | 117.6 s | 6 007 |
| open-moa-alt | refs: kimi-k2.6, minimax-m3, deepseek-v4-flash → agg: deepseek-v4-pro | **16/16** | 92.6 s | 8 596 |
| open-moa-flash | refs: deepseek-v4-flash, qwen3.6-35b-a3b, glm-5.1 → agg: deepseek-v4-flash | **16/16** | 117.3 s | 12 759 |

## Findings

- **Every open-weight MoA combination matched the frontier solos at 16/16**,
  including `open-moa-flash`, which is built entirely from cheap flash-tier
  open models. The composition, not the size of any single member, closed
  the gap on this set.
- **The open-solo misses were all empty responses, not wrong answers** —
  deepseek-v4-pro (pycode) and kimi-k2.6 (snail, pycode) returned no
  content at all (reasoning-budget exhaustion / empty-content responses).
  The same models inside MoA presets never returned empty: the aggregator
  had reference advice to act on and the acting turn is separated from the
  deep-thinking turns. MoA here buys *robustness* as much as accuracy.
- **The price is latency and tokens**: ~20–25× slower and ~20–45× the
  token volume of a frontier solo call. Per-dollar the flash combo is still
  far cheaper than Opus per token, but per-question the frontier solos are
  both faster AND cheaper at this difficulty. MoA-of-open-weights pays off
  where open-weight solos start failing, where data must stay on
  self-hosted/open models, or where robustness against single-model
  no-answers matters.
- Caveats: n=16, frontier at ceiling (a harder set is needed to rank the
  top configs), classic-puzzle contamination is plausible for some items;
  the discriminating items in practice were pycode, snail, rcount, and
  bigmul (exact 9×9-digit multiplication — all MoA configs got it right,
  slowest turns of the run at 400–840 s).

## Follow-ups

- Harder question set (competition-tier) to separate the 16/16 cluster.
- Latency: `reference_max_tokens` tuning and a fast aggregator (see the
  Cerebras gemma-4-31b item in `moa-proxy-backlog.md`) to attack the
  ~2-minute MoA turn times.
- Skill A/B (`moa-proxy-backlog.md` item 3) once agentic traces accumulate.
