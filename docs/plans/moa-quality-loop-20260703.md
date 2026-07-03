# Quality-improvement loop — 2026-07-03

Hypothesize → test → ship → repeat, one measured change per iteration.
Raw data: the `moa-*-20260703.json` files in this directory; public report:
the Turq share (services.turquoisebay.ai/share/moa-next-phase/).

| # | Hypothesis | Result | Verdict / action |
|---|---|---|---|
| 1 | Routing coding to a solo lane removes the MoA editing penalty | moa:auto v2 (coding→kimi lane): 66.7% pass@1 @235 s vs v1 (coding→fan-out) 63.3% @527 s; 71/73 turns routed correctly | Confirmed — kimi-parity within single-run noise; solo lanes made routable (disabled preset + route block) |
| 2 | Self-MoA (best model ×3) beats mixed references | kimi-solo, kimi-selfmoa3, mixed-heavy ALL 40/40; solo used 4× fewer tokens | No signal at ceiling — composition only adds cost once the model saturates the task family |
| 3a | A classifier can pre-route "routine" work to a wafer-cheap lane | Cerebras GLM-4.7 cheap lane: 30% pass@1 on work the classifier correctly called routine | Refuted — never guess difficulty up-front; the cheap lane must be validated per format |
| 3b | Verification-gated escalation: cheap first, frontier on test failure | kimi 22/30 + Fable(with failing tests) 8/8 = 30/30 composite; Fable-solo baseline 26/30 / 30/30 @13.6 s | Confirmed — full Fable pass@2 quality, Fable on 27% of exercises |
| 4 | Escalation replicates on a hard agentic benchmark | SWE-bench Lite 25: kimi 19/25 resolved + Fable 6/6 on the remainder = 25/25 composite, Fable on 24% of instances | Confirmed at scale |
| 5 | Productize: failure-gated escalation lane in the router | `router.escalation` shipped: sticky conversation re-routes once when the latest tool/user feedback matches failure patterns; 6 unit tests | Shipped |
| 6 | Live end-to-end: aider drives moa:auto with the escalation lane | 83.3% pass@1 / 100% pass@2 @36.6 s/case; only 3/30 exercises ever reached Fable | Confirmed in production shape |
| 7 | Capping reference advice cuts fan-out latency for free | cap 1500→600: −17% latency, −15% tokens, same accuracy; cap 300: no further gain | Adopt 600 as guidance; latency is provider-think-time-bound below that |
| 8 | A wafer-speed model can be the cheap coding lane | Cerebras gemma-4-31b on aider: 2.2 s/case, 10% pass@1 (50% pass@2) | Refuted — wafer models own classify/self/reasoning lanes, not editing |
| 9 | Inverted MoA (draft→review→revise) rescues composition for code editing | kimi-reviewed on aider 30: 50% pass@1 / 93.3% pass@2 @279 s vs kimi solo 73.3% / 96.7% @173 s | Refuted — even review-only context degrades the reviser's precision; code lane stays solo + escalation |

## The measured lane map (shipped as moa-routed-config-example.yaml)

- **Classifier + SELF answers**: Cerebras gemma-4-31b (100% routing accuracy,
  p50 288 ms; 39/40 on exact-answer reasoning at 1.1 s/task).
- **Coding lane**: kimi-k2.6 solo (73% pass@1 — cheapest viable editor;
  wafer models fail the format).
- **Hard-reasoning lane**: heavy mixed fan-out with `reference_max_tokens: 600`.
- **General/chat lane**: all-Cerebras MoA (38/40 @ 6.8 s/task) or gemma31 solo.
- **Escalation lane**: Fable/Opus via `router.escalation` — invoked only on
  observed failure; measured at ~25% of frontier calls for full frontier
  quality on verifiable work.

## Open items for the next loop

- Inverted MoA for code (aggregator drafts solo, references only review).
- Agentic learning cycles (aider exercises as evolve train set; per-route
  skills — traces already record routing).
- Local pilot: two vLLM models on pg (5090+4090) as custom providers behind
  the same routed endpoint.
- Full Terminal-Bench 2.x (89 tasks) + SWE-bench Verified slice on the
  winning configs for leaderboard-comparable numbers.
