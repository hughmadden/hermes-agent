# Quality-improvement loop — 2026-07-03

Hypothesize → test → ship → repeat, one measured change per iteration.
Raw data: the `moa-*-20260703.json` files in this directory; public report:
the Turq share (services.turquoisebay.ai/share/moa-next-phase/).

| # | Hypothesis | Result | Verdict / action |
|---|---|---|---|
| 1 | Routing coding to a solo lane removes the MoA editing penalty | moa:auto v2 (coding→kimi lane): 66.7% pass@1 @235 s vs v1 (coding→fan-out) 63.3% @527 s; 71/73 turns routed correctly | Confirmed — kimi-parity within single-run noise; solo lanes made routable (disabled preset + route block) |
| 2 | Self-MoA (best model ×3) beats mixed references | kimi-solo, kimi-selfmoa3, mixed-heavy ALL 40/40; solo used 4× fewer tokens | No signal at ceiling — composition only adds cost once the model saturates the task family |
| 3a | A classifier can pre-route "routine" work to a wafer-cheap lane | Cerebras GLM-4.7 cheap lane: 30% pass@1 on work the classifier correctly called routine | Refuted — never guess difficulty up-front; the cheap lane must be validated per format |
| 3b | Verification-gated escalation: cheap first, frontier on test failure | kimi 22/30 + Fable(with failing tests) 8/8 = 30/30 composite; Fable-solo baseline 26/30 / 30/30 @13.6 s | Confirmed as an ORACLE-DETECTED upper bound (benchmark gold tests picked the escalations) |
| 4 | Escalation replicates on a hard agentic benchmark | SWE-bench Lite 25: kimi 19/25 resolved + Fable 6/6 on the remainder = 25/25 composite, Fable on 24% of instances | Upper bound: failures identified by SWE-bench HIDDEN eval tests, invisible to a runtime agent; live composite would land between 76% and 100% |
| 5 | Productize: failure-gated escalation lane in the router | `router.escalation` shipped: sticky conversation re-routes once when the latest tool/user feedback matches failure patterns; 6 unit tests | Shipped |
| 6 | Live end-to-end: aider drives moa:auto with the escalation lane | 83.3% pass@1 / 100% pass@2 @36.6 s/case; only 3/30 exercises ever reached Fable | Confirmed in production shape — the router only pattern-matches failure evidence the CLIENT produced by running tests itself; works exactly when the agent loop emits verifiable signals |
| 7 | Capping reference advice cuts fan-out latency for free | cap 1500→600: −17% latency, −15% tokens, same accuracy; cap 300: no further gain | Adopt 600 as guidance; latency is provider-think-time-bound below that |
| 8 | A wafer-speed model can be the cheap coding lane | Cerebras gemma-4-31b on aider: 2.2 s/case, 10% pass@1 (50% pass@2) | Refuted — wafer models own classify/self/reasoning lanes, not editing |
| 9 | Inverted MoA (draft→review→revise) rescues composition for code editing | kimi-reviewed on aider 30: 50% pass@1 / 93.3% pass@2 @279 s vs kimi solo 73.3% / 96.7% @173 s | Refuted — even review-only context degrades the reviser's precision; code lane stays solo + escalation |
| 10 | Python verifier tool lifts hard reasoning | v1 harness (4-round cap) collapsed to empty answers — termination artifact. v2 (8 rounds + forced final): kimi 56/60 (93%) vs 88% baseline; cere-moa 87% vs 90% | Confirmed for strong thinking solos (+5 pp, +58% latency); no gain for wafer MoA |
| 11 | Quorum straggler-dropping costs accuracy | mixed-heavy + grace 0.5 on AIME: 60/60 vs 55/60 baseline, similar latency | No accuracy cost (straggler is often the noisy reference); keep on for latency-sensitive lanes |

| 12 | HMMT mix batch: frontier councils, wafer-briefed judges, local trios | Fable saturates HMMT (20/20); councils no-harm-no-lift; plan-GPT-5.5 19/20 at $0; local-trio→GLM-5.2 17/20 best local (+5 pp comp lift); qwen3.6-27b 16/20 sleeper; step-3.5-flash provider-broken on OR (18/40 empty responses) | Frontier evaluation needs long-horizon agentic tasks, not competition math; plan-integrated GPT-5.5 = value king |
| 13 | Live escalation with a plan-covered frontier lane (SWE-bench) | kimi→plan-GPT-5.5 moa:auto: 20/25 = paid-frontier parity at $0 frontier cost; BUT 20/25 conversations escalated (failure patterns over-fire on debugging) | Confirmed + design lesson |
| 14 | min_failures gate stops over-firing | min_failures=3: same 20/25, frontier turn share 33%→18%, kimi drives 82% of turns | Confirmed — shipped as escalation.min_failures; match it to the client's retry depth |

| 15 | (stopped mid-run by Hugh) TB agentic councils | lane 1 only: plan-GPT-5.5 on TB sample 8/10 at $0 in 13 min | Frontier agentic at zero marginal cost |
| 16 | FLAGSHIP — mode:cascade (lazy MoA, consensus-gated) | AIME 93% @9.9s median (tier-0 1.4s, 20/20 precision); HMMT 90% vs cere-moa 75%; frontier ≤15% at $0 | Confirmed — strictly dominates always-on MoA; shipped |
| 17 | k-sampled consensus (4 voters, 3-of-4) | same 93% AIME, median 4.4s, tier-0 rate 80%, frontier 5% | Confirmed — zero-code tuning via duplicated slots; new default |
| 18 | Judge gate extends tier-0 to freeform | tier-0 on 30/30 prose prompts @2.6s (2× faster) but blind quality 2/2/23 vs always-on MoA | Split verdict: consensus transfers correctness, not polish — latency-first option only |

| 19 | Verification rescues correlated-confidence errors | cascade-wafer4v AIME 55/60 (92%) vs 56/60 unverified; 2 WRONG verdicts correctly forced tier-2 (6/7); false consensus survived — 31B verifier can't check what 120B voters missed | Neutral on insight-bound tasks; verification pays only when the check is pure computation. Ship as optional; the frontier dial is the honest accuracy lever |
| 20 | One endpoint can carry every measured optimization | moa:omni (router+cascade+escalation+lanes): 10/10 mixed-traffic prompts routed correctly, 2.6 s median | Confirmed — the product; shipped as the example config |

| 21 | Consensus strictness is the quality/cost dial | HMMT dial sweep: mc2 90% / mc3 85% / mc4 (unanimity) 20/20 (100%) @ 2 frontier calls vs GPT-5.5 solo 19/20 @ 20 calls | Confirmed (n=20 caveat): strictness buys accuracy with wafer aggregation, not frontier spend; mc3 latency default, mc4 quality default |

| 22 | Dial confirmation at n=60 | mc4 AIME 88% < mc3 93%: strictness demoted 95.8%-precise weak consensus into a 73%-accurate tier-1 aggregator | Dial is task-relative, not monotone; constants: unanimous tier-0 ≈ perfect (53/53 cross-set), tier-2 near-perfect; tier-1 aggregator = weak link (upgrade queued) |

| 23 | Stronger wafer tier-1 aggregator lifts the cascade | gpt-oss agg: 90% (vs glm 93%) — within single-run noise; family invariant 92±1.5% | Tier-1 wafer aggregator choice is noise; accuracy is set by consensus precision + frontier share |
| 24 | Frontier-arbitrated disagreements reach ~97% | consensus-or-frontier: 93% @3.5s median, frontier 15%; GPT-5.5-as-arbiter 7/9 on disagreements vs ~98% solo | REFUTED projection — voter context ANCHORS even a frontier arbiter (same contamination as code editing); clean-escalation (discard voter work) queued |

| 25 | Clean arbitration recovers anchored losses | clean_arbiter shipped; disagreement-subset accuracy 78%→86% (directional confirm) but total 92% — 7-variant family mean 92.1% AIME | Anchoring real; ceiling is stochastic false consensus → voter DIVERSITY (new model families, e.g. local vLLM) is the next lever |

| 26 | An independent local voter family cuts false consensus | Hybrid pool (Qwen3-8B on RTX 5090 + Cerebras voters) ran flawlessly but 88% AIME: weak dissent = noise, tier-1 volume up, false consensus unchanged | Law: diversity value = independence × competence; the 4×96GB box needs 27B+ local voters — mechanism proven, model class matters |

## AIME 24+25 retest (harder primary benchmark, 60 problems)

fable 100% · gpt5.5 98% · cere-agg-openrefs(gptoss) 95% · mixed-heavy 92%
(100% with quorum) · gpu4-composed 90% · cere-moa 90% @19 s/task ·
kimi 88% (93% with verifier tool) · v4flash 78% · cere-gemma31 77% @1.7 s ·
nano 32%. Composition lifts hard reasoning +4..+7 pp over component solos —
the 40-task "no gain" was a ceiling artifact. Raw: moa-aime-results-*.json,
moa-verifier-v{1,2}-results-*.json, moa-quorum-results-*.json.

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
