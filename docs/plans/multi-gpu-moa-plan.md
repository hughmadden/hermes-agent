# 4× RTX PRO 6000 Blackwell — MoA-of-small-models vs one big TP=4 model

Plan + evidence for a workstation-class box: 4 × 96 GB Blackwell cards, no
NVLink, PCIe 5.0 only. Hypothesis under test: **independent per-GPU models
composed via MoA/routing beat one large tensor-parallel model running over
PCIe.** Research 2026-07-02/03 (web+X sweep, URLs inline); quality axis
simulated on OpenRouter through the identical `hermes moa serve` code path;
throughput axis measured empirically on pg (RTX 5090 + RTX 4090).

## 1. Hardware dossier (verified)

- **RTX PRO 6000 Blackwell** (GB202, SM 12.0): 96 GB GDDR7 ECC, 1792 GB/s
  (Workstation/Max-Q), PCIe 5.0 x16, **no NVLink**. FP4 ≈ 4 PFLOPS sparse
  ("4000 AI TOPS"), FP8 ≈ 1–2 PFLOPS. Max-Q variant = same silicon at 300 W,
  blower — the intended 4-in-a-box part (NVIDIA: "scale up to four Max-Q");
  validated quad builds: a16z (Threadripper PRO, 4× Gen5 x16), Exxact, HP Z8
  Fury G6i.
- **P2P over PCIe works on stock drivers** for this card (unlike GeForce
  4090/5090, where NVIDIA disabled it): Level1Techs measured 40–51 GB/s
  bidirectional `p2pBandwidthLatencyTest`; Google Cloud G4 (same silicon,
  Server Ed.) reports +168% inference throughput / −41% inter-token latency
  with its P2P fabric on vs off. Landmines to plan for: NCCL hangs with
  IOMMU on AMD hosts (nccl#1999 → `iommu=pt`), vLLM custom-allreduce not
  supported on SM120 (`--disable-custom-all-reduce`, NCCL fallback is fine).
- Price mid-2026: NVIDIA list **$13,250** (raised +55% June 2026, GDDR7
  supply); street $8.5k–12k and volatile. 4 cards ≈ **$34k–53k**.

## 2. What fits where (mid-2026 open-weight landscape)

Single 96 GB card (quantized, with KV headroom):

| Role | Model | Quant / VRAM | Notes |
|------|-------|--------------|-------|
| Coding | Qwen3-Coder-Next (80B MoE A3B) | FP8 ~80 GB; measured ~92 GB w/ 262K ctx | ~336 tok/s decode measured on exactly this GPU (Millstone); SWE-V 70.6 |
| Coding (max quality) | Mistral-Medium-3.5-128B | 4-bit ~66–72 GB | SWE-V 77.6 — highest of anything that fits one card |
| General/reasoning | Qwen3.5-122B-A10B | NVFP4 75.6 GB | MMLU-Pro 86.7, AA #1 in class; vision |
| Reasoning (mature) | gpt-oss-120b | native MXFP4 ~63 GB | ~30 GB KV headroom → parallel sampling |
| Router/small | Qwen3.5-9B | FP8 ~13 GB | classifier + SELF-answer slot |

Genuinely-TP=4-class (fits 384 GB pool at 4-bit):

- **Nemotron-3-Ultra-550B-A55B** (~275 GB NVFP4, AA Index 48 — best open
  model that honestly fits), MiniMax-M3 428B (~220 GB), Qwen3.5-397B-A17B
  (~200 GB, Apache), DeepSeek-V3.2 685B (~345 GB, tight).
- **Does NOT fit 4×96 GB**: GLM-5.2 (753B, ~377 GB weights alone — zero KV
  headroom), Kimi K2.6/K2.7 (1T, ≈594 GB INT4), DeepSeek-V4-Pro (1.6T).
  The frontier open tier has outgrown this box; an 8×96 GB machine is the
  entry ticket there.
- Efficiency outlier: **DeepSeek-V4-Flash** (284B-A13B, native FP4/FP8,
  ~165 GB → TP=2 on two cards, 190 tok/s single-stream measured, AA 40,
  79% SWE-V) — nearly V4-Pro quality on a quarter of the box.

## 3. Published TP-over-PCIe evidence

- NVLink vs PCIe (4×3090, vLLM, batched): **+~50% at TP=2, +~10% at TP=4**
  (himeshp). The penalty *shrinks* at higher TP because PCIe traffic
  dominates either way.
- TP on PCIe helps prefill, hurts single-stream decode: ~2 allreduces per
  layer (3 for MoE) × 10–20 µs PCIe latency ≈ **1–3 ms/token extra ITL**
  (derived; matches the "decode suffers, prefill fine" pattern in
  arXiv:2512.01644). Bandwidth is not the constraint — a dual-PRO-6000 TP=2
  deployment measured only 7–9 GB/s (~25%) of link utilization during decode.
- Community wisdom for ~27B models on 2× PRO 6000: NCCL-over-PCIe allreduce
  costs more than TP gains — **prefer one model per GPU at small sizes**
  (theogravity dual-card repos). Expect a 400–550B NVFP4 model at TP=4 to
  decode at ~45–90 tok/s single-stream vs ~200–340 tok/s for per-card models.

## 4. Composition-vs-monolith literature

- Original MoA (arXiv:2406.04692): open MoA > GPT-4o on AlpacaEval — but
  **Self-MoA** (arXiv:2502.00674) showed ensembling the *single best* model
  beats mixed MoA (+6.6%): MoA is quality-sensitive, not diversity-hungry.
- At matched compute, a 70B CoT pipeline still beats the best 8B multi-agent
  config by ~13% on general reasoning (arXiv:2605.01566). Big model wins
  open-ended reasoning.
- Composition wins where tasks are **verifiable** (code with tests, math
  with checkers): repeated small-model sampling beats one big pass at
  matched FLOPs (arXiv:2404.00725, 2502.06703); routing wins on cost
  (RouteLLM: 95% GPT-4 quality at −85% cost).

## 5. Quality axis — measured on OpenRouter (this repo)

`scripts/moa_gpusim_bench.py`, 40 verified exact-answer tasks (the
moa_learning_cycle train+held-out set), identical MoA turn machinery:

Measured 2026-07-03 (raw: `moa-gpusim` run, results JSON alongside):

| Config (simulates) | Acc | Avg latency | Tokens/q |
|---|---|---|---|
| **gpu4-composed** (coder-next + gpt-oss-120b + 80b-thinking → qwen3.5-122b) | **40/40** | 66 s | 14 598 |
| gpu1 solo qwen3.5-122b | 37/40 | 124 s | 8 189 |
| gpu1 solo gpt-oss-120b | 37/40 | 19 s | 1 199 |
| tp4 nemotron-3-ultra-550b | 24/40 * | 32 s | 1 246 |
| tp4 qwen3.5-397b | 35/40 | 383 s | 9 798 |
| tp2 deepseek-v4-flash | 38/40 | 23 s | 2 336 |

\* 11/16 nemotron misses were empty/invalid OpenRouter provider responses
(24/29 = 83% on clean responses) — a serving-reliability datapoint for the
frontier-open tier, not a quality verdict.

Findings: the composed 4-model MoA was the only perfect config and beat its
own aggregator solo (40 vs 37) — the fan-out adds signal, not just tokens.
The best TP-class solo (V4-Flash, 38/40) came close at ~6× fewer tokens; the
TP=4-class models were either slower (qwen3.5-397b: 383 s/task) or unreliable
via their providers. On this verifiable task family the quality argument for
one big TP=4 model over the composition did not materialize.

Coding axis: aider polyglot runs through `hermes moa serve` (see
`docs/plans/moa-public-bench-results-*.md`).

## 6. Throughput axis — measured on pg (RTX 5090 + RTX 4090)

vLLM (`vllm/vllm-openai:latest`), Qwen3-8B, `bench_endpoint.py` (decode
tok/s from streamed chunk timestamps). Mixed-SKU caveat: TP=2 across a 5090
and a 4090 is *worse* than matched cards (the 4090 gates each allreduce),
and these GeForce cards have **no P2P** — so the measured TP penalty is an
upper bound for the PRO 6000 box, which has P2P on stock drivers.

Measured 2026-07-03 (Qwen3-8B BF16, max_model_len 8192, 1024-token decodes,
NCCL_P2P_DISABLE=1, `--disable-custom-all-reduce`):

| Config | Single-stream decode tok/s | ITL p50 | Aggregate tok/s @ c=16 |
|---|---|---|---|
| 5090 alone | 98.2 | 10.2 ms | 1380 |
| 4090 alone | 58.4 | 17.1 ms | 832 |
| TP=2 over PCIe (5090+4090) | 97.1 | 10.3 ms | 1016 |

Findings, stark even as an upper bound:

- **TP=2 single-stream gain: zero.** 97.1 vs 98.2 tok/s on the 5090 alone —
  two GPUs' compute, one GPU's speed. The 4090 gates every layer and the
  per-layer allreduce eats the rest.
- **TP=2 batch throughput is NEGATIVE vs one card**: 1016 tok/s @16 vs 1380
  on the 5090 alone, and **2.2× worse than independent replicas** (1380 +
  832 = 2212 tok/s from the same two cards serving separately).
- Caveats: mixed SKUs overstate the penalty vs matched cards; GeForce has no
  P2P (the PRO 6000 does); an 8B dense model has proportionally high
  allreduce overhead. TP's real use is models that don't fit one card — this
  measurement is the *cost floor* of paying that tax when you don't have to.

## 7. Recommended configuration (draft — finalize with measurements)

- **GPU1**: Qwen3-Coder-Next FP8 — coding preset reference/aggregator.
- **GPU2**: Qwen3.5-122B-A10B NVFP4 — general reasoning + aggregator slot.
- **GPU3**: gpt-oss-120b MXFP4 — math/tool reasoning; its ~30 GB KV headroom
  serves parallel sampling for verifiable-task self-consistency.
- **GPU4**: Qwen3.5-9B FP8 router (moa:auto classifier + SELF-answer) + a
  second replica of the hottest model (or Mistral-Medium-3.5 as a second
  coder).
- Hermes side: this is exactly the `moa:auto` + presets shape shipped in
  this branch — router classifies onto coding/general/math presets whose
  slots point at the per-GPU endpoints; SELF class answers trivial traffic
  on GPU4 with no fan-out.
- Hybrid option the evidence supports: 2 cards TP=2 running DeepSeek-V4-Flash
  native (AA 40, ~190 tok/s) + coder card + router/sampler card.

## 8. Verdict

**The hypothesis holds for this box.** On the measured evidence:

- **Quality**: the 4-composed-small-models MoA scored 40/40 on the
  verifiable task set — above every solo tested, including the TP=4-class
  models it would replace (best: V4-Flash 38/40) and its own aggregator
  alone (37/40). The literature's caveat stands: one big model should still
  win *open-ended* reasoning at matched compute (AA-index gap), so composed
  is not a universal replacement — but for agentic/coding/verifiable
  workloads (this box's purpose) composition measured better, not just
  cheaper.
- **Throughput**: TP=2 over PCIe on pg delivered zero single-stream gain
  and 2.2× less aggregate throughput than the same two cards serving
  independently. Even discounting the mixed-SKU/no-P2P pessimism, the
  independent-per-GPU configuration is the clear throughput winner whenever
  the model fits one card.
- **Reliability bonus**: the composition degrades gracefully (a failed
  reference becomes a note; the turn completes) — the nemotron provider
  failures in the sim would have been full request failures on a monolith.

**Recommended configuration** (section 7): four independent single-GPU
models (coder / general / reasoning / router+replica) behind `hermes moa
serve` with `moa:auto` routing — exactly the software shipped on this
branch. Keep TP=2 as an optional "quality lane" for a V4-Flash-class model
(2 cards, native FP4/FP8, near-frontier-open quality) rather than ever
running a TP=4 monolith. **Buy verdict**: the 4×96 GB box is justified for
composition-first serving; if the workload were dominated by open-ended
frontier-quality reasoning instead, neither TP=4 on this box nor the
composition closes the gap to the 8×96 GB tier — that workload wants API
models or a bigger box.
