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

| Config (simulates) | Acc | Avg latency | Tokens/q |
|---|---|---|---|
| gpu4-composed (coder-next + gpt-oss-120b + 80b-thinking → qwen3.5-122b) | TBD | TBD | TBD |
| gpu1 solo qwen3.5-122b | TBD | TBD | TBD |
| gpu1 solo gpt-oss-120b | TBD | TBD | TBD |
| tp4 nemotron-3-ultra-550b | TBD | TBD | TBD |
| tp4 qwen3.5-397b | TBD | TBD | TBD |
| tp2 deepseek-v4-flash | TBD | TBD | TBD |

Coding axis: aider polyglot runs through `hermes moa serve` (see
`docs/plans/moa-public-bench-results-*.md`).

## 6. Throughput axis — measured on pg (RTX 5090 + RTX 4090)

vLLM (`vllm/vllm-openai:latest`), Qwen3-8B, `bench_endpoint.py` (decode
tok/s from streamed chunk timestamps). Mixed-SKU caveat: TP=2 across a 5090
and a 4090 is *worse* than matched cards (the 4090 gates each allreduce),
and these GeForce cards have **no P2P** — so the measured TP penalty is an
upper bound for the PRO 6000 box, which has P2P on stock drivers.

| Config | Single-stream decode tok/s | ITL p50 | Aggregate tok/s @ c=16 |
|---|---|---|---|
| 5090 alone | TBD | TBD | TBD |
| 4090 alone | TBD | TBD | TBD |
| TP=2 over PCIe (5090+4090) | TBD | TBD | TBD |

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

## 8. Verdict (draft)

TBD after measurements. Shape of the answer from the evidence so far: the
hypothesis holds for **latency, throughput, cost, and verifiable/agentic
work** (per-GPU models decode 3–5× faster than a TP=4 550B-class model and
the MoA/router composition retains most of the quality), while the one-big-
model side keeps a real edge on **open-ended general reasoning** (AA 48 vs
~30s tier). The box should therefore be provisioned for composition-first
with a TP=2 "quality lane" (V4-Flash-class) rather than a single TP=4
monolith.
