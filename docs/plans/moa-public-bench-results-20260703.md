# Public benchmarks through `hermes moa serve` (2026-07-03)

Three community-standard harnesses driven against the OpenAI-compatible MoA
endpoint (`moa:<preset>` models over one long-running proxy container). All
harnesses fully containerized; no host installs. Configs: frontier solos
(claude-opus-4.8, gpt-5.5 via OpenRouter), open solos (kimi-k2.6,
deepseek-v4-pro), the best open MoA combos from the 2026-07-02 composition
bench, and the routed system (`moa:auto`).

Harness versions (verified current 2026-07): aider v0.86 polyglot benchmark
(whole edit format), mini-swe-agent v2.4.4 (SWE-bench Lite, official
bash-only scaffold), Harbor v0.16.1 + terminus-2 (Terminal-Bench sample
2.0 dataset, 10 tasks, 2× timeout multiplier for MoA latency).

## Aider polyglot — 30 Python exercises, whole format, pass@1 / pass@2

| Config | pass@1 | pass@2 | s/case | Notes |
|---|---|---|---|---|
| frontier-opus-solo | 90.0% | 100% | 9.9 | |
| frontier-gpt55-solo | 73.3% | 100% | 22.7 | |
| open-kimi-solo | 73.3% | 96.7% | 173 | |
| open-moa-heavy | 60.0% | 96.7% | 347 | 1 exhausted context window |
| open-moa-flash | 43.3% | 93.3% | 341 | |
| moa:auto (routed) | TBD | TBD | TBD | running |

**Finding — MoA hurts precise code editing at pass@1.** `open-moa-heavy`
scored *below its own aggregator run solo* (60% vs kimi's 73.3%) at 2× the
latency. The advisory context appears to distract exact-format editing
(whole-file rewrites) even though it clearly helps exact-answer reasoning
(see the gpusim/learning results). pass@2 converges (96.7% both). Frontier
solos dominate on speed. This is the honest counterpoint to the composition
story: **route coding traffic to a strong solo (or a coding-tuned
aggregator), keep the fan-out for reasoning-heavy work** — exactly what
`moa:auto` route descriptions can encode.

## SWE-bench Lite — mini-swe-agent, 25-instance slice (0:25), test split

| Config | Submitted | Resolved | Notes |
|---|---|---|---|
| open-kimi-solo | TBD | TBD | |
| open-moa-heavy | TBD | TBD | |
| frontier-gpt55-solo | TBD | TBD | |

(Resolution via local `swebench==4.1.0` docker evaluation.)

## Terminal-Bench sample 2.0 — Harbor + terminus-2, 10 tasks

| Config | Solved | Errors | Notes |
|---|---|---|---|
| open-moa-flash | 6/10 (0.60) | 2 | 1 agent-timeout + 1 harness RuntimeError among the misses |
| open-moa-heavy | TBD | TBD | |
| open-kimi-solo | TBD | TBD | |
| frontier-gpt55-solo | TBD | TBD | |
| moa:auto (routed) | TBD | TBD | |

## Methodology notes

- The proxy forwards client tools to the aggregator (client executes them);
  mini-swe-agent's native tool-calling worked unmodified through
  `moa:`-prefixed models (litellm passes the id verbatim after the
  `openai/` provider prefix).
- Sticky routing (`moa:auto`) keys on the harness conversation, so
  multi-step tool loops never re-classify mid-task.
- Terminal-Bench runs use `--timeout-multiplier 2`: MoA turn latency
  (reference fan-out) eats agent time budgets tuned for solo models.
- Costs and exact model list: see `scripts/moa_bench.py` CONFIGS and the
  serve config used for the runs.
