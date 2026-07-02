# MoA OpenAI-endpoint backlog

Deferred work for `hermes moa serve` (see `docs/moa-openai-endpoint.md` for
the shipped surface). Both items below are scoped but intentionally not
implemented yet.

## 1. Preset routing — classify each request onto the best MoA group

Today the client picks the preset by model name (`moa:<preset>`). The next
step is a virtual routed model — e.g. `model: "moa:auto"` — where a fast
classifier assigns each incoming request to the best preset out of a
preconfigured set.

Design sketch:

- **Config.** Each preset gains an optional `route` block:
  ```yaml
  moa:
    router:
      enabled: true
      classifier: {provider: cerebras, model: gemma-4-31b}   # fast/cheap slot
      default: general
    presets:
      coding:   {route: {description: "code writing, debugging, refactors, shell"}}
      math:     {route: {description: "calculation, proofs, quantitative puzzles"}}
      general:  {route: {description: "everything else"}}
  ```
- **Classifier call.** One small completion over the request tail (last user
  message + a capped transcript digest), constrained to output one preset
  key (logit-bias-free JSON/one-word protocol; same `_slot_runtime`
  resolution as every other slot). Latency budget ~200–400 ms, which is why
  the slot should be a fast host (Cerebras, Groq-class) rather than a
  frontier model.
- **Placement.** `handle_chat_completions` resolves `moa:auto` →
  classifier → concrete preset, then proceeds unchanged. The chosen preset
  is surfaced to the client in the first streamed reasoning delta
  (`[Routed to preset 'coding' — ...]`) and in `usage.moa.routed_preset`.
- **Fallbacks.** Classifier error/timeout → `router.default`. Unknown label
  → default. Router disabled → `moa:auto` 404s as today.
- **Sticky sessions.** Within one client conversation (same
  `x-hermes-session-id` or advisory-signature prefix), reuse the previous
  routing decision instead of re-classifying every tool iteration.
- **Evolution tie-in.** Trace records gain the routing decision, so
  `hermes moa evolve` can grade routing quality (wrong-preset symptoms:
  references consistently out of domain) and future work can tune the
  route descriptions the same way aggregation heuristics are tuned.

## 2. Cerebras Gemma 4 as fast classifier / aggregator

Cerebras' wafer-scale inference serves `gemma-4-31b` at very high
tokens/sec with an OpenAI-compatible surface — a strong fit for (a) the
routing classifier above and (b) a low-latency *aggregator* slot where the
references provide the depth and the aggregator mostly needs to merge
fast (chat-style presets, interactive tool loops).

Integration notes (verified against a working Hermes custom-provider setup):

- Custom provider, `base_url: https://api.cerebras.ai/v1`,
  `api_mode: chat_completions`, model `gemma-4-31b`, context 131072,
  key via `CEREBRAS_API_KEY`.
- The Cerebras edge 403s (Cloudflare error 1010) on default Python
  `urllib` user-agents. The Hermes custom-provider path (OpenAI SDK /
  httpx UA) is known to work; keep any new direct HTTP path on the same
  client stack or set a browser-like UA.
- Preset slots would look like
  `{provider: custom:cerebras, model: gemma-4-31b}` (or a first-class
  `cerebras` provider entry if promoted), resolved through the existing
  `resolve_runtime_provider` chokepoint — no MoA-specific code needed.
- Benchmark TODO once wired: `scripts/moa_bench.py` with
  `open-moa-*` reference sets and a `gemma-4-31b@cerebras` aggregator vs
  the same sets with heavyweight aggregators — measures how much accuracy
  the fast-merge trade costs against the wall-clock win.

## 3. (Stretch) Skill-aware A/B in the bench

`scripts/moa_bench.py` currently measures composition only. Once real
agentic traces accumulate and `hermes moa evolve` has produced a
non-trivial skill, add `--with-skill/--without-skill` paired runs to
quantify the evolution loop's lift on held-out tasks.
