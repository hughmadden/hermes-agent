# PR draft — MoA OpenAI endpoint: serve, evolve, route, benchmark

For Hugh to review before any upstream submission. Branch:
`feature/moa-openai-proxy` on the `hughmadden/hermes-agent` fork. Do NOT
open the upstream PR without explicit approval.

---

## Title

feat(moa): OpenAI-compatible MoA endpoint with learning loop, moa:auto
routing, and a docker release path

## Body

### What

Four increments that turn Hermes' Mixture-of-Agents mode into an upstream
model other agents can use:

1. **`hermes moa serve`** — an OpenAI-compatible `/v1/chat/completions` +
   `/v1/models` endpoint where every MoA preset is a model (`moa:<preset>`).
   The calling client owns tools and turn termination; references stream as
   `reasoning` deltas; `usage` sums every upstream call with a per-slot
   `usage.moa` breakdown. Hard per-slot timeouts (`moa.slot_timeout_s`).
2. **`hermes moa evolve`** — offline distillation of recorded MoA traces
   (`moa.save_traces`) into a bounded aggregation-heuristics skill that is
   injected into every subsequent aggregator prompt. Trace records accept
   externally graded outcomes, so eval harnesses feed supervised signal.
3. **`moa:auto` routing** — a fast classifier assigns each request to the
   best route-described preset or answers trivial requests directly (SELF
   class, no fan-out). Sticky per-conversation decisions, hard fallbacks,
   decision surfaced in reasoning deltas / `usage.moa.routed_preset` /
   traces.
4. **Docker release path** — slim `docker/moa-proxy/Dockerfile` +
   `docker-compose.moa.yml` + docs quickstart; fresh clone → compose up →
   point any OpenAI client (aider, mini-swe-agent, Harbor/terminus-2) at
   `:8646`.

### Evidence

- Unit: 62 tests (server, router, evolve) — hermetic, no network.
- Live: 9 integration tests (streaming thinking, client-side tool round
  trip, evolve end-to-end, routing incl. Cerebras custom-provider
  classifier).
- Router bench (`scripts/moa_router_bench.py`): 100% routing accuracy,
  p50 338 ms via `google/gemma-4-31b-it` on OpenRouter
  (`docs/plans/moa-router-bench-20260702.json`).
- Learning loop (`scripts/moa_learning_cycle.py`): train/held-out split,
  paired with/without-skill evals, leakage scan —
  `docs/plans/moa-learning-results-20260703.md`. (TBD: final numbers)
- Public benchmarks through the endpoint (aider polyglot, SWE-bench Lite
  via mini-swe-agent, Terminal-Bench via Harbor/terminus-2) —
  `docs/plans/moa-public-bench-results-20260703.md`. (TBD: final numbers)
- Composition bench: `docs/plans/moa-bench-results-20260702.md` (open-weight
  MoA combos matched frontier solos 16/16 on the exact-answer set).

### Compatibility

- No behavior change for existing users: everything is opt-in config
  (`moa.presets[].route`, `moa.router`, `moa.save_traces`,
  `moa.slot_timeout_s` defaults to 300 s on proxied turns only).
- New deps: none in core (aiohttp already optional for `moa serve`).

### Commits

(one-line summaries; see branch for full messages)

- feat(moa): serve MoA presets as an OpenAI-compatible endpoint
- feat(moa): evolution loop (hermes moa evolve), composition bench
- feat(moa): moa:auto routing + graded-outcome traces + learning-cycle bench
- test(moa): live routing tests + router accuracy/latency mini-bench
- docs(moa): moa:auto routing section + measured router bench results
- fix(moa): timeout forwarding on non-streaming aggregator path
- feat(moa): hard per-slot timeout for proxied turns
- feat(moa): standalone docker path for the MoA proxy
- docs(moa): multi-GPU MoA plan (research + measurements)

---

## Reviewer notes for Hugh

- The Cerebras custom-provider classifier works but the free-tier key
  RPM-collapses under bursts; production config uses OpenRouter-hosted
  gemma-4-31b-it (decision + data in the routing docs section).
- `docs/plans/multi-gpu-moa-plan.md` is Turq-specific research; consider
  keeping it out of the upstream PR (drop the commit or move the file) if
  upstream shouldn't carry hardware-plan docs.
- Squash-vs-keep: the branch history is clean enough to keep, but the two
  timeout commits could be squashed into the serve commit on rebase.
