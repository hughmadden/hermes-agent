# MoA as an OpenAI-Compatible Endpoint (`hermes moa serve`)

Hermes' Mixture-of-Agents runtime is normally an *execution mode around the
main Hermes agent*: references advise, the aggregator acts, and the Hermes
loop owns tools. `hermes moa serve` inverts that ownership so **other agents
can use Hermes MoA as their upstream model**: OpenCode, OpenClaw, another
Hermes instance, or any OpenAI-SDK client points at the endpoint, and *the
client* executes tools and owns turn termination. The endpoint contributes
what MoA is uniquely good at — multi-model judgement plus a strong acting
synthesis — over the standard `/v1/chat/completions` wire format.

## Quickstart

```bash
hermes moa serve                     # 127.0.0.1:8646, no auth (loopback)
hermes moa serve --port 9000 --api-key s3cret
HERMES_MOA_API_KEY=s3cret hermes moa serve --host 0.0.0.0   # LAN, auth required
```

Client side (any OpenAI SDK):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8646/v1", api_key="s3cret")
resp = client.chat.completions.create(
    model="moa:default",             # a Hermes MoA preset
    messages=[{"role": "user", "content": "plan this refactor"}],
    tools=my_tools,                  # executed by YOUR agent, not Hermes
    stream=True,
)
```

### Docker (standalone proxy, fresh clone)

```bash
git clone <repo> && cd hermes-agent
mkdir moa-home
cat > moa-home/config.yaml <<'EOF'
moa:
  default_preset: heavy
  presets:
    heavy:
      reference_models:
        - {provider: openrouter, model: deepseek/deepseek-v4-pro}
        - {provider: openrouter, model: qwen/qwen3.7-max}
        - {provider: openrouter, model: z-ai/glm-5.2}
      aggregator: {provider: openrouter, model: moonshotai/kimi-k2.6}
      reference_max_tokens: 1500
EOF
export OPENROUTER_API_KEY=sk-or-...
docker compose -f docker-compose.moa.yml up -d
curl -s http://127.0.0.1:8646/v1/models | jq -r '.data[].id'
```

The image (`docker/moa-proxy/Dockerfile`) is intentionally slim — the CLI
plus aiohttp; no gateway/dashboard/messaging stacks. Add a `moa.router`
block (below) to also get `moa:auto`. Any harness that speaks the OpenAI
API can now use MoA as its model: `aider --openai-api-base
http://127.0.0.1:8646/v1 --model openai/moa:heavy`, `mini-extra swebench
--model openai/moa:heavy` with `OPENAI_API_BASE` set, Harbor's terminus-2
via `--ak api_base=...`, etc. (litellm-based tools pass the `moa:` model id
through verbatim after the `openai/` provider prefix).

## API surface

| Route | Behavior |
|-------|----------|
| `POST /v1/chat/completions` | Full MoA turn: reference fan-out → aggregator acts. Streaming + client-side tools supported. |
| `GET /v1/models` | Lists enabled MoA presets as models (`moa:<preset>`), with reference/aggregator metadata. |
| `GET /health` | Liveness; never requires auth. |

**Model naming.** `model` selects a preset: `moa:review`, `moa/review`, or
bare `review`; empty / `moa` / `default` select the configured default
preset. Unknown presets return an OpenAI-shaped 404 (`model_not_found`).

**Providers.** Preset slots resolve through the canonical
`resolve_runtime_provider` path, so every provider Hermes supports works —
API-key providers (OpenRouter etc.) and plan/OAuth providers (Anthropic
OAuth, openai-codex, nous, xai) alike, with per-provider wire-format
handling applied per slot.

## Tool calling — downstream by design

The endpoint never executes tools. Per request:

1. References receive the *advisory view* of the client transcript
   (`_reference_messages`): tool calls and tool results flattened to text, so
   strict providers don't reject tool messages they never produced.
2. The aggregator receives the client transcript **verbatim** (including
   `tool_calls` / `tool` messages) plus the injected reference-context block,
   and the client's `tools` (and `tool_choice`, via extra body) pass through
   to it.
3. Aggregator `tool_calls` stream back to the client, which executes them and
   sends the results in its next request — where the references judge the
   advanced state again. This is exactly the internal per-tool-iteration MoA
   loop, driven across the wire by the client.

A server-wide LRU keyed on the advisory-view signature means a retried or
duplicated request reuses the previous fan-out instead of re-billing it; any
state advance (new user turn, new tool result) is a cache miss and re-runs
the references.

## Streaming semantics

All intermediate model activity streams as *reasoning* deltas — both
`delta.reasoning` (OpenRouter style) and `delta.reasoning_content`
(DeepSeek style) are set, so most reasoning-aware clients render it as
thinking; clients that ignore unknown fields still get a clean answer.

Order on the wire:

1. `[Reference i/N — provider:model]` header, then that reference's
   own reasoning and advice text, live token-by-token. Reference 1 streams
   live while later references buffer in per-reference queues, each flushed
   in order as its predecessor finishes — live first-token latency without
   interleaving unlabelled text from concurrent models.
2. `[Aggregating — provider:model acting on N reference(s)]` marker.
3. The aggregator's own reasoning deltas (as reasoning), then its acting
   output as plain `delta.content` / `delta.tool_calls`, then
   `finish_reason`, an optional usage chunk (`stream_options.include_usage`),
   and `data: [DONE]`.

## Usage accounting

Top-level `usage` is the **sum across every upstream call** (all references +
aggregator) — the true cost of the request. The per-slot split is under
`usage.moa`:

```json
"usage": {
  "prompt_tokens": 5210, "completion_tokens": 940, "total_tokens": 6150,
  "moa": {
    "references": [{"label": "openrouter:...", "cached": false, "prompt_tokens": ...}],
    "aggregator": {"prompt_tokens": ..., "completion_tokens": ...}
  }
}
```

Cached reference replays contribute zero to the totals and are marked
`"cached": true`.

## Security

- Defaults to loopback with no auth. `--api-key` (or `HERMES_MOA_API_KEY`)
  enforces `Authorization: Bearer` on all `/v1` routes; binding a
  non-loopback host without a key prints a loud warning — the endpoint
  spends *your* configured provider credentials on behalf of callers.
- Client `Authorization` headers are never forwarded upstream; slot
  credentials come from the normal Hermes provider resolution.

## Traces and the improvement loop (skills / evolution)

Set `moa.save_traces: true` and every proxied turn that runs the fan-out is
appended to `<hermes_home>/moa-traces/<session>.jsonl` via the canonical
`save_moa_turn` writer — the exact messages each reference saw, each
reference's full advice, the exact aggregator input (with the injected
guidance block) and its acting output. Proxy turns are keyed by the client's
`x-hermes-session-id` header when supplied, else a stable hash of the first
user message, so one client conversation lands in one trace file.

That trace stream is the substrate for making MoA *get better with use*. The
loop is implemented:

1. **Grade + distill: `hermes moa evolve`.** Reads the newest recorded turns
   (`--max-turns`, default 30), has an LLM (`--model provider:model`, default
   the default preset's aggregator) grade them — which references were
   followed/ignored/right, recurring aggregation mistakes — and REWRITES
   `skills/moa-aggregation/SKILL.md` as a bounded heuristics document
   (existing rules merged/dropped, ~4 KB cap since it rides in every MoA
   prompt). `--dry-run` prints instead of writing. Run it ad hoc or from a
   Hermes cron job.
2. **Inject: automatic.** When the skill file exists, its body is appended to
   every aggregator guidance block — in-process MoA turns and proxied turns
   alike (`aggregation_skill_block()` in `agent/moa_loop.py`, mtime-cached).
   No skill file = zero behavior change. The block sits at the *end* of the
   prompt, so the conversation prefix stays KV-cache-stable.
3. **Ground truth for free.** Client-side tool results arriving in the *next*
   proxied request show whether the previous synthesis was right; they are in
   the traces the grader reads.
4. **Graded outcomes (optional).** Eval harnesses that know the expected
   answer can attach an external grade to a trace record (the `outcome`
   field via `consume_and_save_trace(outcome=...)` /
   `save_moa_turn(outcome=...)`); `hermes moa evolve` treats it as ground
   truth and mines incorrect turns hardest. `scripts/moa_learning_cycle.py`
   runs the full loop as a train/held-out benchmark.
5. **Evolve preset composition (future).** The same graded data ranks
   reference models per task domain — enough signal to auto-tune presets
   (drop a reference that is never followed; cap `reference_max_tokens` when
   long advice adds latency but no lift). See
   `docs/plans/moa-proxy-backlog.md`.

## Routing — `moa:auto`

With a router block configured, clients can request `model: "moa:auto"` and
a fast classifier assigns each request to the best preset — or answers it
directly (the SELF class) when it is trivial, skipping the reference fan-out
entirely. That self-answer path is the main latency/cost win: greetings,
acknowledgements, and single-fact queries stop paying the full MoA
multiplier.

```yaml
moa:
  default_preset: general
  router:
    enabled: true
    classifier: {provider: openrouter, model: google/gemma-4-31b-it}
    default: general          # fallback on classifier error/timeout
    self_answer: true         # enable the SELF class (default)
    # self_answer_model: {provider: ..., model: ...}   # default: classifier
    timeout_s: 8
  presets:
    coding:
      route: {description: "writing or debugging code, refactors, shell"}
      # ... reference_models / aggregator as usual
    general:
      route: {description: "everything that is not code and not trivial"}
      # ...
```

Semantics:

- Only presets carrying a `route.description` are routing candidates; the
  router refuses to enable without a classifier slot and at least one.
- **Sticky sessions.** One client conversation (same `x-hermes-session-id`,
  else the first user message) keeps its first routing decision, so tool
  loops never re-classify or flip presets mid-conversation.
- **Fallbacks.** Classifier error, timeout, or an unparseable label routes to
  `router.default`. A routing failure never fails the request.
- **Surface.** The response `model` echoes the routed preset
  (`moa:coding`, `moa:self`); streaming announces the decision in the first
  reasoning delta; `usage.moa.routed_preset` + `usage.moa.routing`
  (method/latency) carry it structurally; trace records gain a `routing`
  field so `hermes moa evolve` can grade routing quality offline.
- `GET /v1/models` lists `moa:auto` with classifier + routable-preset
  metadata when the router is enabled.

Classifier slot guidance (measured 2026-07-02/03,
`scripts/moa_router_bench.py`, 24 labelled cases x2): Cerebras
`gemma-4-31b` on a **paid** key is the best measured classifier — 100%
accuracy, p50 288 ms, p90 377 ms, zero fallbacks. `google/gemma-4-31b-it`
via OpenRouter is the drop-in substitute (100% accuracy, p50 338 ms, but a
long p90 tail of 1.45 s). A **free-tier** Cerebras key RPM-collapses under
bursts (42/48 fell back — all served correctly via the default preset), so
don't put one in production. Cerebras also benches well beyond
classification: as a fast-merge *aggregator* over strong references it
matched the best accuracy at the lowest full-fan-out latency, and an
all-Cerebras preset makes a strong interactive fast lane — see
`scripts/moa_cerebras_bench.py` and
`docs/plans/moa-cerebras-bench-20260703.json`.

## Testing

- Unit (no network): `pytest tests/hermes_cli/test_moa_proxy_server.py tests/hermes_cli/test_moa_router.py`
- Live end-to-end (real OpenRouter models, small spend):
  `OPENROUTER_API_KEY=... pytest -m integration tests/integration/test_moa_proxy_live.py tests/integration/test_moa_router_live.py`
  (router live tests also honor `CEREBRAS_API_KEY` for the custom-provider classifier)
- Both in Docker: `tests/integration/docker/run-moa-proxy-tests.sh`
- Router accuracy/latency: `scripts/moa_router_bench.py`; learning loop
  benchmark: `scripts/moa_learning_cycle.py`; composition benchmark:
  `scripts/moa_bench.py`
