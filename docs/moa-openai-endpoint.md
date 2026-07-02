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

That trace stream is the substrate for making MoA *get better with use*:

1. **Grade turns after the fact.** A scheduled Hermes job (cron/kanban)
   replays recent traces and scores them: did the aggregator follow, correct,
   or ignore each reference? Did the client's next request show the tool call
   succeeded or bounce back with an error? Client-side tool results arriving
   in the *next* proxied request are free ground truth about whether the
   previous synthesis was right.
2. **Distill graded traces into a skill.** The grader's durable lessons —
   "reference X is consistently wrong about SQL migrations", "when references
   disagree on file paths, verify before acting" — belong in a Hermes skill
   (e.g. `skills/moa-aggregation/SKILL.md`) as concrete aggregation
   heuristics with examples mined from traces.
3. **Feed the skill back into the aggregator.** The natural injection point
   is the guidance block built in `_reference_guidance` /
   `aggregate_moa_context`: prepend the distilled heuristics so the
   aggregator synthesizes with accumulated judgement, not just this turn's
   advice. Because the block sits at the *end* of the prompt, the
   conversation prefix stays KV-cache-stable.
4. **Evolve preset composition.** Longer-horizon, the same graded data ranks
   reference models per task domain — enough signal to auto-tune presets
   (drop a reference that is never followed; cap `reference_max_tokens` when
   long advice adds latency but no lift).

Steps 1–2 need no runtime changes — they are offline consumers of the trace
files this endpoint already writes. Step 3 is a small, deliberate change to
the guidance builder once a distilled skill exists and has been reviewed.

## Testing

- Unit (no network): `pytest tests/hermes_cli/test_moa_proxy_server.py`
- Live end-to-end (real OpenRouter models, small spend):
  `OPENROUTER_API_KEY=... pytest -m integration tests/integration/test_moa_proxy_live.py`
- Both in Docker: `tests/integration/docker/run-moa-proxy-tests.sh`
