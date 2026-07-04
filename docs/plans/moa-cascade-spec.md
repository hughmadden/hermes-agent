# Cascade mode (`mode: cascade`) — implementation spec v1

Lazy MoA with observation-based gating. Motivation (measured, see
moa-quality-loop-20260703.md): predicted difficulty fails (30% cascade
refuted); observed signals win (escalation at frontier parity, quorum,
verifier). The cheapest observation nobody has used: **answer agreement
between wafer-speed voters** (~2 s, ~$0.002 on Cerebras).

## Semantics

A preset with `mode: cascade` serves a non-streaming, tool-free request as:

- **Tier 0** — every `reference_models` slot answers **directly** (verbatim
  client messages, NO advisory system prompt), in parallel. Extract a
  candidate answer from each. If ≥ `min_consensus` normalized candidates
  are equal → return the full text of the FIRST voter holding the consensus
  answer. No aggregator call.
- **Tier 1** — on no-consensus: the preset's `aggregator` runs with the
  voter outputs attached as reference context (existing guidance builder).
  If its candidate answer agrees with ≥1 voter candidate, or no
  `cascade.escalate_to` is configured → return the aggregator response.
- **Tier 2** — aggregator agrees with NO voter (or produced no candidate)
  and `cascade.escalate_to` names a valid preset → call THAT preset's
  aggregator (solo), with voters + tier-1 output attached as reference
  context (tier-1 labelled `tier1-aggregator — <label>`). Return its
  response.

Fallbacks: requests with `tools`, or streaming requests, or presets with
<2 reference slots use the existing fanout path unchanged (documented v1
limitation; config normalization downgrades mode to "fanout" when there are
<2 reference slots).

Surface: `usage.moa.cascade = {"tier": 0|1|2, "consensus": <str|None>,
"votes": <int>, "voters": [labels], "candidates": [normalized-or-None]}`
(+ `"tier1_candidate"` when tier ≥1 ran).

## Files & contracts (exact names; tests are written against these)

### 1. `hermes_cli/moa_config.py`
- `_normalize_preset`: `mode` valid values become `{"fanout", "draft_review",
  "cascade"}`. When `mode == "cascade"` and the raw preset has <2 cleaned
  reference slots → mode falls back to `"fanout"`.
- New normalized key on every preset: `"cascade"` — `None` unless
  mode=="cascade", else:
  ```python
  {"escalate_to": <str|None>, "min_consensus": <int>=len(refs) clamped to >=2}
  ```
  `min_consensus` from raw `cascade.min_consensus` (int, clamp 2..len(refs);
  default len(refs) = unanimity).
- In `normalize_moa_config` (post-pass, mirroring router validation):
  for each preset with a cascade block, if `escalate_to` is not a key of the
  presets map (enabled or disabled both fine) → set it to `None`.

### 2. NEW `hermes_cli/proxy/moa_cascade.py` (pure helpers, no I/O)
```python
def extract_candidate(text: str) -> str | None
def normalize_candidate(s: str) -> str
def consensus(candidates: list[str | None], min_consensus: int) -> str | None
def agrees(a: str | None, b: str | None) -> bool
```
- `extract_candidate`: last `ANSWER\s*:\s*(.+)` match (case-insensitive,
  take last, first line of capture); else last `\\boxed{...}`; else the last
  non-empty line IF ≤80 chars; else None. Strip backticks/markdown
  emphasis/trailing `.`,`!`.
- `normalize_candidate`: strip whitespace/quotes/`$`, lowercase, collapse
  inner whitespace; `-?\d+` → canonical int string; `a/b` and
  `\\frac{a}{b}` → lowest-terms `a/b` via `fractions.Fraction`; other text
  returned cleaned.
- `consensus`: ignore None; group by normalized value; largest group ≥
  min_consensus → that normalized value, else None.
- `agrees(a,b)`: both not None and `normalize_candidate(a) ==
  normalize_candidate(b)`.

### 3. `agent/moa_loop.py`
`_run_reference(...)` and `_run_references_parallel(...)` gain
`direct: bool = False` (keyword-only, default preserves behavior). When
True, `_run_reference` does NOT prepend `_REFERENCE_SYSTEM_PROMPT` — the
messages list is used as given.

### 4. `hermes_cli/proxy/moa_server.py` (non-streaming path only)
- `handle_chat_completions`: when the resolved preset has mode=="cascade",
  pre-resolve `common["cascade_escalate_preset"]` = the normalized preset
  dict for `cascade.escalate_to` (or None) and store
  `common["cascade"] = preset["cascade"]`.
- `_handle_non_streaming._run_turn`: branch
  `if common["preset"].get("mode") == "cascade" and reference_models and not common["tools"]:`
  1. voters = `_run_references_parallel(reference_models,
     [dict(m) for m in messages], temperature=reference_temperature,
     max_tokens=(reference_max_tokens or common max_tokens),
     timeout=slot_timeout, quorum_grace=preset quorum, direct=True)`
  2. candidates = [extract_candidate(text) for each voter text]
  3. cons = consensus(candidates, common["cascade"]["min_consensus"])
  4. If cons: winner = first voter whose normalized candidate == cons.
     Build a response-equivalent: the handler returns
     `(reference_outputs, refs_from_cache=False, agg_messages=messages,
     response=None, cascade_info={...}, winner_text=...)` — implementer may
     restructure `_run_turn`'s return into a small dict; keep the existing
     non-cascade flow byte-identical. Winner's usage: voters' usages sum
     into the reference-usage total exactly like today's fan-out accounting
     (they ARE reference_outputs); aggregator usage = zeroed CanonicalUsage.
     Final JSON: message.content = winner full text, finish_reason "stop",
     usage.moa.cascade as per Surface above (tier 0).
  5. Else tier 1: existing guidance attach (voters as reference outputs) +
     aggregator call (existing code path, tools=None). Compute
     agg_candidate. If escalate preset configured AND (agg_candidate is
     None or no voter candidate agrees with it) → tier 2: guidance =
     voters + ("tier1-aggregator — " + agg slot label, agg text) appended
     as an extra reference tuple (usage accounting: tier-1 agg usage must
     still be counted — fold it as an extra reference-output acct entry);
     call escalate preset's aggregator via `call_llm(task="moa_aggregator",
     messages=<client messages + guidance>, temperature=escalate preset's
     aggregator_temperature, max_tokens=common max_tokens,
     timeout=slot_timeout, **_slot_runtime(escalate aggregator))`.
  6. usage.moa.cascade set for tiers 1/2 accordingly (include
     `tier1_candidate`).
- Keep `_save_proxy_trace` working: reference_outputs = voters (+tier-1
  entry when tier 2 ran), aggregator output = final acting text.

### 5. Tests — NEW `tests/hermes_cli/test_moa_cascade.py`
Mirror `tests/hermes_cli/test_moa_proxy_server.py` conventions exactly:
same `_response`/`_usage` fakes, a `fake_llm` fixture patching BOTH
`moa_server.call_llm` and `"agent.moa_loop.call_llm"`, handlers keyed by
`task`, aiohttp TestServer/TestClient helpers, `HERMES_HOME` tmp fixture
writing a cascade config:
```yaml
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade: {escalate_to: big}
      reference_models: [voter-a(openrouter), voter-b(openrouter)]
      aggregator: mid-model(openrouter)
    big:
      enabled: false
      reference_models: [unused]
      aggregator: big-model(openrouter)
```
Required tests (names indicative):
1. unit: extract/normalize/consensus/agrees table cases (ints, `ANSWER: 3/6`
   → `1/2`, boxed, None, min_consensus=2 of 3, 80-char guard).
2. tier-0 consensus end-to-end: both voter handlers return "ANSWER: 42" →
   HTTP 200; content contains the voter text; NO `moa_aggregator` call in
   fake_llm.calls; `usage.moa.cascade["tier"] == 0`.
3. voters got the client's message verbatim with NO advisory system prompt
   (inspect recorded reference-task calls' messages).
4. disagreement → aggregator ran once; its guidance message contains both
   voter texts; when agg answer matches voter-b → tier 1, no big-model call.
5. discord (agg answer matches neither) → big-model called;
   `usage.moa.cascade["tier"] == 2`; big-model's guidance includes the
   tier1-aggregator output.
6. `escalate_to: nonexistent` → normalized None → discord still returns
   tier 1 (no big-model call).
7. request WITH tools on the cascade preset → fanout behavior (aggregator
   called with tools; no "cascade" key in usage.moa).
8. config: cascade preset with 1 reference slot → normalized mode "fanout",
   cascade None.
9. moa_loop direct=True: `_run_references_parallel(..., direct=True)` calls
   carry no `_REFERENCE_SYSTEM_PROMPT` (unit-level with patched call_llm).

### 6. NEW `scripts/moa_cascade_bench.py` (HTTP driver)
- `--base` (default `http://127.0.0.1:8652/v1`), `--models`
  (comma-separated model ids e.g. `moa:cascade-wafer,moa:cere-moa`),
  `--dataset aime|hmmt` (import `fetch_aime` from `moa_hard_bench` and
  `fetch_hmmt`, `extract_answer`/`is_correct` graders from the matching
  module — `sys.path.insert(0, scripts dir)` like the other benches),
  `--out`, `--workers` (default 8).
- POST non-streaming with the SAME ANSWER-instruction as the source bench
  module; max_tokens 16000; read `usage.moa.cascade` (may be absent for
  non-cascade models) and record: correct, latency_s, tier, total_tokens
  (usage.total_tokens).
- Final report per model: accuracy, tier histogram, mean AND median
  latency, mean tokens. Incremental JSON writes every 20 turns.

## Bench plan (run by the orchestrator after integration)
Serve config `cascade-wafer`: voters `custom:cerebras gpt-oss-120b` +
`custom:cerebras gemma-4-31b`, aggregator `custom:cerebras zai-glm-4.7`,
`cascade.escalate_to: gpt55-plan-lane` (openai-codex gpt-5.5),
`reference_max_tokens: 8000` (voters must finish real answers),
quorum_grace 0.5. Datasets: AIME-60 then HMMT-20. Baselines already
measured: gemma31 77%@1.7s · cere-moa 90%@19s · plan-gpt5.5 98%@40s ·
fable 100%@24s. Success = ≥93% AIME with median latency ≤8 s and tier-2
fraction ≤20%.
