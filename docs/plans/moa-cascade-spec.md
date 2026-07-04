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

## Addendum v1.1 — judge gate for freeform traffic (iteration 18)

Problem: exact-match consensus only fires on short comparable answers;
freeform requests always fall through to tier 1, paying aggregation even
when the voters substantively agree.

### Config
`cascade.gate: "exact" | "judge"` (normalized key `gate`, default
`"exact"` = current behavior). Optional `cascade.judge: {provider, model}`
slot (cleaned via `_clean_slot`); when absent, the judge defaults to the
ROUTER classifier slot if the router is enabled, else gate falls back to
"exact" (normalize this fallback at config time so the server never has to
guess: normalized `gate` is only "judge" when a judge slot resolved).

### Semantics (server, `_run_cascade_turn`)
With `gate == "judge"`:
1. Exact consensus is still tried FIRST (it is free). Fires → tier 0
   exactly as today (`cascade.gate_used = "exact"` in the usage block).
2. Else, collect voters whose full output is non-boilerplate (reuse the
   boilerplate guard). If >= min(2, min_consensus) remain, make ONE judge
   call via `call_llm(task="moa_router", ..., temperature=0.0,
   max_tokens=8, timeout=min(30, slot_timeout), **_slot_runtime(judge))`:
   system: "You compare answers for substantive agreement. Reply with
   exactly one word: CONSISTENT if they give the same answer/conclusion,
   DIFFERENT otherwise."
   user: request tail (last user message, cap 1500 chars) + "Answer A:\n" +
   voter[0] text (cap 2000) + "\n\nAnswer B:\n" + voter[1] text (cap 2000).
   (With >2 voters judge the FIRST TWO non-boilerplate outputs only — v1.)
3. Reply parsing: strip/upper; startswith "CONSISTENT" → tier-0 return of
   the FIRST judged voter's full text, `cascade.gate_used = "judge"`,
   `consensus = null`. Anything else (incl. judge error/timeout — wrap in
   try/except) → tier 1 as today (`gate_used = "judge-different"` or
   "judge-error").
4. Tier-2 logic unchanged (exact candidates only; freeform discord never
   escalates in v1).

### Usage surface
`usage.moa.cascade` gains `"gate_used": "exact" | "judge" |
"judge-different" | "judge-error" | null` (null when gate is exact-only
and no consensus). Judge call usage: fold into the reference usage total
as an extra `_RefAccounting` entry labelled `"consensus-judge — <slot>"`
ONLY when the judge ran (so billing stays truthful).

### Tests (append to tests/hermes_cli/test_moa_cascade.py)
- judge fires on freeform agreement: two voters return long prose with the
  same conclusion, no ANSWER lines; judge handler returns "CONSISTENT" →
  tier 0, gate_used "judge", no aggregator call; judge call visible in
  fake_llm.calls with task "moa_router".
- judge says DIFFERENT → tier 1, aggregator ran, gate_used "judge-different".
- judge raises → tier 1, gate_used "judge-error".
- exact consensus still wins WITHOUT a judge call when candidates match.
- config: gate judge with no judge slot and no router → normalized gate
  "exact"; with router enabled → judge defaults to classifier slot.

### Eval (scripts/moa_judge_gate_eval.py — NEW)
Freeform A/B without gold labels: N=30 general-knowledge/explanation
prompts (embedded list, mixed difficulty). For each prompt query TWO
models via HTTP: the judge-gated cascade and a comparison model
(--baseline, e.g. moa:cere-moa). Record latency + tier/gate_used. Then
blind pairwise quality judging: for each prompt send both answers
(shuffled A/B) to --grader (default moa:fable-lane... use
openrouter fable via a grader-capable serve preset; the script just needs
a model id reachable at --base) asking for "A", "B", or "TIE". Report:
win/tie/loss, tier-0 rate, median latency both sides.

## Addendum v1.2 — verified cascade (iteration 19)

Residual failure mode (measured, cascade-wafer4 on AIME): correlated
confidence — 2 false weak-consensus returns + 2 confident-but-wrong tier-1
agreements. Verification is orthogonal to agreement (+5 pp measured in the
verifier bench) and attacks exactly this.

### Config (normalized under the existing `cascade` block)
- `verify: "python" | absent` (default absent = off).
- `verifier: {provider, model}` slot; default chain at CONFIG time:
  explicit slot → `cascade.judge` slot (if resolved) → router classifier
  (if router enabled) → verify disabled (normalized `verify` becomes None).
- `verify_when: "weak" | "always"` (default "weak").

### Semantics (server, `_run_cascade_turn`, exact-candidate flows only)
Verification of (problem, candidate_text, candidate_answer):
1. One `call_llm(task="moa_verifier", temperature=0.0, max_tokens=2000,
   timeout=min(60, slot_timeout), **_slot_runtime(verifier_slot))` with:
   system: "You write a short standalone Python 3 program that CHECKS a
   candidate answer. The program must recompute or verify the answer
   independently and print exactly one final line: VERDICT: CORRECT or
   VERDICT: WRONG. If the claim cannot be checked by computation, print
   VERDICT: UNCHECKABLE. No network, no files, stdlib only, under 5
   seconds of compute."
   user: problem statement (last user message, cap 4000 chars) +
   "\n\nCandidate answer: " + candidate.
2. Extract the first ```python fenced block (or whole reply if none);
   execute via `subprocess.run([sys.executable, "-I", "-c", code],
   capture_output=True, text=True, timeout=12, env={}, cwd=<tempdir>)`
   in a helper `run_verification(code) -> str` placed in moa_cascade.py
   (module gains this one I/O function; keep pure helpers pure otherwise).
3. Parse the LAST `VERDICT:\s*(CORRECT|WRONG|UNCHECKABLE)` from stdout.
   WRONG → verdict "wrong". CORRECT → "correct". Anything else (no
   verdict, exec error, timeout, LLM error) → "inconclusive" — NEVER block
   the cascade on verification infrastructure failure.

Gate integration:
- Tier 0 exact consensus: if `verify_when == "weak"`, verify ONLY when
  consensus votes < number of reference slots (i.e. non-unanimous);
  "always" verifies every consensus. Verdict "wrong" → strike: record the
  struck value, proceed to tier 1 as if no consensus (aggregator guidance
  additionally gets one appended reference entry labelled
  `"verifier — <slot>"` whose text is "Automated check REJECTED the
  consensus answer <value>:\n<last 500 chars of stdout>"). Other verdicts →
  return tier 0 as usual.
- Tier 1 (exact-gate flows only, judge-gated freeform skips verification):
  verify the aggregator's candidate (respecting verify_when: "weak" =
  always verify tier 1 — tier 1 is already slow; the knob only guards the
  fast path). Verdict "wrong" AND escalate preset configured → escalate to
  tier 2 (regardless of voter agreement), with the verifier note appended
  to the tier-2 guidance. Verdict "wrong" without escalate → return tier 1
  anyway (surface the verdict).
- Tier 2 output is never verified (top of the ladder).

Surface: `usage.moa.cascade.verify = {"ran": true|false, "verdict":
"correct"|"wrong"|"inconclusive"|null, "on": "consensus"|"tier1"|null}`
(+ verifier LLM usage folded as a reference entry labelled
`"verifier — <slot>"` when it ran). `struck_consensus: <value|null>` in the
cascade block when a strike happened.

### Tests (append to test_moa_cascade.py; fake the LLM as usual AND
monkeypatch `moa_cascade.run_verification` — no real subprocess in tests
except one direct unit test of run_verification with trivial code)
- weak consensus (2-of-3... use a 3-slot preset with min_consensus 2) +
  verifier says WRONG → tier 1 runs; struck_consensus set; aggregator
  guidance contains the verifier rejection note.
- weak consensus + CORRECT → tier 0, verify.verdict "correct".
- unanimous consensus + verify_when weak → NO verifier call, verify.ran
  false.
- tier-1 answer + verifier WRONG + escalate configured → tier 2, and
  verify.on == "tier1".
- verifier LLM raises / run_verification returns no verdict → verdict
  "inconclusive", cascade proceeds normally (tier 0 returned).
- run_verification unit: code printing VERDICT: CORRECT → "correct";
  code raising → "inconclusive"; timeout code (sleep) → "inconclusive".
- config: verify python with no resolvable verifier slot → verify None.

### Bench
cascade-wafer4v = cascade-wafer4 + verify python (verifier = gemma-4-31b
cerebras), verify_when weak. AIME-60 + HMMT-20 via moa_cascade_bench
(record usage.moa.cascade.verify in rows — extend the bench script to
carry the whole cascade block through to the JSON). Success: AIME ≥58/60,
median ≤7 s, frontier ≤10%.
