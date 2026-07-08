"""Tests for cascade mode (`mode: cascade`) — lazy MoA with observation-based
gating (see docs/plans/moa-cascade-spec.md).

Conventions mirror tests/hermes_cli/test_moa_proxy_server.py exactly: fake
call_llm patched in both moa_server and agent.moa_loop, aiohttp
TestServer/TestClient, HERMES_HOME pointed at a tmp config.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

import hermes_cli.proxy.moa_server as moa_server
from hermes_cli.proxy.moa_server import create_moa_app
from hermes_cli.proxy.moa_cascade import (
    agrees,
    consensus,
    extract_candidate,
    normalize_candidate,
)


# ---------------------------------------------------------------------------
# Fakes (verbatim conventions from test_moa_proxy_server.py)
# ---------------------------------------------------------------------------


def _usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=None
    )


def _response(content="done", *, tool_calls=None, reasoning=None, finish_reason=None):
    message = SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        reasoning_content=reasoning,
    )
    choice = SimpleNamespace(
        message=message,
        finish_reason=finish_reason or ("tool_calls" if tool_calls else "stop"),
    )
    return SimpleNamespace(choices=[choice], usage=_usage(), model="fake-model")


def _write_cascade_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        escalate_to: big
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
    big:
      enabled: false
      reference_models:
        - provider: openrouter
          model: unused
      aggregator:
        provider: openrouter
        model: big-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_cascade_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def _write_bad_escalate_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        escalate_to: nonexistent
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_bad_escalate(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_bad_escalate_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.fixture()
def fake_llm(monkeypatch):
    """Patch call_llm in both moa_server (streaming path) and agent.moa_loop
    (non-streaming reference fan-out). Returns the recorded calls list."""
    calls: list[dict] = []
    handlers: dict[str, object] = {}

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        handler = handlers.get(kwargs.get("task"))
        if callable(handler):
            return handler(kwargs)
        if kwargs.get("task") == "moa_reference":
            return _response("reference advice")
        return _response("aggregator acted")

    monkeypatch.setattr(moa_server, "call_llm", fake_call_llm)
    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    fake_call_llm.calls = calls
    fake_call_llm.handlers = handlers
    return fake_call_llm


async def _client(app):
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# 1. unit: extract_candidate / normalize_candidate / consensus / agrees
# ---------------------------------------------------------------------------


def test_cascade_helper_table():
    # extract_candidate: ANSWER: pattern, last match wins.
    assert extract_candidate("some steps\nANSWER: 42") == "42"
    assert extract_candidate("first ANSWER: 41\nsecond ANSWER: 42") == "42"

    # extract_candidate: boxed fallback when no ANSWER: present.
    assert extract_candidate("some steps\n\\boxed{7}") == "7"

    # extract_candidate: last non-empty line <=80 chars, else None.
    short_line = "final answer text"
    assert extract_candidate(f"reasoning here\n{short_line}") == short_line
    long_line = "x" * 81
    assert extract_candidate(f"reasoning here\n{long_line}") is None

    # extract_candidate: strips backticks/markdown emphasis/trailing punctuation.
    assert extract_candidate("`42`.") == "42"
    assert extract_candidate("**42**") == "42"

    # extract_candidate: nothing extractable -> None.
    assert extract_candidate("") is None

    # normalize_candidate: ints canonicalized, fractions reduced, text cleaned.
    assert normalize_candidate("42") == "42"
    assert normalize_candidate("$42") == "42"
    assert normalize_candidate("007") == "7"
    assert normalize_candidate("-08") == "-8"
    assert normalize_candidate("3/6") == "1/2"
    assert normalize_candidate("\\frac{4}{8}") == "1/2"
    assert normalize_candidate("  Hello   World  ") == "hello world"
    assert normalize_candidate('"42"') == "42"

    # extract_candidate: ANSWER: 3/6 -> normalize_candidate -> 1/2.
    assert extract_candidate("work shown\nANSWER: 3/6") == "3/6"
    assert normalize_candidate(extract_candidate("work shown\nANSWER: 3/6")) == "1/2"

    # consensus: ignores None, groups by normalized value, honors min_consensus.
    assert consensus(["42", "42", "43"], 2) == "42"
    assert consensus(["42", "43", "44"], 2) is None
    assert consensus([None, "42", "42"], 2) == "42"
    assert consensus(["42"], 2) is None
    assert consensus(["42", "42", "42"], 3) == "42"
    assert consensus(["42", "42"], 3) is None

    # agrees: both non-None and normalize equal.
    assert agrees("42", " 42 ") is True
    assert agrees(None, "42") is False
    assert agrees("42", None) is False
    assert agrees("3/6", "1/2") is True
    assert agrees("42", "43") is False


# ---------------------------------------------------------------------------
# 2. tier-0 consensus end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier0_consensus_returns_first_voter_no_aggregator(moa_home, fake_llm):
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response(
        "Reasoning...\nANSWER: 42"
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 6*7?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()
        content = body["choices"][0]["message"]["content"]
        assert "ANSWER: 42" in content
        assert body["choices"][0]["finish_reason"] == "stop"

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert "moa_aggregator" not in tasks

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["consensus"] == "42"
        assert cascade["votes"] == 2
        assert cascade["candidates"] == ["42", "42"]
        assert len(cascade["voters"]) == 2
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 3. voters get the client message verbatim, no advisory system prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier0_voters_get_verbatim_no_advisory_prompt(moa_home, fake_llm):
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")
    client_messages = [{"role": "user", "content": "what is 6*7?"}]
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": client_messages},
        )
        assert resp.status == 200

        ref_calls = [c for c in fake_llm.calls if c["task"] == "moa_reference"]
        assert len(ref_calls) == 2
        for call in ref_calls:
            assert call["messages"] == client_messages
            assert all(m.get("role") != "system" for m in call["messages"])
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 4. disagreement -> aggregator runs; matches voter-b -> tier 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier1_disagreement_matches_voter_b(moa_home, fake_llm):
    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("First path.\nANSWER: 1")
        return _response("Second path.\nANSWER: 2")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        "Tier1 synthesis.\nANSWER: 2"
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 1 or 2?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        agg_text = str(agg_calls[0]["messages"])
        assert "First path" in agg_text
        assert "Second path" in agg_text

        content = body["choices"][0]["message"]["content"]
        assert "Tier1 synthesis" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1

        # No escalation: the big-model preset's aggregator never ran.
        assert not any(c.get("model") == "big-model" for c in fake_llm.calls)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 5. discord (agg matches neither) -> escalate to big preset's aggregator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tier2_escalates_when_aggregator_matches_neither(moa_home, fake_llm):
    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("First path.\nANSWER: 1")
        return _response("Second path.\nANSWER: 2")

    def agg_handler(kwargs):
        if kwargs.get("model") == "big-model":
            return _response("Escalated final.\nANSWER: 3")
        return _response("Tier1 discord.\nANSWER: 5")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = agg_handler

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 1, 2 or 5?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        big_calls = [c for c in fake_llm.calls if c.get("model") == "big-model"]
        assert len(big_calls) == 1
        big_text = str(big_calls[0]["messages"])
        assert "tier1-aggregator" in big_text
        assert "Tier1 discord" in big_text

        content = body["choices"][0]["message"]["content"]
        assert "Escalated final" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 2
        assert cascade.get("tier1_candidate") == "5"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 6. escalate_to names a nonexistent preset -> normalized None -> stays tier 1
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_to_nonexistent_normalizes_to_none_stays_tier1(
    moa_home_bad_escalate, fake_llm
):
    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("First path.\nANSWER: 1")
        return _response("Second path.\nANSWER: 2")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        "Tier1 discord.\nANSWER: 5"
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 1, 2 or 5?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1

        content = body["choices"][0]["message"]["content"]
        assert "Tier1 discord" in content

        # No escalate preset configured -> nothing named "nonexistent" or
        # otherwise ran a second aggregator call.
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 7. request WITH tools on the cascade preset -> acting-model solo path
#    (superseded by addendum v1.4: tool-carrying requests no longer use
#    the fanout path -- see test_cascade_tools_non_streaming_is_tool_solo
#    below for the full contract).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_with_tools_uses_fanout(moa_home_tool_turns_solo, fake_llm):
    # Addendum v1.5: the default `tool_turns: "detect"` re-checks a fresh
    # user turn via the VOTER GATE instead of bypassing unconditionally (see
    # the "Addendum v1.5" test section below), so this scenario now needs
    # the explicit `tool_turns: solo` opt-out to exercise the unconditional
    # v1.4 bypass this test is actually checking.
    tools = [
        {
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}},
        }
    ]
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascsolo",
                "messages": [{"role": "user", "content": "look something up"}],
                "tools": tools,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        assert agg_calls[0]["tools"] == tools

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_reference" not in tasks
        assert body["usage"]["moa"]["cascade"] == {"tier": None, "mode": "tool-solo"}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 8. config: <2 reference slots on a cascade preset downgrades to fanout
# ---------------------------------------------------------------------------


def test_single_reference_slot_downgrades_mode_to_fanout():
    from hermes_cli.moa_config import normalize_moa_config

    raw = {
        "default_preset": "casc1",
        "presets": {
            "casc1": {
                "mode": "cascade",
                "cascade": {"escalate_to": "big"},
                "reference_models": [
                    {"provider": "openrouter", "model": "only-voter"}
                ],
                "aggregator": {"provider": "openrouter", "model": "agg-model"},
            },
        },
    }
    cfg = normalize_moa_config(raw)
    preset = cfg["presets"]["casc1"]
    assert preset["mode"] == "fanout"
    assert preset["cascade"] is None


# ---------------------------------------------------------------------------
# 9. moa_loop direct=True omits the advisory reference system prompt
# ---------------------------------------------------------------------------


def test_moa_loop_direct_true_omits_reference_system_prompt(fake_llm):
    import agent.moa_loop as moa_loop

    slots = [
        {"provider": "openrouter", "model": "voter-a"},
        {"provider": "openrouter", "model": "voter-b"},
    ]
    ref_messages = [{"role": "user", "content": "what is 6*7?"}]
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")

    outputs = moa_loop._run_references_parallel(
        slots,
        [dict(m) for m in ref_messages],
        temperature=0.6,
        max_tokens=None,
        timeout=None,
        direct=True,
    )
    assert len(outputs) == 2

    ref_calls = [c for c in fake_llm.calls if c["task"] == "moa_reference"]
    assert len(ref_calls) == 2
    for call in ref_calls:
        assert call["messages"] == ref_messages
        assert all(m.get("role") != "system" for m in call["messages"])
        assert not any(
            moa_loop._REFERENCE_SYSTEM_PROMPT in str(m.get("content"))
            for m in call["messages"]
        )


# ---------------------------------------------------------------------------
# Review-finding regressions (2026-07-04 adversarial review)
# ---------------------------------------------------------------------------


def test_boxed_extraction_is_nesting_aware():
    """A naive [^}]* regex truncated \\boxed{\\frac{1}{2}} at the first inner
    brace, collapsing different fractions into a false consensus."""
    from hermes_cli.proxy.moa_cascade import consensus, extract_candidate

    a = extract_candidate("thus \\boxed{\\frac{1}{2}}")
    b = extract_candidate("thus \\boxed{\\frac{1}{3}}")
    assert a == "\\frac{1}{2}"
    assert b == "\\frac{1}{3}"
    assert consensus([a, b], 2) is None  # genuinely different answers


def test_failure_boilerplate_never_votes():
    """Identical failure/drop notes across voters must not manufacture a
    tier-0 consensus (a shared provider outage would otherwise become the
    client-facing 'answer')."""
    from hermes_cli.proxy.moa_cascade import consensus, extract_candidate

    dropped = "[dropped: reference exceeded the quorum deadline]"
    failed = "[failed: Error code: 429 - overloaded]"
    assert extract_candidate(dropped) is None
    assert extract_candidate(failed) is None
    votes = [extract_candidate(dropped), extract_candidate(dropped),
             extract_candidate("ANSWER: 42")]
    assert consensus(votes, 2) is None


def _write_judge_gate_cfg(home):
    """Cascade preset with `cascade.gate: judge` and NO explicit judge slot,
    plus a router block (classifier + one routable preset) so the addendum's
    config-time fallback resolves `judge` to the router's classifier slot.
    The client always addresses `moa:casc` directly in these tests — the
    router block exists purely to source the judge default, mirroring how
    tests/hermes_cli/test_moa_router.py builds router configs."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  router:
    enabled: true
    classifier:
      provider: openrouter
      model: judge-classifier
    default: casc
  presets:
    casc:
      mode: cascade
      cascade:
        escalate_to: big
        gate: judge
      route:
        description: cascade freeform judge testing
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
    big:
      enabled: false
      reference_models:
        - provider: openrouter
          model: unused
      aggregator:
        provider: openrouter
        model: big-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_judge_gate(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_judge_gate_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


# Long freeform prose with no ANSWER:/\boxed{} markers and no short (<=80
# char) final line, so extract_candidate() -> None for both voters and exact
# consensus never fires -- these exist to drive the addendum v1.1 judge gate.
_PROSE_A = (
    "Photosynthesis converts light energy into chemical energy stored in "
    "glucose, forming the base of most food chains and releasing the oxygen "
    "aerobic organisms depend on for cellular respiration."
)
_PROSE_B = (
    "Plants use photosynthesis to turn sunlight into chemical energy stored "
    "as glucose, which underlies nearly every food chain and supplies the "
    "oxygen animals need to breathe."
)


# ---------------------------------------------------------------------------
# Addendum v1.1 -- judge gate for freeform traffic (iteration 18)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_gate_fires_on_freeform_agreement(moa_home_judge_gate, fake_llm):
    """Two voters give long same-conclusion prose (no ANSWER: lines) -- exact
    consensus can't see it, but the judge call says CONSISTENT -> tier 0."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        return _response(_PROSE_A if model == "voter-a" else _PROSE_B)

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_router"] = lambda kwargs: _response("CONSISTENT")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "explain photosynthesis"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert tasks.count("moa_router") == 1
        assert "moa_aggregator" not in tasks

        judge_calls = [c for c in fake_llm.calls if c["task"] == "moa_router"]
        judge_kwargs = judge_calls[0]
        assert judge_kwargs.get("model") == "judge-classifier"
        assert judge_kwargs.get("temperature") == 0.0
        assert judge_kwargs.get("max_tokens") == 8
        judge_text = str(judge_kwargs["messages"])
        assert "CONSISTENT" in judge_text and "DIFFERENT" in judge_text
        assert "Answer A" in judge_text
        assert "Answer B" in judge_text
        assert _PROSE_A in judge_text
        assert _PROSE_B in judge_text

        content = body["choices"][0]["message"]["content"]
        assert _PROSE_A in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["gate_used"] == "judge"
        assert cascade["consensus"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_judge_gate_different_falls_to_tier1(moa_home_judge_gate, fake_llm):
    """Judge says DIFFERENT -> tier 1: aggregator runs exactly as today."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        return _response(_PROSE_A if model == "voter-a" else _PROSE_B)

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_router"] = lambda kwargs: _response("DIFFERENT")
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        "Tier1 synthesis of the two views."
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "explain photosynthesis"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_router") == 1
        assert tasks.count("moa_aggregator") == 1

        content = body["choices"][0]["message"]["content"]
        assert "Tier1 synthesis" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1
        assert cascade["gate_used"] == "judge-different"

        # No escalation: the big-model preset's aggregator never ran.
        assert not any(c.get("model") == "big-model" for c in fake_llm.calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_judge_gate_error_falls_to_tier1(moa_home_judge_gate, fake_llm):
    """Judge call raises -> wrapped, degrades to tier 1 (never crashes the
    turn), gate_used records the distinct 'judge-error' reason."""

    def raising_judge(kwargs):
        raise RuntimeError("judge upstream failure")

    def ref_handler(kwargs):
        model = kwargs.get("model")
        return _response(_PROSE_A if model == "voter-a" else _PROSE_B)

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_router"] = raising_judge
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        "Tier1 fallback synthesis."
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "explain photosynthesis"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        content = body["choices"][0]["message"]["content"]
        assert "Tier1 fallback synthesis" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1
        assert cascade["gate_used"] == "judge-error"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_judge_gate_exact_consensus_skips_judge_call(
    moa_home_judge_gate, fake_llm
):
    """Exact consensus is tried first and is free: matching ANSWER: lines
    must win at tier 0 WITHOUT ever placing a judge (moa_router) call, even
    though this preset is gate: judge."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 6*7?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_router" not in tasks
        assert "moa_aggregator" not in tasks

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["gate_used"] == "exact"
        assert cascade["consensus"] == "42"
    finally:
        await client.close()


def test_judge_gate_config_normalization_defaults():
    """Config-time fallback (no server involved): `gate: judge` without an
    explicit judge slot normalizes to "exact" unless the router is enabled,
    in which case `judge` defaults to the router's classifier slot."""
    from hermes_cli.moa_config import normalize_moa_config

    presets = {
        "casc": {
            "mode": "cascade",
            "cascade": {"escalate_to": "big", "gate": "judge"},
            "reference_models": [
                {"provider": "openrouter", "model": "voter-a"},
                {"provider": "openrouter", "model": "voter-b"},
            ],
            "aggregator": {"provider": "openrouter", "model": "mid-model"},
        },
        "big": {
            "enabled": False,
            "reference_models": [{"provider": "openrouter", "model": "unused"}],
            "aggregator": {"provider": "openrouter", "model": "big-model"},
        },
    }

    # No judge slot, no router -> normalized gate falls back to "exact".
    cfg_no_router = normalize_moa_config(
        {"default_preset": "casc", "presets": presets}
    )
    cascade_no_router = cfg_no_router["presets"]["casc"]["cascade"]
    assert cascade_no_router["gate"] == "exact"
    assert cascade_no_router.get("judge") is None

    # Router enabled -> judge defaults to the router's classifier slot, and
    # gate stays "judge" because a judge slot now resolves.
    presets_routable = {
        **presets,
        "casc": {**presets["casc"], "route": {"description": "judge gate testing"}},
    }
    cfg_with_router = normalize_moa_config(
        {
            "default_preset": "casc",
            "router": {
                "enabled": True,
                "classifier": {"provider": "openrouter", "model": "judge-classifier"},
                "default": "casc",
            },
            "presets": presets_routable,
        }
    )
    assert cfg_with_router["router"]["enabled"] is True
    cascade_with_router = cfg_with_router["presets"]["casc"]["cascade"]
    assert cascade_with_router["gate"] == "judge"
    assert cascade_with_router["judge"] == cfg_with_router["router"]["classifier"]


@pytest.mark.asyncio
async def test_tier0_trace_attributes_winner_voter(moa_home, fake_llm, monkeypatch):
    """Trace records must attribute the acting text to the model that
    produced it (a voter at tier 0), not the preset's aggregator — evolve
    grades on this attribution."""
    captured = {}

    def fake_save_moa_turn(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("agent.moa_trace.save_moa_turn", fake_save_moa_turn)
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 7")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"]["tier"] == 0
        # Attribution: a voter model, not the cascade preset's aggregator.
        assert captured.get("aggregator_model") in {"voter-a", "voter-b"}
        assert captured.get("aggregator_model") != "mid-model"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Addendum v1.2 -- verified cascade (iteration 19)
#
# Config fixtures below use 3 reference slots with `min_consensus: 2` so a
# "weak" (non-unanimous, 2-of-3) consensus and a unanimous (3-of-3) consensus
# are both constructible from the same preset shape -- the addendum's
# `verify_when: "weak"` default treats those two cases differently (only the
# former pays for a verifier call). Per the task brief, end-to-end tests
# monkeypatch `hermes_cli.proxy.moa_cascade.run_verification` directly rather
# than exercising a real subprocess; only the dedicated run_verification unit
# tests near the bottom of this section use the real function.
# ---------------------------------------------------------------------------


def _write_verify_cascade_cfg(home):
    """3-voter cascade preset (`min_consensus: 2` => 2-of-3 is a WEAK
    majority, not unanimity) with an explicit `verify: python` + `verifier`
    slot, plus an `escalate_to` target -- covers every addendum v1.2 branch
    (weak-consensus strike, tier-1-forced escalation) from one config."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc3
  presets:
    casc3:
      mode: cascade
      cascade:
        escalate_to: big
        min_consensus: 2
        verify: python
        verifier:
          provider: openrouter
          model: verifier-model
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
        - provider: openrouter
          model: voter-c
      aggregator:
        provider: openrouter
        model: mid-model
    big:
      enabled: false
      reference_models:
        - provider: openrouter
          model: unused
      aggregator:
        provider: openrouter
        model: big-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_verify3(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_verify_cascade_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def _write_verify_cascade_cfg_no_escalate(home):
    """Same 3-voter verify config, but with no `escalate_to` target --
    isolates the tier-0-strike -> tier-1-final path from tier-2 escalation
    so a forced tier-1 re-verification (verify_when "weak" verifies tier 1
    unconditionally, per the addendum) can never additionally escalate and
    complicate the assertions for the pure strike behavior."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc3
  presets:
    casc3:
      mode: cascade
      cascade:
        min_consensus: 2
        verify: python
        verifier:
          provider: openrouter
          model: verifier-model
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
        - provider: openrouter
          model: voter-c
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_verify3_no_escalate(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_verify_cascade_cfg_no_escalate(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.mark.asyncio
async def test_verify_weak_consensus_correct_stays_tier0(
    moa_home_verify3, fake_llm, monkeypatch
):
    """2-of-3 consensus is WEAK (votes < len(reference_models)), so the
    default verify_when="weak" triggers exactly one verifier call; a
    CORRECT verdict changes nothing -- tier 0 still returns the consensus
    voter's text and no aggregator call is made."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-c":
            return _response("Different path.\nANSWER: 2")
        return _response("Shared path.\nANSWER: 1")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_verifier"] = lambda kwargs: _response(
        "```python\nprint('VERDICT: CORRECT')\n```"
    )
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_verification",
        lambda code: "correct",
        raising=False,
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is the shared answer?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_verifier") == 1
        assert "moa_aggregator" not in tasks

        verifier_calls = [c for c in fake_llm.calls if c["task"] == "moa_verifier"]
        assert verifier_calls[0].get("model") == "verifier-model"
        assert verifier_calls[0].get("temperature") == 0.0
        verifier_text = str(verifier_calls[0]["messages"])
        assert "Candidate answer: 1" in verifier_text

        content = body["choices"][0]["message"]["content"]
        assert "Shared path" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["consensus"] == "1"
        assert cascade["votes"] == 2
        assert cascade["verify"] == {"ran": True, "verdict": "correct", "on": "consensus"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_unanimous_consensus_skips_verifier_call(
    moa_home_verify3, fake_llm, monkeypatch
):
    """All 3 voters agree (unanimity: votes == len(reference_models)) --
    verify_when "weak" only checks NON-unanimous consensus, so no verifier
    call is made at all and verify.ran is false."""
    run_verification_calls = []
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_verification",
        lambda code: run_verification_calls.append(code) or "correct",
        raising=False,
    )
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is 6*7?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_verifier" not in tasks
        assert run_verification_calls == []

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["votes"] == 3
        assert cascade["verify"] == {"ran": False, "verdict": None, "on": None}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_weak_consensus_wrong_strikes_to_tier1(
    moa_home_verify3_no_escalate, fake_llm, monkeypatch
):
    """A 2-of-3 consensus verified WRONG must not be returned -- the struck
    value is recorded and the cascade proceeds to tier 1 as if there had
    been no consensus at all; the aggregator's guidance carries an explicit
    rejection note naming the struck answer."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-c":
            return _response("Different path.\nANSWER: 3")
        return _response("Shared path.\nANSWER: 1")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_verifier"] = lambda kwargs: _response(
        "```python\nprint('VERDICT: WRONG')\n```"
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        "Tier1 after strike.\nANSWER: 3"
    )
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_verification",
        lambda code: "wrong",
        raising=False,
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is the shared answer?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        agg_text = str(agg_calls[0]["messages"])
        assert "REJECTED" in agg_text
        assert "consensus" in agg_text and "1" in agg_text
        assert "verifier —" in agg_text

        content = body["choices"][0]["message"]["content"]
        assert "Tier1 after strike" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1
        assert cascade["struck_consensus"] == "1"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_tier1_wrong_escalates_to_tier2(
    moa_home_verify3, fake_llm, monkeypatch
):
    """No exact consensus among the 3 voters; the aggregator's tier-1
    candidate agrees with one voter (which alone would settle at tier 1
    under the addendum v1.1 rules) -- but a verified WRONG verdict forces
    escalation regardless of that agreement, since an escalate preset is
    configured, and the verifier note rides along in the tier-2 guidance."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("Path A.\nANSWER: 1")
        if model == "voter-b":
            return _response("Path B.\nANSWER: 2")
        return _response("Path C.\nANSWER: 3")

    def agg_handler(kwargs):
        if kwargs.get("model") == "big-model":
            return _response("Escalated after verify.\nANSWER: 9")
        return _response("Tier1 agrees with B.\nANSWER: 2")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = agg_handler
    fake_llm.handlers["moa_verifier"] = lambda kwargs: _response(
        "```python\nprint('VERDICT: WRONG')\n```"
    )
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_verification",
        lambda code: "wrong",
        raising=False,
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is 1, 2 or 3?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        big_calls = [c for c in fake_llm.calls if c.get("model") == "big-model"]
        assert len(big_calls) == 1
        assert "verifier" in str(big_calls[0]["messages"]).lower()

        content = body["choices"][0]["message"]["content"]
        assert "Escalated after verify" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 2
        assert cascade.get("tier1_candidate") == "2"
        assert cascade["verify"] == {"ran": True, "verdict": "wrong", "on": "tier1"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_llm_call_raises_is_inconclusive(moa_home_verify3, fake_llm):
    """The verifier's own call_llm() raising (upstream error/timeout) must
    never block the cascade: it degrades to an "inconclusive" verdict and
    tier 0 still returns the consensus exactly as an unverified weak
    consensus would."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-c":
            return _response("Different path.\nANSWER: 2")
        return _response("Shared path.\nANSWER: 1")

    def raising_verifier(kwargs):
        raise RuntimeError("verifier upstream failure")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_verifier"] = raising_verifier

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is the shared answer?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        assert "moa_aggregator" not in [c["task"] for c in fake_llm.calls]

        content = body["choices"][0]["message"]["content"]
        assert "Shared path" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["verify"] == {
            "ran": True,
            "verdict": "inconclusive",
            "on": "consensus",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_verify_run_verification_no_verdict_is_inconclusive(
    moa_home_verify3, fake_llm, monkeypatch
):
    """run_verification itself finding no parseable VERDICT line (exec
    error/timeout inside the sandboxed check) also degrades to
    "inconclusive" without blocking the cascade -- exercised separately from
    the verifier LLM call raising above, since it is a distinct failure
    point in the pipeline."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-c":
            return _response("Different path.\nANSWER: 2")
        return _response("Shared path.\nANSWER: 1")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_verifier"] = lambda kwargs: _response(
        "```python\nprint('no parseable verdict here')\n```"
    )
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_verification",
        lambda code: "inconclusive",
        raising=False,
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc3",
                "messages": [{"role": "user", "content": "what is the shared answer?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()

        content = body["choices"][0]["message"]["content"]
        assert "Shared path" in content

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["verify"] == {
            "ran": True,
            "verdict": "inconclusive",
            "on": "consensus",
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# run_verification direct unit tests -- real subprocess, no monkeypatching
# (per the task brief: end-to-end tests fake run_verification, but these
# exercise the actual sandboxed-execution helper against trivial code).
# ---------------------------------------------------------------------------


def test_run_verification_correct_verdict_real_subprocess():
    from hermes_cli.proxy.moa_cascade import run_verification

    assert run_verification("print('VERDICT: CORRECT')") == "correct"


def test_run_verification_wrong_verdict_real_subprocess():
    from hermes_cli.proxy.moa_cascade import run_verification

    assert run_verification("print('VERDICT: WRONG')") == "wrong"


def test_run_verification_exec_error_is_inconclusive_real_subprocess():
    from hermes_cli.proxy.moa_cascade import run_verification

    assert run_verification("raise ValueError('boom')") == "inconclusive"


def test_run_verification_timeout_is_inconclusive_real_subprocess():
    from hermes_cli.proxy.moa_cascade import run_verification

    assert run_verification("import time\ntime.sleep(20)") == "inconclusive"


# ---------------------------------------------------------------------------
# Config: verifier slot fallback chain (explicit -> judge slot -> router
# classifier -> disabled), mirroring the judge-gate fallback chain tests.
# ---------------------------------------------------------------------------


def test_verify_config_no_resolvable_verifier_slot_disables_verify():
    """`verify: python` with no explicit verifier slot, no judge slot, and no
    router configured has nothing to call -- normalizes to fully disabled,
    mirroring the judge-gate fallback's terminal "nothing to call" case."""
    from hermes_cli.moa_config import normalize_moa_config

    presets = {
        "casc": {
            "mode": "cascade",
            "cascade": {"escalate_to": "big", "verify": "python"},
            "reference_models": [
                {"provider": "openrouter", "model": "voter-a"},
                {"provider": "openrouter", "model": "voter-b"},
            ],
            "aggregator": {"provider": "openrouter", "model": "mid-model"},
        },
        "big": {
            "enabled": False,
            "reference_models": [{"provider": "openrouter", "model": "unused"}],
            "aggregator": {"provider": "openrouter", "model": "big-model"},
        },
    }
    cfg = normalize_moa_config({"default_preset": "casc", "presets": presets})
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["verify"] is None
    assert cascade.get("verifier") is None


def test_verify_config_fallback_chain_explicit_then_judge_then_router():
    """Config-time verifier resolution chain (addendum v1.2): an explicit
    `cascade.verifier` slot wins outright; absent that, a resolved
    `cascade.judge` slot is reused; absent both, the router's classifier
    (when routing is enabled) is the last resort before disabling verify."""
    from hermes_cli.moa_config import normalize_moa_config

    base_presets = {
        "casc": {
            "mode": "cascade",
            "cascade": {"escalate_to": "big", "verify": "python"},
            "route": {"description": "verify fallback chain testing"},
            "reference_models": [
                {"provider": "openrouter", "model": "voter-a"},
                {"provider": "openrouter", "model": "voter-b"},
            ],
            "aggregator": {"provider": "openrouter", "model": "mid-model"},
        },
        "big": {
            "enabled": False,
            "reference_models": [{"provider": "openrouter", "model": "unused"}],
            "aggregator": {"provider": "openrouter", "model": "big-model"},
        },
    }

    # 1) Explicit verifier slot always wins.
    explicit_presets = {
        **base_presets,
        "casc": {
            **base_presets["casc"],
            "cascade": {
                **base_presets["casc"]["cascade"],
                "verifier": {"provider": "openrouter", "model": "explicit-verifier"},
            },
        },
    }
    cfg = normalize_moa_config({"default_preset": "casc", "presets": explicit_presets})
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["verify"] == "python"
    assert cascade["verifier"] == {"provider": "openrouter", "model": "explicit-verifier"}

    # 2) No explicit verifier, but gate: judge resolved its own slot -> reused.
    judge_presets = {
        **base_presets,
        "casc": {
            **base_presets["casc"],
            "cascade": {
                **base_presets["casc"]["cascade"],
                "gate": "judge",
                "judge": {"provider": "openrouter", "model": "judge-slot"},
            },
        },
    }
    cfg = normalize_moa_config({"default_preset": "casc", "presets": judge_presets})
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["verify"] == "python"
    assert cascade["verifier"] == {"provider": "openrouter", "model": "judge-slot"}

    # 3) No explicit verifier, no judge slot, router enabled -> classifier.
    cfg = normalize_moa_config(
        {
            "default_preset": "casc",
            "router": {
                "enabled": True,
                "classifier": {"provider": "openrouter", "model": "router-classifier"},
                "default": "casc",
            },
            "presets": base_presets,
        }
    )
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["verify"] == "python"
    assert cascade["verifier"] == {"provider": "openrouter", "model": "router-classifier"}


def test_verify_when_normalization_default_and_explicit():
    """`verify_when` defaults to "weak"; an explicit "always" is preserved;
    anything unrecognized falls back to the "weak" default."""
    from hermes_cli.moa_config import normalize_moa_config

    def _cfg(verify_when=None):
        cascade = {
            "escalate_to": "big",
            "verify": "python",
            "verifier": {"provider": "openrouter", "model": "verifier-model"},
        }
        if verify_when is not None:
            cascade["verify_when"] = verify_when
        return {
            "default_preset": "casc",
            "presets": {
                "casc": {
                    "mode": "cascade",
                    "cascade": cascade,
                    "reference_models": [
                        {"provider": "openrouter", "model": "voter-a"},
                        {"provider": "openrouter", "model": "voter-b"},
                    ],
                    "aggregator": {"provider": "openrouter", "model": "mid-model"},
                },
                "big": {
                    "enabled": False,
                    "reference_models": [{"provider": "openrouter", "model": "unused"}],
                    "aggregator": {"provider": "openrouter", "model": "big-model"},
                },
            },
        }

    cfg = normalize_moa_config(_cfg())
    assert cfg["presets"]["casc"]["cascade"]["verify_when"] == "weak"

    cfg = normalize_moa_config(_cfg("always"))
    assert cfg["presets"]["casc"]["cascade"]["verify_when"] == "always"

    cfg = normalize_moa_config(_cfg("bogus"))
    assert cfg["presets"]["casc"]["cascade"]["verify_when"] == "weak"


def test_run_verification_distrusts_verdict_after_crash():
    """A script that prints a verdict then exits non-zero is inconclusive —
    the crash may be the very computation the verdict depended on."""
    from hermes_cli.proxy.moa_cascade import run_verification

    assert (
        run_verification("print('VERDICT: CORRECT')\nraise RuntimeError('late crash')")
        == "inconclusive"
    )
    assert (
        run_verification("print('VERDICT: WRONG')\nimport sys; sys.exit(3)")
        == "inconclusive"
    )


def _write_clean_arbiter_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: cleanc
  presets:
    cleanc:
      mode: cascade
      cascade: {clean_arbiter: true}
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_clean(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_clean_arbiter_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def test_extract_candidate_final_line():
    """Addendum v1.3: extract_candidate's ANSWER regex alternation gains
    "FINAL" so RLM voters' ``FINAL: <answer>`` terminal line is recognized
    exactly like ``ANSWER: <answer>`` -- last match wins across BOTH forms."""
    from hermes_cli.proxy.moa_cascade import extract_candidate

    assert extract_candidate("FINAL: 42") == "42"
    assert extract_candidate("some steps\nFINAL: 42") == "42"
    assert extract_candidate("Reasoning...\nFINAL: 4") == "4"

    # Last match wins regardless of which keyword it used.
    assert extract_candidate("ANSWER: 1\nFINAL: 2") == "2"
    assert extract_candidate("FINAL: 1\nANSWER: 2") == "2"


def test_clean_slot_rlm_agent_and_rounds_normalization():
    """Addendum v1.3 config: `agent: "rlm"` (any other value ignored) plus
    optional `rlm_rounds` (int, default 6, clamp 2..12) on a reference slot,
    normalized alongside the existing `max_tokens` passthrough."""
    from hermes_cli.moa_config import _clean_slot

    # agent: "rlm" with no rlm_rounds -> default 6.
    slot = _clean_slot({"provider": "custom", "model": "gemma-4-31b", "agent": "rlm"})
    assert slot["agent"] == "rlm"
    assert slot["rlm_rounds"] == 6

    # rlm_rounds clamped to the 2..12 range.
    slot_hi = _clean_slot(
        {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm", "rlm_rounds": 99}
    )
    assert slot_hi["rlm_rounds"] == 12

    slot_lo = _clean_slot(
        {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm", "rlm_rounds": 0}
    )
    assert slot_lo["rlm_rounds"] == 2

    slot_mid = _clean_slot(
        {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm", "rlm_rounds": 4}
    )
    assert slot_mid["rlm_rounds"] == 4

    # Any other agent value is ignored -- neither key is saved.
    slot_other = _clean_slot(
        {"provider": "custom", "model": "gemma-4-31b", "agent": "something-else"}
    )
    assert "agent" not in slot_other
    assert "rlm_rounds" not in slot_other

    # No agent key at all -> unaffected, matching prior behavior.
    slot_plain = _clean_slot({"provider": "custom", "model": "gemma-4-31b"})
    assert "agent" not in slot_plain
    assert "rlm_rounds" not in slot_plain

    # agent: "rlm" alongside the existing max_tokens passthrough -- both
    # normalize independently onto the same cleaned slot.
    slot_both = _clean_slot(
        {
            "provider": "custom",
            "model": "gemma-4-31b",
            "agent": "rlm",
            "rlm_rounds": 8,
            "max_tokens": 4000,
        }
    )
    assert slot_both["agent"] == "rlm"
    assert slot_both["rlm_rounds"] == 8
    assert slot_both["max_tokens"] == 4000


@pytest.mark.asyncio
async def test_clean_arbiter_gets_no_voter_context(moa_home_clean, fake_llm):
    """With cascade.clean_arbiter, a disagreement arbiter re-solves from
    scratch — its messages carry NO voter outputs (anchoring guard)."""
    replies = {"voter-a": "ANSWER: 1", "voter-b": "ANSWER: 2"}

    def ref_handler(kwargs):
        return _response(replies[kwargs.get("model")])

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("ANSWER: 2")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:cleanc", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"]["tier"] == 1
        agg_calls = [c for c in fake_llm.calls if c.get("task") == "moa_aggregator"]
        assert len(agg_calls) == 1
        import json as _json
        joined = _json.dumps(agg_calls[0]["messages"])
        assert "ANSWER: 1" not in joined and "voter-a" not in joined
        assert "Mixture of Agents reference context" not in joined
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Addendum v1.4 -- real-world cascade: streaming + tool-carrying requests
# (iteration 37)
#
# Streaming-specific fakes below mirror tests/hermes_cli/test_moa_proxy_server
# .py's `_delta_chunk`/`_usage_chunk`/`_tool_call_delta`/`_read_sse` helpers
# verbatim (that file is the SSE convention oracle for this proxy). Tier-0
# voters remain NON-streaming even on a streaming HTTP request (the addendum
# runs the existing non-streaming voter fan-out inside a worker thread), so
# `fake_llm.handlers["moa_reference"]` fakes below return plain `_response(...)`
# objects exactly like the non-streaming cascade tests above -- only the
# aggregator fakes below become chunk iterators, and only for the tier-1
# "no escalate" case where the addendum specifies the aggregator streams
# live token-by-token.
# ---------------------------------------------------------------------------


def _delta_chunk(*, content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning,
        reasoning=None,
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage_chunk(prompt=10, completion=5):
    return SimpleNamespace(choices=[], usage=_usage(prompt, completion))


def _tool_call_delta(index=0, call_id="call_1", name=None, arguments=None):
    fn = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, type="function", function=fn)


async def _read_sse(resp):
    """Parse an SSE body into the list of decoded ``data:`` payloads."""
    import json as _json

    raw = await resp.text()
    events = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(_json.loads(payload))
    return events


def _write_rlm_cascade_cfg(home):
    """cascade preset with one plain voter and one RLM-agent voter (addendum
    v1.3), no escalate_to -- exists purely to prove an RLM voter's " [rlm]"
    label surfaces in the streaming tier-0 voters list (addendum v1.4)."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: cascrlm
  presets:
    cascrlm:
      mode: cascade
      reference_models:
        - provider: custom
          model: voter-a
        - provider: custom
          model: voter-b
          agent: rlm
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_rlm(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_rlm_cascade_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


# ---------------------------------------------------------------------------
# A. streaming + consensus -> tier 0, voter reasoning deltas, ONE content
#    delta with the winner text, no aggregator call.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_tier0_consensus_sse(moa_home, fake_llm):
    """Streaming, tool-free cascade request with voter consensus: the SSE
    stream carries the "[cascade: N voters answering...]" announcement, a
    "[voter <label>: done]" reasoning delta per voter, then the winner's
    FULL text as a single content delta -- never a token-by-token stream --
    finish_reason "stop", and the final usage chunk's usage.moa.cascade
    matches the non-streaming tier-0 shape. No moa_aggregator call at all."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response(
        "Reasoning...\nANSWER: 42"
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 6*7?"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = await _read_sse(resp)
        assert events[-1] == "[DONE]"
        chunks = [e for e in events if e != "[DONE]"]

        reasoning_text = "".join(
            c["choices"][0]["delta"].get("reasoning_content", "")
            for c in chunks
            if c["choices"]
        )
        assert "cascade" in reasoning_text.lower()
        assert "2 voters answering" in reasoning_text
        assert "[voter openrouter:voter-a: done]" in reasoning_text
        assert "[voter openrouter:voter-b: done]" in reasoning_text
        # No content leakage into the voter-progress reasoning deltas.
        assert "ANSWER: 42" not in reasoning_text

        content_chunks = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == (
            "Reasoning...\nANSWER: 42"
        )

        finish = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finish == ["stop"]

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        cascade = usage_chunks[0]["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["consensus"] == "42"
        assert cascade["votes"] == 2

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert "moa_aggregator" not in tasks
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# B. streaming + disagreement + no escalate_to (clean-arbiter preset) ->
#    tier 1 streams the aggregator LIVE, token-by-token.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_tier1_no_escalate_streams_aggregator_live(
    moa_home_clean, fake_llm
):
    """No consensus and no escalate_to configured (the clean-arbiter preset
    from the earlier addendum, reused here escalate-free): the aggregator
    streams live -- MULTIPLE content deltas arrive from the fake streaming
    handler, not one buffered chunk -- and usage.moa.cascade lands at tier 1."""
    replies = {"voter-a": "ANSWER: 1", "voter-b": "ANSWER: 2"}

    def ref_handler(kwargs):
        return _response(replies[kwargs.get("model")])

    def agg_stream_handler(kwargs):
        assert kwargs.get("stream") is True
        return iter(
            [
                _delta_chunk(reasoning="synthesizing... "),
                _delta_chunk(content="final "),
                _delta_chunk(content="answer"),
                _delta_chunk(finish_reason="stop"),
                _usage_chunk(prompt=40, completion=20),
            ]
        )

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = agg_stream_handler

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cleanc",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]

        content_chunks = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        # Live, token-by-token: MORE than one content delta (unlike tier 0's
        # single buffered winner-text delta).
        assert len(content_chunks) >= 2
        content_text = "".join(
            c["choices"][0]["delta"]["content"] for c in content_chunks
        )
        assert content_text == "final answer"

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        cascade = usage_chunks[0]["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# C. streaming + disagreement + escalate_to configured + discord -> tier 1
#    runs NON-streaming (must be inspected before the client sees it), and
#    the final tier-2 answer is emitted as ONE content delta.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_tier2_escalation_discord_one_content_delta(
    moa_home, fake_llm
):
    """escalate_to is configured (moa_home's casc/big preset pair): the
    tier-1 aggregator's candidate must be inspected BEFORE the client can see
    anything, so it runs non-streaming, same as the existing non-streaming
    tier-2 escalation path; when it agrees with NEITHER voter (discord), the
    escalate preset's aggregator (also non-streaming) produces the final
    text, which reaches the client as a SINGLE content delta -- not a live
    token stream."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("First path.\nANSWER: 1")
        return _response("Second path.\nANSWER: 2")

    def agg_handler(kwargs):
        if kwargs.get("model") == "big-model":
            return _response("Escalated final.\nANSWER: 3")
        return _response("Tier1 discord.\nANSWER: 5")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = agg_handler

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "what is 1, 2 or 5?"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]

        content_chunks = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == (
            "Escalated final.\nANSWER: 3"
        )

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        cascade = usage_chunks[0]["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 2

        big_calls = [c for c in fake_llm.calls if c.get("model") == "big-model"]
        assert len(big_calls) == 1
        mid_calls = [
            c
            for c in fake_llm.calls
            if c.get("task") == "moa_aggregator" and c.get("model") == "mid-model"
        ]
        assert len(mid_calls) == 1
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# D. tool-carrying cascade requests -> acting-model SOLO ("tool-solo"),
#    both non-streaming and streaming.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_tools_non_streaming_is_tool_solo(
    moa_home_tool_turns_solo, fake_llm
):
    """A cascade preset request carrying `tools`, with the explicit
    `tool_turns: solo` opt-out (addendum v1.5), never runs voters or attaches
    reference guidance -- the aggregator acts SOLO on the client's messages +
    tools, exactly like a plain (non-cascade) tool call, and the turn is
    surfaced as usage.moa.cascade == {"tier": None, "mode": "tool-solo"}.
    (The `tool_turns: "detect"` default's fresh-user-turn VOTER GATE behavior
    is covered separately below in the "Addendum v1.5" test section.)"""
    tool_call = SimpleNamespace(
        id="call_abc",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "HK"}'),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascsolo",
                "messages": [{"role": "user", "content": "weather in HK?"}],
                "tools": tools,
            },
        )
        assert resp.status == 200
        body = await resp.json()
        choice = body["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["tool_calls"] == [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "HK"}'},
            }
        ]

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_reference" not in tasks
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == tools

        assert body["usage"]["moa"]["cascade"] == {"tier": None, "mode": "tool-solo"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cascade_tools_streaming_is_tool_solo(moa_home_tool_turns_solo, fake_llm):
    """Streaming counterpart, with the explicit `tool_turns: solo` opt-out
    (addendum v1.5): zero moa_reference calls, the aggregator is called WITH
    the client's tools, a tool_calls delta passes straight through to the
    client, and the final usage chunk carries usage.moa.cascade ==
    {"tier": None, "mode": "tool-solo"}."""

    def agg_stream_handler(kwargs):
        assert kwargs.get("stream") is True
        assert kwargs.get("tools")
        return iter(
            [
                _delta_chunk(
                    tool_calls=[_tool_call_delta(0, "call_xyz", name="get_weather")]
                ),
                _delta_chunk(
                    tool_calls=[_tool_call_delta(0, None, arguments='{"city": "HK"}')]
                ),
                _delta_chunk(finish_reason="tool_calls"),
                _usage_chunk(),
            ]
        )

    fake_llm.handlers["moa_aggregator"] = agg_stream_handler
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascsolo",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": tools,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_reference" not in tasks

        tool_deltas = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
        ]
        assert tool_deltas[0]["id"] == "call_xyz"
        assert tool_deltas[0]["function"]["name"] == "get_weather"
        assert tool_deltas[1]["function"]["arguments"] == '{"city": "HK"}'

        finish = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finish == ["tool_calls"]

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        assert usage_chunks[0]["usage"]["moa"]["cascade"] == {
            "tier": None,
            "mode": "tool-solo",
        }
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# E. RLM voter active in streaming tier-0 -> the " [rlm]" label surfaces in
#    the cascade voters list (addendum v1.3's RLM loop reused inside the
#    addendum v1.4 streaming voter fan-out, unchanged).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_rlm_voter_label_in_cascade_voters(moa_home_rlm, fake_llm, monkeypatch):
    """voter-b runs the RLM reason->python->observe loop (a python-fence turn
    then FINAL); voter-a answers directly. Both land on the same answer, so
    tier 0 fires -- and the voters list in usage.moa.cascade carries voter-b's
    " [rlm]"-suffixed label, proving the RLM loop ran inside the streaming
    voter fan-out exactly as it does in the non-streaming cascade turn."""
    rlm_calls = {"n": 0}

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "voter-a":
            return _response("ANSWER: 42")
        rlm_calls["n"] += 1
        if rlm_calls["n"] == 1:
            return _response("Let's compute.\n```python\nprint(42)\n```")
        return _response("Verified.\nFINAL: 42")

    fake_llm.handlers["moa_reference"] = ref_handler
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_rlm_exec", lambda code: "42\n", raising=False
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascrlm",
                "messages": [{"role": "user", "content": "what is 6*7?"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        cascade = usage_chunks[0]["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert any(label.endswith("[rlm]") for label in cascade["voters"])
        assert "custom:voter-b [rlm]" in cascade["voters"]

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_aggregator" not in tasks
    finally:
        await client.close()


def _write_context_solo_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: ctx
  presets:
    ctx:
      mode: cascade
      cascade: {max_context_tokens: 1000}
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_ctx(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_context_solo_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.mark.asyncio
async def test_oversized_request_bypasses_voters_context_solo(moa_home_ctx, fake_llm):
    """A request bigger than cascade.max_context_tokens skips the voter pool
    and runs the acting aggregator solo (mode "context-solo")."""
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("long answer")
    big = "x" * 8000  # ~2000 tokens > the 1000-token cap
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:ctx", "messages": [{"role": "user", "content": big}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"] == {
            "tier": None,
            "mode": "context-solo",
        }
        assert not [c for c in fake_llm.calls if c.get("task") == "moa_reference"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_normal_request_under_context_cap_still_cascades(moa_home_ctx, fake_llm):
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 7")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:ctx", "messages": [{"role": "user", "content": "2+5? ANSWER format"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"]["tier"] == 0
    finally:
        await client.close()


def _write_rlm_arbiter_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: openladder
  presets:
    openladder:
      mode: cascade
      cascade: {clean_arbiter: true}
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: open-arbiter
        agent: rlm
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_rlm_arbiter(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_rlm_arbiter_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.mark.asyncio
async def test_clean_rlm_arbiter_runs_voter_loop(moa_home_rlm_arbiter, fake_llm, monkeypatch):
    """A clean arbiter slot with agent: rlm re-solves via the RLM loop —
    visible as moa_reference-task calls for the arbiter model and a FINAL
    answer, with no plain moa_aggregator call."""
    replies = {"voter-a": "ANSWER: 1", "voter-b": "ANSWER: 2"}

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model in replies:
            return _response(replies[model])
        return _response("Solved it.\nFINAL: 42")  # the RLM arbiter's turn

    fake_llm.handlers["moa_reference"] = ref_handler
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:openladder", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"]["tier"] == 1
        assert "FINAL: 42" in body["choices"][0]["message"]["content"]
        assert not [c for c in fake_llm.calls if c.get("task") == "moa_aggregator"]
        arbiter_calls = [
            c for c in fake_llm.calls
            if c.get("task") == "moa_reference" and c.get("model") == "open-arbiter"
        ]
        assert arbiter_calls
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Addendum v1.5 -- session-aware tool turns: cascade re-engagement
# (iteration 40)
#
# Agent clients resend `tools` on EVERY request, so v1.4's tool-solo bypass
# pins whole sessions to acting-solo forever. `cascade.tool_turns: "detect"`
# (the normalized default) decides per turn, by observation: a mid-loop turn
# (last message is a tool/assistant turn) stays acting-solo; a fresh user
# turn asks the tier-0 voters to vote either a real answer or the literal
# sentinel "ANSWER: TOOL_TURN" first, so the cascade can re-engage on
# reasoning/answer turns instead of paying acting-solo on every single turn
# of a tool-using session. `moa_home` (tool_turns unset -> normalizes to the
# "detect" default) is reused for every detect-mode test below; a dedicated
# `moa_home_tool_turns_solo` fixture covers the "solo" (v1.4-identical, no
# per-turn detection at all) config knob.
# ---------------------------------------------------------------------------


# Two named client tools, reused by every test below that needs a `tools`
# array -- the tool-awareness system line lists these function names.
_TOOLS_V15 = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    },
    {
        "type": "function",
        "function": {"name": "search_docs", "parameters": {"type": "object"}},
    },
]

# A mid-tool-loop transcript: the last non-system message is the "tool" role
# (a tool result just landed) -- the addendum's turn gate must recognize this
# as "mid-loop" without ever consulting the voters.
_MID_LOOP_MESSAGES = [
    {"role": "user", "content": "what's the weather in HK?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "HK"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_abc", "content": "31C, humid"},
]

# A fresh user turn -- last non-system message is "user" -- so the addendum's
# turn gate must run the tier-0 voter fan-out (the "VOTER GATE") instead of
# going straight to acting-solo.
_USER_TURN_MESSAGES = [{"role": "user", "content": "what should I do next?"}]


def _write_tool_turns_solo_cfg(home):
    """Same voter-pair/aggregator shape as `moa_home`, but with an explicit
    `cascade.tool_turns: solo` -- the addendum's opt-out that preserves the
    v1.4 tool-solo-on-every-turn behavior byte-identically, without ever
    running the "detect" turn gate."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: cascsolo
  presets:
    cascsolo:
      mode: cascade
      cascade:
        tool_turns: solo
      reference_models:
        - provider: openrouter
          model: voter-a
        - provider: openrouter
          model: voter-b
      aggregator:
        provider: openrouter
        model: mid-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home_tool_turns_solo(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_tool_turns_solo_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.mark.asyncio
async def test_tool_turns_detect_mid_loop_is_acting_solo(moa_home, fake_llm):
    """Last message role "tool" -> mid-loop: acting-solo with tools, with NO
    voter consultation at all (the addendum only votes on fresh user turns)."""
    tool_call = SimpleNamespace(
        id="call_xyz",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "HK"}'),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _MID_LOOP_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_reference" not in tasks
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == _TOOLS_V15

        assert body["usage"]["moa"]["cascade"] == {
            "tier": None,
            "mode": "tool-solo",
            "reason": "mid-loop",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_detect_vote_tool_turn_is_acting_solo(moa_home, fake_llm):
    """Fresh user turn, both voters vote the literal sentinel
    "ANSWER: TOOL_TURN" (they judge the request needs a tool they can't call)
    -> acting-solo with tools, reason "tool_turn_vote". The voters still RAN
    (and are billed) even though their vote never becomes client-visible
    text -- `usage.moa.references` must show both of them."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response(
        "This needs live data.\nANSWER: TOOL_TURN"
    )
    tool_call = SimpleNamespace(
        id="call_xyz",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments="{}"),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert tasks.count("moa_aggregator") == 1
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == _TOOLS_V15

        assert body["usage"]["moa"]["cascade"] == {
            "tier": None,
            "mode": "tool-solo",
            "reason": "tool_turn_vote",
            "votes": 2,
        }
        # Voters ran and are billed, even though their vote never surfaces as
        # client-visible content.
        assert len(body["usage"]["moa"]["references"]) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_detect_vote_real_answer_reverts_to_tier0(moa_home, fake_llm):
    """Fresh user turn, both voters agree on a REAL answer (not TOOL_TURN) ->
    the session has reverted to cascade: tier 0 returns the winner's text
    exactly as the tool-free path does, no aggregator call at all, and the
    client's tools are never forwarded to anything (voters never take a
    `tools` kwarg, and no acting call happens to forward them to)."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response(
        "Reasoning...\nANSWER: 42"
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        content = body["choices"][0]["message"]["content"]
        assert "ANSWER: 42" in content

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert "moa_aggregator" not in tasks
        assert all(not c.get("tools") for c in fake_llm.calls)

        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["consensus"] == "42"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_detect_no_consensus_is_acting_solo_no_guidance(
    moa_home, fake_llm
):
    """Fresh user turn, voters disagree (neither TOOL_TURN consensus nor a
    real-answer consensus) -> acting-solo WITH tools and NO reference
    guidance at all (not tier 1: the arbiter may need to call tools, which
    the tier-1 guidance-attach machinery does not forward)."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        return _response("ANSWER: 1" if model == "voter-a" else "ANSWER: 2")

    fake_llm.handlers["moa_reference"] = ref_handler
    tool_call = SimpleNamespace(
        id="call_xyz",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments="{}"),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert tasks.count("moa_aggregator") == 1
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == _TOOLS_V15
        # No tier-1 guidance attach: the acting call's messages are the
        # client's own turn, nothing else -- no voter text, no "Mixture of
        # Agents" guidance header.
        import json as _json

        joined = _json.dumps(agg_call["messages"])
        assert "ANSWER: 1" not in joined
        assert "ANSWER: 2" not in joined
        assert "Mixture of Agents reference context" not in joined

        assert body["usage"]["moa"]["cascade"] == {
            "tier": None,
            "mode": "tool-solo",
            "reason": "no-consensus",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_detect_voter_messages_carry_tool_awareness_line(
    moa_home, fake_llm
):
    """The voter gate appends ONE extra system line at the END of each
    voter's message list (voters get no advisory system prompt in direct
    mode otherwise) naming the client's tool functions and instructing the
    literal "ANSWER: TOOL_TURN" reply convention."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200

        ref_calls = [c for c in fake_llm.calls if c["task"] == "moa_reference"]
        assert len(ref_calls) == 2
        for call in ref_calls:
            msgs = call["messages"]
            # The client's own messages are untouched and come first; the
            # tool-awareness line is appended AFTER them, not prepended.
            assert msgs[: len(_USER_TURN_MESSAGES)] == _USER_TURN_MESSAGES
            last = msgs[-1]
            assert last["role"] == "system"
            assert "get_weather" in last["content"]
            assert "search_docs" in last["content"]
            assert "cannot call them" in last["content"]
            assert "ANSWER: TOOL_TURN" in last["content"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_solo_preserves_v14_behavior(
    moa_home_tool_turns_solo, fake_llm
):
    """`cascade.tool_turns: solo` is the v1.4 opt-out: EVERY tool-carrying
    request is acting-solo with no per-turn detection whatsoever -- even a
    fresh user-turn message (which "detect" would send to the voter gate)
    skips the voters entirely, and the usage surface stays byte-identical to
    the pre-addendum-v1.5 tool-solo shape (no "reason" key)."""
    tool_call = SimpleNamespace(
        id="call_abc",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "HK"}'),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascsolo",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        tasks = [c["task"] for c in fake_llm.calls]
        assert "moa_reference" not in tasks
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == _TOOLS_V15

        assert body["usage"]["moa"]["cascade"] == {"tier": None, "mode": "tool-solo"}
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_tool_turns_detect_streaming_reengagement_tier0(moa_home, fake_llm):
    """Streaming counterpart of the vote-real-answer case: a fresh user turn
    carrying tools still runs the live voter fan-out (progress reasoning
    deltas), the voters agree on a real answer, and the client sees the
    winner's FULL text as ONE content delta -- never a live token stream, and
    never an aggregator call -- with `usage.moa.cascade` landing at tier 0."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response(
        "Reasoning...\nANSWER: 42"
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": _USER_TURN_MESSAGES,
                "tools": _TOOLS_V15,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = await _read_sse(resp)
        assert events[-1] == "[DONE]"
        chunks = [e for e in events if e != "[DONE]"]

        content_chunks = [
            c
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("content")
        ]
        assert len(content_chunks) == 1
        assert content_chunks[0]["choices"][0]["delta"]["content"] == (
            "Reasoning...\nANSWER: 42"
        )

        finish = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finish == ["stop"]

        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        cascade = usage_chunks[0]["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 0
        assert cascade["consensus"] == "42"

        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert "moa_aggregator" not in tasks
    finally:
        await client.close()


def test_stacked_terminators_extract_clean_candidate():
    """RLM voters may emit 'FINAL: ANSWER: 4' when the client prompt also
    demands an ANSWER: line — the candidate must still be '4' or a mixed
    RLM/plain pool can never reach consensus (found live, 2026-07-06)."""
    from hermes_cli.proxy.moa_cascade import extract_candidate, normalize_candidate

    assert extract_candidate("blah\nFINAL: ANSWER: 4") == "4"
    assert extract_candidate("FINAL: answer: 13") == "13"
    assert extract_candidate("ANSWER: FINAL: 7") == "7"
    assert normalize_candidate(extract_candidate("done\nFINAL: ANSWER: 36")) == "36"


# ---------------------------------------------------------------------------
# Real-world serving: upstream cache visibility + prompt-cache decoration
# ---------------------------------------------------------------------------


def test_usage_to_openai_surfaces_cached_tokens():
    """usage always carries prompt_tokens_details.cached_tokens (OpenAI wire
    shape) so production sessions can OBSERVE provider-side prompt caching —
    a hard 0 on a warm turn is the signal a lane is not caching, so the key
    is emitted even when zero. cache_write_tokens only appears when real
    (Anthropic-style writes), keeping the plain-OpenAI shape untouched."""
    from agent.usage_pricing import CanonicalUsage

    warm = moa_server._usage_to_openai(
        CanonicalUsage(
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=500,
            cache_write_tokens=200,
        )
    )
    assert warm["prompt_tokens"] == 800  # input + cache reads + cache writes
    assert warm["prompt_tokens_details"] == {"cached_tokens": 500}
    assert warm["cache_write_tokens"] == 200

    cold = moa_server._usage_to_openai(
        CanonicalUsage(input_tokens=100, output_tokens=10)
    )
    assert cold["prompt_tokens_details"] == {"cached_tokens": 0}
    assert "cache_write_tokens" not in cold


@pytest.mark.asyncio
async def test_tool_solo_applies_cache_decoration(
    moa_home_tool_turns_solo, fake_llm, monkeypatch
):
    """The solo lane (the dominant path of a real agent session's tool loop)
    routes its outgoing messages through _maybe_apply_moa_cache_control,
    judged on the ACTING slot's own runtime — and call_llm receives exactly
    what the decorator returned."""
    seen: dict = {}

    def fake_decorate(messages, runtime):
        seen["runtime"] = runtime
        return [{"role": "system", "content": "DECORATED"}, *messages]

    monkeypatch.setattr(moa_server, "_maybe_apply_moa_cache_control", fake_decorate)
    tool_call = SimpleNamespace(
        id="call_abc",
        type="function",
        function=SimpleNamespace(name="get_weather", arguments='{"city": "HK"}'),
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response(
        None, tool_calls=[tool_call]
    )
    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:cascsolo",
                "messages": [{"role": "user", "content": "weather in HK?"}],
                "tools": tools,
            },
        )
        assert resp.status == 200
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["messages"][0] == {"role": "system", "content": "DECORATED"}
        assert seen["runtime"].get("provider") == "openrouter"
    finally:
        await client.close()
