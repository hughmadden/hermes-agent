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
# 7. request WITH tools on the cascade preset -> existing fanout path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cascade_with_tools_uses_fanout(moa_home, fake_llm):
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
                "model": "moa:casc",
                "messages": [{"role": "user", "content": "look something up"}],
                "tools": tools,
            },
        )
        assert resp.status == 200
        body = await resp.json()

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        assert agg_calls[0]["tools"] == tools

        assert "cascade" not in body["usage"]["moa"]
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
