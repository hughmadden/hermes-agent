"""Tests for the advisor lane (docs/plans/moa-cascade-spec.md addendum
v1.6 §B) — a concern-gated async reviewer for cascade sessions.

Conventions mirror tests/hermes_cli/test_moa_cascade.py exactly: fake
call_llm patched in moa_server, aiohttp TestServer/TestClient, HERMES_HOME
pointed at a tmp config.

Async determinism: the advisor is spawned via a fire-and-forget
``threading.Thread(...).start()`` at the end of the non-streaming cascade
turn (see ``moa_server._handle_non_streaming``). Tests that need the
advisor's effect to be observable on the NEXT request use the
``inline_thread`` fixture below, which monkeypatches ``threading.Thread``
(the same stdlib module object ``moa_server`` imports) so ``.start()``
calls ``target(*args, **kwargs)`` synchronously instead of on a background
thread — deterministic, and reverted automatically at test teardown.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

import hermes_cli.proxy.moa_server as moa_server
from hermes_cli.proxy.moa_server import _parse_advisor_reply, create_moa_app


# ---------------------------------------------------------------------------
# Fakes (verbatim conventions from test_moa_cascade.py)
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


def _write_advisor_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        advisor:
          provider: openrouter
          model: advisor-model
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
def moa_home_advisor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_advisor_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def _write_advisor_escalate_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        advisor:
          provider: openrouter
          model: advisor-model
        advisor_mode: escalate
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
def moa_home_advisor_escalate(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_advisor_escalate_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def _write_no_advisor_cfg(home):
    """Same voter-pair/aggregator shape, but no `cascade.advisor` slot at
    all — the config-normalization default, and the common case."""
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
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
def moa_home_no_advisor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_no_advisor_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.fixture()
def fake_llm(monkeypatch):
    """Patch call_llm in both moa_server (advisor/aggregator/tier-2 calls)
    and agent.moa_loop (the voter fan-out's own call_llm reference).
    Returns the recorded calls list."""
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


@pytest.fixture()
def inline_thread(monkeypatch):
    """Make the advisor's fire-and-forget spawn run SYNCHRONOUSLY.

    `moa_server` spawns the advisor via `threading.Thread(target=...,
    args=(common,), daemon=True).start()`. Real background execution would
    make its effect on a NEXT request racy/nondeterministic in a test.

    A NAIVE monkeypatch of `threading.Thread` itself breaks the rest of the
    request, though: `_handle_non_streaming` also runs `_run_turn` via
    `asyncio.to_thread` (and `_run_references_parallel` via its own
    `ThreadPoolExecutor`), both of which spin up REAL worker threads with
    `threading.Thread(...)` under the hood — replacing the class process-
    wide breaks those too (the executor's worker never loops, so `submit()`
    hangs). So this patches `threading.Thread` with a SELECTIVE stand-in: a
    call whose ``target`` is `moa_server._run_cascade_advisor` runs inline
    on the calling thread; every other call (executor workers, etc.) gets
    the real `threading.Thread` unchanged. monkeypatch reverts this
    automatically at teardown, so it never leaks into other test modules.
    """
    real_thread = moa_server.threading.Thread

    class _InlineAdvisorThread:
        def __init__(self, target=None, args=(), kwargs=None, daemon=None, **_extra):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

        def join(self, timeout=None):
            return None

    def _selective_thread(*args, target=None, **kwargs):
        if target is moa_server._run_cascade_advisor:
            return _InlineAdvisorThread(target=target, **kwargs)
        return real_thread(*args, target=target, **kwargs)

    monkeypatch.setattr(moa_server.threading, "Thread", _selective_thread)


async def _client(app):
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


# ---------------------------------------------------------------------------
# 1. config normalization: advisor slot + advisor_mode + defaults
# ---------------------------------------------------------------------------


def _advisor_cfg(advisor=None, advisor_mode=None):
    cascade: dict = {}
    if advisor is not None:
        cascade["advisor"] = advisor
    if advisor_mode is not None:
        cascade["advisor_mode"] = advisor_mode
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
            }
        },
    }


def test_advisor_config_normalization_table():
    from hermes_cli.moa_config import normalize_moa_config

    # No advisor block at all -> None slot, "notes" default mode.
    cfg = normalize_moa_config(_advisor_cfg())
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["advisor"] is None
    assert cascade["advisor_mode"] == "notes"

    # Explicit slot resolves via the same `_clean_slot` path as every other
    # cascade slot (judge/verifier/...).
    cfg = normalize_moa_config(
        _advisor_cfg(advisor={"provider": "custom:cerebras", "model": "gemma-4-31b"})
    )
    cascade = cfg["presets"]["casc"]["cascade"]
    assert cascade["advisor"] == {"provider": "custom:cerebras", "model": "gemma-4-31b"}
    assert cascade["advisor_mode"] == "notes"

    # A slot missing provider/model degrades to None, same tolerant-degrade
    # style as every other `_clean_slot` caller.
    cfg = normalize_moa_config(_advisor_cfg(advisor={"model": "gemma-4-31b"}))
    assert cfg["presets"]["casc"]["cascade"]["advisor"] is None

    # advisor_mode: explicit "escalate" preserved.
    cfg = normalize_moa_config(
        _advisor_cfg(
            advisor={"provider": "custom:cerebras", "model": "gemma-4-31b"},
            advisor_mode="escalate",
        )
    )
    assert cfg["presets"]["casc"]["cascade"]["advisor_mode"] == "escalate"

    # advisor_mode: unrecognized value degrades to "notes".
    cfg = normalize_moa_config(
        _advisor_cfg(
            advisor={"provider": "custom:cerebras", "model": "gemma-4-31b"},
            advisor_mode="bogus",
        )
    )
    assert cfg["presets"]["casc"]["cascade"]["advisor_mode"] == "notes"

    # An MoA slot is rejected (recursive-MoA guard) just like every other
    # cascade slot -- degrades to None.
    cfg = normalize_moa_config(_advisor_cfg(advisor={"provider": "moa", "model": "x"}))
    assert cfg["presets"]["casc"]["cascade"]["advisor"] is None


# ---------------------------------------------------------------------------
# 2. charter parse table: OK / CONCERN / BLOCKER / garbage
# ---------------------------------------------------------------------------


def test_parse_advisor_reply_table():
    # "OK" (with or without trailing punctuation/whitespace) -> None.
    assert _parse_advisor_reply("OK") is None
    assert _parse_advisor_reply("  OK  ") is None
    assert _parse_advisor_reply("OK.") is None
    assert _parse_advisor_reply("OK, nothing to add") is None

    # CONCERN -> a concern note, text after the colon, stripped.
    note = _parse_advisor_reply("CONCERN: check for a race condition")
    assert note == {"kind": "concern", "text": "check for a race condition"}

    # BLOCKER -> a blocker note.
    note = _parse_advisor_reply("BLOCKER: about to rm -rf without confirmation")
    assert note == {"kind": "blocker", "text": "about to rm -rf without confirmation"}

    # Note text capped at 300 chars.
    long_text = "x" * 400
    note = _parse_advisor_reply(f"CONCERN: {long_text}")
    assert note["text"] == long_text[:300]
    assert len(note["text"]) == 300

    # Empty body after the prefix -> None (nothing worth injecting).
    assert _parse_advisor_reply("CONCERN:") is None
    assert _parse_advisor_reply("CONCERN:   ") is None

    # Garbage: empty, wrong format, no leading OK/CONCERN/BLOCKER, an essay
    # a model wrote despite the charter -- all fail-open to None.
    assert _parse_advisor_reply("") is None
    assert _parse_advisor_reply("   ") is None
    assert _parse_advisor_reply("Sure, here is my review of the conversation...") is None
    assert _parse_advisor_reply("Looks fine to me") is None
    assert _parse_advisor_reply(None) is None


# ---------------------------------------------------------------------------
# 3. note storage + one-shot injection, end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisor_note_stored_and_injected_exactly_once(
    moa_home_advisor, fake_llm, inline_thread
):
    """Turn 1's advisor run stores a CONCERN note (voters disagree both
    turns, so every turn reaches tier 1 and calls the aggregator — the
    acting call site the note is injected into). Turn 2's aggregator call
    carries the bracketed note; turn 2's OWN advisor run replies "OK", so
    turn 3's aggregator call carries no note at all."""
    advisor_replies = iter(["CONCERN: check for a race condition", "OK", "OK"])

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "advisor-model":
            return _response(next(advisor_replies, "OK"))
        if model == "voter-a":
            return _response("ANSWER: 1")
        return _response("ANSWER: 2")  # voter-b

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("aggregator acted")

    client = await _client(create_moa_app())
    headers = {"x-hermes-session-id": "advisor-note-flow"}
    try:
        # Turn 1: fresh session, no note yet.
        resp1 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "turn one"}]},
            headers=headers,
        )
        assert resp1.status == 200
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        assert "[advisor note" not in json.dumps(agg_calls[0]["messages"])
        body1 = await resp1.json()
        assert body1["usage"]["moa"]["session"]["advisor_note"] is None

        advisor_calls = [c for c in fake_llm.calls if c.get("model") == "advisor-model"]
        assert len(advisor_calls) == 1  # turn 1's advisor ran once, async

        # Turn 2: the CONCERN note from turn 1 is injected as the tail
        # system message on the aggregator's call.
        resp2 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "turn two"}]},
            headers=headers,
        )
        assert resp2.status == 200
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 2
        turn2_messages = agg_calls[1]["messages"]
        assert turn2_messages[-1] == {
            "role": "system",
            "content": "[advisor note (concern): check for a race condition]",
        }
        body2 = await resp2.json()
        assert body2["usage"]["moa"]["session"]["advisor_note"] == {
            "kind": "concern",
            "text": "check for a race condition",
        }

        # Turn 3: fired exactly once -- gone now, and turn 2's own advisor
        # run replied "OK" so nothing new was queued either.
        resp3 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "turn three"}]},
            headers=headers,
        )
        assert resp3.status == 200
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 3
        assert "[advisor note" not in json.dumps(agg_calls[2]["messages"])
        body3 = await resp3.json()
        assert body3["usage"]["moa"]["session"]["advisor_note"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_advisor_note_injected_on_solo_turn(moa_home_advisor, fake_llm, inline_thread):
    """The other designated injection site: `_run_cascade_solo_turn`. A
    mid-tool-loop request (last non-system message has role "tool") hits
    the addendum v1.5 "mid-loop" bypass straight to `_run_cascade_solo_turn`
    with no voter fan-out at all -- exactly the acting-solo call the
    injection must reach. Seed a pending note directly on the registry,
    then confirm the acting call carries the bracketed note at the tail."""
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("solo acted")

    moa_server._session_registry.resolve("advisor-solo-flow", [{"role": "user", "content": "x"}])
    moa_server._session_registry.note_advisor(
        "advisor-solo-flow", {"kind": "concern", "text": "watch the tool budget"}
    )

    mid_loop_messages = [
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

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": mid_loop_messages,
                "tools": [
                    {"type": "function", "function": {"name": "get_weather", "parameters": {}}}
                ],
            },
            headers={"x-hermes-session-id": "advisor-solo-flow"},
        )
        assert resp.status == 200
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        assert agg_calls[0]["messages"][-1] == {
            "role": "system",
            "content": "[advisor note (concern): watch the tool budget]",
        }
        # No VOTER fan-out at all -- the mid-loop bypass never consults
        # voters. The one "moa_reference" call that does show up is the
        # advisor's OWN post-turn review (same task name voters use), run
        # inline here by `inline_thread`; no voter-labelled model appears.
        ref_calls = [c for c in fake_llm.calls if c["task"] == "moa_reference"]
        assert [c.get("model") for c in ref_calls] == ["advisor-model"]
        cascade_usage = (await resp.json())["usage"]["moa"]["cascade"]
        assert cascade_usage == {"tier": None, "mode": "tool-solo", "reason": "mid-loop"}
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 4. advisor failure never breaks a turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_advisor_upstream_failure_never_breaks_the_turn(
    moa_home_advisor, fake_llm, inline_thread
):
    """The advisor slot's own call raising must not affect the turn that
    triggered it -- `_run_cascade_advisor` wraps its whole body in one
    try/except, and the spawn happens after the response is already
    assembled."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "advisor-model":
            raise RuntimeError("advisor upstream exploded")
        if model == "voter-a":
            return _response("ANSWER: 1")
        return _response("ANSWER: 2")  # voter-b, disagreement -> tier 1

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("aggregator acted")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "q"}]},
            headers={"x-hermes-session-id": "advisor-fail-turn"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["choices"][0]["message"]["content"] == "aggregator acted"
        assert body["usage"]["moa"]["session"]["advisor_note"] is None
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_advisor_malformed_reply_never_stores_a_note(
    moa_home_advisor, fake_llm, inline_thread
):
    """A reply that violates the charter (no OK/CONCERN/BLOCKER prefix)
    fails open -- `_parse_advisor_reply` returns None, so nothing is
    stored, and the next turn is unaffected."""

    def ref_handler(kwargs):
        model = kwargs.get("model")
        if model == "advisor-model":
            return _response("I have thought about it and everything seems fine!")
        if model == "voter-a":
            return _response("ANSWER: 1")
        return _response("ANSWER: 2")

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("aggregator acted")

    client = await _client(create_moa_app())
    headers = {"x-hermes-session-id": "advisor-malformed-flow"}
    try:
        resp1 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "turn one"}]},
            headers=headers,
        )
        assert resp1.status == 200

        resp2 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "turn two"}]},
            headers=headers,
        )
        assert resp2.status == 200
        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert "[advisor note" not in json.dumps(agg_calls[-1]["messages"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_no_advisor_slot_never_spawns_a_thread(moa_home_no_advisor, fake_llm, monkeypatch):
    """No `cascade.advisor` slot configured (the default) -- the
    fire-and-forget spawn must not even construct a `threading.Thread` for
    the advisor. Not using `inline_thread` here on purpose: a SPY on
    `threading.Thread` records every ``target=_run_cascade_advisor`` call
    while delegating construction to the real class always (the request
    still needs REAL worker threads for `asyncio.to_thread`/the voter
    fan-out's `ThreadPoolExecutor` -- see `inline_thread`'s docstring for
    why a naive patch would hang those instead)."""
    real_thread = moa_server.threading.Thread
    advisor_thread_calls: list[object] = []

    def _spy_thread(*args, target=None, **kwargs):
        if target is moa_server._run_cascade_advisor:
            advisor_thread_calls.append(target)
        return real_thread(*args, target=target, **kwargs)

    monkeypatch.setattr(moa_server.threading, "Thread", _spy_thread)
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "what is 6*7?"}]},
        )
        assert resp.status == 200
        assert advisor_thread_calls == []
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# 5. "escalate" mode: a pending BLOCKER forces tier 1 despite consensus
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalate_mode_blocker_forces_tier1_despite_consensus(
    moa_home_advisor_escalate, fake_llm, inline_thread
):
    """Addendum v1.6 §B's escalate hook: a pending BLOCKER note (seeded
    directly on the registry, isolating the gating behavior from the
    advisor call itself) forces the NEXT turn past the tier-0 exact-
    consensus shortcut into tier 1, even though the voters agree -- and the
    injected note still lands on the tier-1 aggregator's messages."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("ANSWER: 42 (double-checked)")

    moa_server._session_registry.resolve(
        "advisor-escalate-flow", [{"role": "user", "content": "seed"}]
    )
    moa_server._session_registry.note_advisor(
        "advisor-escalate-flow", {"kind": "blocker", "text": "prior turn nearly deleted prod"}
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "q"}]},
            headers={"x-hermes-session-id": "advisor-escalate-flow"},
        )
        assert resp.status == 200
        body = await resp.json()
        cascade = body["usage"]["moa"]["cascade"]
        assert cascade["tier"] == 1
        assert cascade["advisor_escalated"] is True
        assert cascade["tier0_consensus"] == "42"

        agg_calls = [c for c in fake_llm.calls if c["task"] == "moa_aggregator"]
        assert len(agg_calls) == 1
        assert agg_calls[0]["messages"][-1] == {
            "role": "system",
            "content": "[advisor note (blocker): prior turn nearly deleted prod]",
        }
        assert body["usage"]["moa"]["session"]["advisor_note"] == {
            "kind": "blocker",
            "text": "prior turn nearly deleted prod",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_notes_mode_never_forces_tier1_despite_blocker(
    moa_home_advisor, fake_llm, inline_thread
):
    """Control for the escalate test above: the default "notes" mode never
    changes gating, even with a pending BLOCKER note -- consensus still
    wins tier 0 outright (no aggregator call at all)."""
    fake_llm.handlers["moa_reference"] = lambda kwargs: _response("ANSWER: 42")

    moa_server._session_registry.resolve(
        "advisor-notes-mode-flow", [{"role": "user", "content": "seed"}]
    )
    moa_server._session_registry.note_advisor(
        "advisor-notes-mode-flow", {"kind": "blocker", "text": "should not force anything"}
    )

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "q"}]},
            headers={"x-hermes-session-id": "advisor-notes-mode-flow"},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["usage"]["moa"]["cascade"]["tier"] == 0
        assert not any(c["task"] == "moa_aggregator" for c in fake_llm.calls)
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Addendum v1.7 — OMP-faithful inline advisor mode
# ---------------------------------------------------------------------------


def _write_inline_advisor_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        advisor:
          provider: openrouter
          model: advisor-model
        advisor_mode: inline
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
def moa_home_inline_advisor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_inline_advisor_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


def test_inline_advisor_mode_normalizes():
    from hermes_cli.moa_config import normalize_moa_config

    cfg = normalize_moa_config(
        {
            "presets": {
                "p": {
                    "mode": "cascade",
                    "reference_models": [
                        {"provider": "openrouter", "model": "a"},
                        {"provider": "openrouter", "model": "b"},
                    ],
                    "aggregator": {"provider": "openrouter", "model": "m"},
                    "cascade": {
                        "advisor": {"provider": "openrouter", "model": "adv"},
                        "advisor_mode": "inline",
                    },
                }
            }
        }
    )
    assert cfg["presets"]["p"]["cascade"]["advisor_mode"] == "inline"


@pytest.mark.asyncio
async def test_inline_advisor_injects_same_turn_synchronously(
    moa_home_inline_advisor, fake_llm
):
    """OMP-faithful: the advisor runs ON this turn (no thread, no prior turn)
    and its note is injected into the acting call's messages the SAME turn.
    No async spawn is needed, so this test does NOT use the inline_thread
    fixture — if inline mode wrongly relied on the background worker, the
    note would be absent here."""
    # Voters disagree -> tier-1 aggregator acts (an acting call exists to
    # inject into).
    answers = iter(["ANSWER: 1", "ANSWER: 2"])
    fake_llm.handlers["moa_reference"] = lambda kwargs: (
        _response("advice: check the edge case")
        if kwargs.get("model") == "advisor-model"
        else _response(next(answers))
    )
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("acted")

    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        # advisor was called on THIS turn
        advisor_calls = [c for c in fake_llm.calls if c.get("model") == "advisor-model"]
        assert len(advisor_calls) == 1
        # and its note reached the acting aggregator's messages this turn
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        sys_texts = " ".join(
            str(m.get("content")) for m in agg_call["messages"] if m.get("role") == "system"
        )
        assert "advisor:" in sys_texts and "edge case" in sys_texts
        assert body["usage"]["moa"]["session"]["advisor_note"]["kind"] == "inline"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_inline_advisor_failure_never_breaks_turn(moa_home_inline_advisor, fake_llm):
    """A failing inline advisor degrades to a normal turn (fail-open),
    injecting nothing rather than erroring."""
    def ref_handler(kwargs):
        if kwargs.get("model") == "advisor-model":
            raise RuntimeError("advisor upstream down")
        return _response("ANSWER: 5")

    fake_llm.handlers["moa_reference"] = ref_handler
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "2+3?"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        # unanimous voters -> tier-0, turn succeeds regardless of advisor
        assert "5" in (body["choices"][0]["message"]["content"] or "")
    finally:
        await client.close()


def _write_rlm_advisor_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: casc
  presets:
    casc:
      mode: cascade
      cascade:
        advisor:
          provider: openrouter
          model: advisor-model
          agent: rlm
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
def moa_home_rlm_advisor(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_rlm_advisor_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    return home


@pytest.mark.asyncio
async def test_rlm_advisor_routes_through_loop_and_strips_final(
    moa_home_rlm_advisor, fake_llm, inline_thread
):
    """An advisor slot with agent: rlm runs the reason->observe loop; the
    loop's 'FINAL: CONCERN: ...' terminator is stripped so the stored note is
    a normal concern (kind=concern), injected next turn."""
    # The RLM loop calls task=moa_reference repeatedly; make the advisor model
    # finish immediately with a FINAL line, voters disagree so tier-1 acts.
    voter_answers = iter(["ANSWER: 1", "ANSWER: 2", "ANSWER: 1", "ANSWER: 2"])

    def ref_handler(kwargs):
        if kwargs.get("model") == "advisor-model":
            return _response("FINAL: CONCERN: the fix misses a None case")
        return _response(next(voter_answers))

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("acted")

    client = await _client(create_moa_app())
    try:
        # turn 1 — advisor runs (async, inline_thread makes it synchronous)
        r1 = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:casc", "messages": [{"role": "user", "content": "fix it"}]},
        )
        assert r1.status == 200
        advisor_calls = [c for c in fake_llm.calls if c.get("model") == "advisor-model"]
        assert advisor_calls, "RLM advisor never called"
        # turn 2 — the concern (FINAL: stripped) is injected
        r2 = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:casc",
                "messages": [
                    {"role": "user", "content": "fix it"},
                    {"role": "assistant", "content": "done"},
                    {"role": "user", "content": "next"},
                ],
            },
        )
        b2 = await r2.json()
        note = b2["usage"]["moa"]["session"].get("advisor_note")
        assert note and note["kind"] == "concern"
        assert "None case" in note["text"] and "FINAL" not in note["text"]
    finally:
        await client.close()
