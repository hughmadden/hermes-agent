"""Tests for the OpenAI-compatible MoA endpoint (`hermes moa serve`).

Everything here fakes call_llm — both the moa_server module's own reference
streaming/aggregator calls and agent.moa_loop's non-streaming reference
fan-out — so no network or provider credentials are involved. Live end-to-end
coverage (real OpenRouter models) lives in
tests/integration/test_moa_proxy_live.py behind the ``integration`` marker.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

import hermes_cli.proxy.moa_server as moa_server
from hermes_cli.proxy.moa_server import create_moa_app, resolve_preset_name


# ---------------------------------------------------------------------------
# Fakes
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


def _write_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: ref-model-a
        - provider: openrouter
          model: ref-model-b
      aggregator:
        provider: openrouter
        model: agg-model
    solo:
      reference_models:
        - provider: openrouter
          model: ref-model-a
      aggregator:
        provider: openrouter
        model: agg-model
    hidden:
      enabled: false
      reference_models:
        - provider: openrouter
          model: ref-model-a
      aggregator:
        provider: openrouter
        model: agg-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def moa_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    _write_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    # The server-wide reference cache persists across tests; isolate each one.
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


async def _read_sse(resp):
    """Parse an SSE body into the list of decoded ``data:`` payloads."""
    raw = await resp.text()
    events = []
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


# ---------------------------------------------------------------------------
# resolve_preset_name
# ---------------------------------------------------------------------------


def test_resolve_preset_name_variants(moa_home):
    from hermes_cli.config import load_config

    config = load_config()
    assert resolve_preset_name("moa:review", config) == "review"
    assert resolve_preset_name("moa/solo", config) == "solo"
    assert resolve_preset_name("review", config) == "review"
    assert resolve_preset_name("", config) == "review"
    assert resolve_preset_name(None, config) == "review"
    assert resolve_preset_name("moa", config) == "review"
    assert resolve_preset_name("default", config) == "review"
    with pytest.raises(KeyError):
        resolve_preset_name("nope", config)


# ---------------------------------------------------------------------------
# /health, /v1/models, auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_open_even_with_api_key(moa_home):
    client = await _client(create_moa_app(api_key="sekrit"))
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        assert (await resp.json())["status"] == "ok"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_models_lists_enabled_presets(moa_home):
    client = await _client(create_moa_app())
    try:
        resp = await client.get("/v1/models")
        assert resp.status == 200
        payload = await resp.json()
        ids = {m["id"] for m in payload["data"]}
        # load_config deep-merges DEFAULT_CONFIG, which always contributes a
        # built-in 'default' preset alongside the user's presets.
        assert {"moa:review", "moa:solo"} <= ids
        assert "moa:hidden" not in ids  # disabled presets are not advertised
        default = next(m for m in payload["data"] if m["id"] == "moa:review")
        assert default["moa"]["default"] is True
        assert default["moa"]["references"] == [
            "openrouter:ref-model-a",
            "openrouter:ref-model-b",
        ]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auth_required_when_api_key_set(moa_home):
    client = await _client(create_moa_app(api_key="sekrit"))
    try:
        resp = await client.get("/v1/models")
        assert resp.status == 401
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer wrong"})
        assert resp.status == 401
        resp = await client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
        assert resp.status == 200
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — request validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_rejects_bad_json(moa_home):
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            data=b"{nope",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_rejects_missing_messages(moa_home):
    client = await _client(create_moa_app())
    try:
        resp = await client.post("/v1/chat/completions", json={"model": "moa:review"})
        assert resp.status == 400
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_unknown_preset_is_404(moa_home, fake_llm):
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:nope", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 404
        body = await resp.json()
        assert body["error"]["code"] == "model_not_found"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — non-streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_streaming_completion(moa_home, fake_llm):
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:review",
                "messages": [{"role": "user", "content": "what should I do?"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "moa:review"
        choice = body["choices"][0]
        assert choice["message"]["content"] == "aggregator acted"
        assert choice["finish_reason"] == "stop"

        # Two references + one aggregator were called.
        tasks = [c["task"] for c in fake_llm.calls]
        assert tasks.count("moa_reference") == 2
        assert tasks.count("moa_aggregator") == 1

        # The aggregator saw the client transcript plus the injected guidance.
        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        agg_text = agg_call["messages"][-1]["content"]
        assert "what should I do?" in agg_text
        assert "[Mixture of Agents reference context]" in agg_text
        assert "reference advice" in agg_text

        # Usage sums references + aggregator (3 fake calls x 10/5 tokens).
        assert body["usage"]["prompt_tokens"] == 30
        assert body["usage"]["completion_tokens"] == 15
        assert body["usage"]["total_tokens"] == 45
        assert len(body["usage"]["moa"]["references"]) == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_non_streaming_forwards_tools_and_returns_tool_calls(moa_home, fake_llm):
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
                "model": "moa:solo",
                "messages": [{"role": "user", "content": "weather in HK?"}],
                "tools": tools,
                "tool_choice": "auto",
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

        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        assert agg_call["tools"] == tools
        assert agg_call["extra_body"]["tool_choice"] == "auto"
        # References never see the client's tools — they are advisory-only.
        for ref_call in (c for c in fake_llm.calls if c["task"] == "moa_reference"):
            assert not ref_call.get("tools")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_non_streaming_tool_result_round_trip(moa_home, fake_llm):
    """Second leg of a client-side tool loop: assistant tool_calls + tool
    result in the transcript must reach the aggregator verbatim and be
    flattened (not dropped) in the reference advisory view."""
    messages = [
        {"role": "user", "content": "weather in HK?"},
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
            json={"model": "moa:solo", "messages": messages},
        )
        assert resp.status == 200

        agg_call = next(c for c in fake_llm.calls if c["task"] == "moa_aggregator")
        roles = [m["role"] for m in agg_call["messages"]]
        assert roles[:3] == ["user", "assistant", "tool"]

        ref_call = next(c for c in fake_llm.calls if c["task"] == "moa_reference")
        ref_text = "\n".join(str(m.get("content")) for m in ref_call["messages"])
        assert "get_weather" in ref_text
        assert "31C, humid" in ref_text
        # Advisory view flattens tools: no tool-role messages, no tool_calls.
        assert all(m.get("role") != "tool" for m in ref_call["messages"])
        assert all(not m.get("tool_calls") for m in ref_call["messages"])
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_reference_cache_hit_skips_fanout(moa_home, fake_llm):
    client = await _client(create_moa_app())
    try:
        payload = {
            "model": "moa:review",
            "messages": [{"role": "user", "content": "same state"}],
        }
        await client.post("/v1/chat/completions", json=payload)
        await client.post("/v1/chat/completions", json=payload)
        tasks = [c["task"] for c in fake_llm.calls]
        # References ran once (2 slots); the aggregator ran per request.
        assert tasks.count("moa_reference") == 2
        assert tasks.count("moa_aggregator") == 2
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions — streaming
# ---------------------------------------------------------------------------


def _install_streaming_fakes(fake_llm, *, agg_events=None):
    """References stream two deltas each; aggregator streams reasoning,
    content, a finish_reason, and a usage chunk (or the given events)."""

    def ref_handler(kwargs):
        assert kwargs.get("stream") is True
        return iter(
            [
                _delta_chunk(reasoning="hmm "),
                _delta_chunk(content=f"advice from {kwargs['model']}"),
                _usage_chunk(),
            ]
        )

    def agg_handler(kwargs):
        assert kwargs.get("stream") is True
        return iter(
            agg_events
            if agg_events is not None
            else [
                _delta_chunk(reasoning="let me merge... "),
                _delta_chunk(content="final "),
                _delta_chunk(content="answer"),
                _delta_chunk(finish_reason="stop"),
                _usage_chunk(prompt=40, completion=20),
            ]
        )

    fake_llm.handlers["moa_reference"] = ref_handler
    fake_llm.handlers["moa_aggregator"] = agg_handler


@pytest.mark.asyncio
async def test_streaming_reasoning_then_content(moa_home, fake_llm):
    _install_streaming_fakes(fake_llm)
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:review",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        assert resp.status == 200
        assert resp.headers["Content-Type"].startswith("text/event-stream")
        events = await _read_sse(resp)
        assert events[-1] == "[DONE]"
        chunks = [e for e in events if e != "[DONE]"]
        assert all(c["object"] == "chat.completion.chunk" for c in chunks)
        assert all(c["model"] == "moa:review" for c in chunks)

        reasoning_text = "".join(
            c["choices"][0]["delta"].get("reasoning_content", "")
            for c in chunks
            if c["choices"]
        )
        content_text = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c["choices"]
        )
        # Reference thinking + advice + labels stream as reasoning...
        assert "[Reference 1/2 — openrouter:ref-model-a]" in reasoning_text
        assert "[Reference 2/2 — openrouter:ref-model-b]" in reasoning_text
        assert "advice from ref-model-a" in reasoning_text
        assert "[Aggregating — openrouter:agg-model" in reasoning_text
        assert "let me merge... " in reasoning_text
        # ...and the aggregator's acting output is plain content.
        assert content_text == "final answer"
        # Reference 1's block streams before reference 2's.
        assert reasoning_text.index("advice from ref-model-a") < reasoning_text.index(
            "[Reference 2/2"
        )

        # First delta chunk carries the assistant role.
        first_delta = next(c for c in chunks if c["choices"])
        assert first_delta["choices"][0]["delta"].get("role") == "assistant"

        # finish chunk then usage chunk (include_usage was requested).
        finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
        assert finish and finish[-1]["choices"][0]["finish_reason"] == "stop"
        usage_chunks = [c for c in chunks if not c["choices"]]
        assert len(usage_chunks) == 1
        usage = usage_chunks[0]["usage"]
        # 2 refs (10/5 each) + aggregator (40/20).
        assert usage["prompt_tokens"] == 60
        assert usage["completion_tokens"] == 30
        assert usage["moa"]["aggregator"]["prompt_tokens"] == 40
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_streaming_tool_call_deltas_pass_through(moa_home, fake_llm):
    _install_streaming_fakes(
        fake_llm,
        agg_events=[
            _delta_chunk(
                tool_calls=[_tool_call_delta(0, "call_xyz", name="get_weather")]
            ),
            _delta_chunk(
                tool_calls=[_tool_call_delta(0, None, arguments='{"city": "HK"}')]
            ),
            _delta_chunk(finish_reason="tool_calls"),
        ],
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:solo",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "stream": True,
            },
        )
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]
        tool_deltas = [
            c["choices"][0]["delta"]["tool_calls"][0]
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("tool_calls")
        ]
        assert tool_deltas[0]["id"] == "call_xyz"
        assert tool_deltas[0]["function"]["name"] == "get_weather"
        assert tool_deltas[1]["function"]["arguments"] == '{"city": "HK"}'
        assert all(d["index"] == 0 for d in tool_deltas)
        finish = [
            c["choices"][0]["finish_reason"]
            for c in chunks
            if c["choices"] and c["choices"][0]["finish_reason"]
        ]
        assert finish == ["tool_calls"]
        # No usage chunk without include_usage.
        assert all(c["choices"] for c in chunks)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_streaming_cached_references_replay_as_reasoning(moa_home, fake_llm):
    _install_streaming_fakes(fake_llm)
    client = await _client(create_moa_app())
    try:
        payload = {
            "model": "moa:review",
            "messages": [{"role": "user", "content": "same state"}],
            "stream": True,
        }
        await (await client.post("/v1/chat/completions", json=payload)).text()
        ref_calls_before = sum(
            1 for c in fake_llm.calls if c["task"] == "moa_reference"
        )
        resp = await client.post("/v1/chat/completions", json=payload)
        events = await _read_sse(resp)
        ref_calls_after = sum(1 for c in fake_llm.calls if c["task"] == "moa_reference")
        assert ref_calls_after == ref_calls_before  # cache hit — no re-run
        reasoning_text = "".join(
            e["choices"][0]["delta"].get("reasoning_content", "")
            for e in events
            if e != "[DONE]" and e["choices"]
        )
        assert "(cached)" in reasoning_text
        assert "advice from ref-model-a" in reasoning_text
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_streaming_reference_failure_still_aggregates(moa_home, fake_llm):
    def failing_ref(kwargs):
        raise RuntimeError("provider down")

    fake_llm.handlers["moa_reference"] = failing_ref
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: iter(
        [_delta_chunk(content="acted anyway"), _delta_chunk(finish_reason="stop")]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:solo",
                "messages": [{"role": "user", "content": "q"}],
                "stream": True,
            },
        )
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]
        reasoning_text = "".join(
            c["choices"][0]["delta"].get("reasoning_content", "")
            for c in chunks
            if c["choices"]
        )
        content_text = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks
            if c["choices"]
        )
        assert "[failed: provider down]" in reasoning_text
        assert content_text == "acted anyway"
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# moa:auto routing through the endpoint
# ---------------------------------------------------------------------------


def _write_routed_cfg(home):
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: general
  router:
    enabled: true
    classifier:
      provider: openrouter
      model: fast-classifier
    default: general
  presets:
    coding:
      route:
        description: code writing, debugging, refactors, shell
      reference_models:
        - provider: openrouter
          model: ref-model-a
      aggregator:
        provider: openrouter
        model: agg-model
    general:
      route:
        description: everything else
      reference_models:
        - provider: openrouter
          model: ref-model-b
      aggregator:
        provider: openrouter
        model: agg-model
""".strip(),
        encoding="utf-8",
    )


@pytest.fixture()
def routed_home(monkeypatch, tmp_path):
    import hermes_cli.proxy.moa_router as moa_router

    home = tmp_path / ".hermes"
    _write_routed_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_server._ref_cache.clear()
    moa_router.sticky_clear()
    return home


@pytest.mark.asyncio
async def test_models_lists_auto_when_router_enabled(routed_home):
    client = await _client(create_moa_app())
    try:
        resp = await client.get("/v1/models")
        data = (await resp.json())["data"]
        ids = [m["id"] for m in data]
        assert "moa:auto" in ids
        auto = next(m for m in data if m["id"] == "moa:auto")
        assert auto["moa"]["router"] is True
        assert auto["moa"]["classifier"] == "openrouter:fast-classifier"
        assert auto["moa"]["routable_presets"] == ["coding", "general"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_404_when_router_disabled(moa_home, fake_llm):
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:auto", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 404
        body = await resp.json()
        assert "router" in body["error"]["message"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_routes_to_classified_preset(routed_home, fake_llm):
    fake_llm.handlers["moa_router"] = lambda kwargs: _response("coding")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:auto",
                "messages": [{"role": "user", "content": "fix my bug"}],
            },
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model"] == "moa:coding"
        assert body["usage"]["moa"]["routed_preset"] == "coding"
        assert body["usage"]["moa"]["routing"]["method"] == "classified"
        # coding preset's reference model ran
        ref_calls = [c for c in fake_llm.calls if c.get("task") == "moa_reference"]
        assert any(c.get("model") == "ref-model-a" for c in ref_calls)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_self_answer_skips_fanout(routed_home, fake_llm):
    fake_llm.handlers["moa_router"] = lambda kwargs: _response("self")
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: _response("hello there")
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:auto", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model"] == "moa:self"
        assert body["choices"][0]["message"]["content"] == "hello there"
        assert body["usage"]["moa"]["routed_preset"] == "self"
        assert body["usage"]["moa"]["references"] == []
        # No reference fan-out ran; the acting call used the classifier slot.
        assert not [c for c in fake_llm.calls if c.get("task") == "moa_reference"]
        agg_calls = [c for c in fake_llm.calls if c.get("task") == "moa_aggregator"]
        assert agg_calls and agg_calls[0].get("model") == "fast-classifier"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_classifier_failure_uses_default(routed_home, fake_llm):
    def boom(kwargs):
        raise RuntimeError("classifier down")

    fake_llm.handlers["moa_router"] = boom
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:auto", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["model"] == "moa:general"
        assert body["usage"]["moa"]["routing"]["method"] == "fallback"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_streaming_announces_route_first(routed_home, fake_llm):
    fake_llm.handlers["moa_router"] = lambda kwargs: _response("coding")
    fake_llm.handlers["moa_aggregator"] = lambda kwargs: iter(
        [_delta_chunk(content="done"), _delta_chunk(finish_reason="stop")]
    )
    client = await _client(create_moa_app())
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:auto",
                "messages": [{"role": "user", "content": "fix my bug"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        events = await _read_sse(resp)
        chunks = [e for e in events if e != "[DONE]"]
        first_reasoning = next(
            c["choices"][0]["delta"].get("reasoning_content")
            for c in chunks
            if c["choices"] and c["choices"][0]["delta"].get("reasoning_content")
        )
        assert first_reasoning.startswith("[moa:auto → 'coding'")
        usage_chunk = next(c for c in chunks if c.get("usage"))
        assert usage_chunk["usage"]["moa"]["routed_preset"] == "coding"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_auto_sticky_across_requests(routed_home, fake_llm):
    replies = iter(["coding", "general"])
    fake_llm.handlers["moa_router"] = lambda kwargs: _response(next(replies))
    client = await _client(create_moa_app())
    try:
        for _ in range(2):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "moa:auto",
                    "messages": [{"role": "user", "content": "same conversation"}],
                },
                headers={"x-hermes-session-id": "conv-1"},
            )
            body = await resp.json()
            assert body["model"] == "moa:coding"
        router_calls = [c for c in fake_llm.calls if c.get("task") == "moa_router"]
        assert len(router_calls) == 1
    finally:
        await client.close()
