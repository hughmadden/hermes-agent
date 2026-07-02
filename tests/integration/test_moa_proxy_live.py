"""Live end-to-end tests for the OpenAI-compatible MoA endpoint.

Exercises `hermes moa serve` against REAL OpenRouter models: reference
fan-out, aggregator streaming with reasoning deltas, and a full client-side
tool-call round trip. Requires OPENROUTER_API_KEY in the environment and
spends a small amount of real credit, so it is gated behind the
``integration`` marker (excluded by default via addopts).

Run with:  pytest -m integration tests/integration/test_moa_proxy_live.py
"""

from __future__ import annotations

import json
import os

import pytest

# Capture the key at import (collection) time: the autouse
# _hermetic_environment fixture in tests/conftest.py deletes every
# credential-shaped env var before each test, so the fixture below must
# re-set it explicitly for this opt-in live suite.
_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set"),
]

aiohttp_test_utils = pytest.importorskip("aiohttp.test_utils")
TestClient = aiohttp_test_utils.TestClient
TestServer = aiohttp_test_utils.TestServer

# Cheap, tool-capable OpenRouter models (from the curated featured list).
_REF_A = "google/gemini-3.5-flash"
_REF_B = "deepseek/deepseek-v4-flash"
_AGGREGATOR = "openai/gpt-5.4-mini"

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


@pytest.fixture()
def moa_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        f"""
moa:
  default_preset: live
  save_traces: true
  presets:
    live:
      reference_models:
        - provider: openrouter
          model: {_REF_A}
        - provider: openrouter
          model: {_REF_B}
      aggregator:
        provider: openrouter
        model: {_AGGREGATOR}
      reference_max_tokens: 400
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_KEY)
    import agent.moa_loop as moa_loop
    import hermes_cli.proxy.moa_server as moa_server

    moa_server._ref_cache.clear()
    moa_loop._skill_cache.clear()
    return home


async def _client():
    from hermes_cli.proxy.moa_server import create_moa_app

    server = TestServer(create_moa_app())
    client = TestClient(server)
    await client.start_server()
    return client


async def _read_sse(resp):
    events = []
    raw = await resp.text()
    for line in raw.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            events.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return events


def _reasoning_text(events):
    return "".join(
        e["choices"][0]["delta"].get("reasoning_content", "")
        for e in events
        if e != "[DONE]" and e.get("choices")
    )


def _content_text(events):
    return "".join(
        e["choices"][0]["delta"].get("content") or ""
        for e in events
        if e != "[DONE]" and e.get("choices")
    )


def _assemble_tool_calls(events):
    calls: dict[int, dict] = {}
    for e in events:
        if e == "[DONE]" or not e.get("choices"):
            continue
        for tc in e["choices"][0]["delta"].get("tool_calls") or []:
            entry = calls.setdefault(
                tc.get("index", 0),
                {"id": "", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                entry["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                entry["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                entry["function"]["arguments"] += fn["arguments"]
    return [calls[i] for i in sorted(calls)]


@pytest.mark.asyncio
async def test_live_non_streaming_completion(moa_home):
    client = await _client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:live",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is 17 * 23? Answer with just the number.",
                    }
                ],
                "max_tokens": 4000,
            },
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        content = body["choices"][0]["message"]["content"]
        assert "391" in content
        assert body["usage"]["total_tokens"] > 0
        assert len(body["usage"]["moa"]["references"]) == 2
        assert body["usage"]["moa"]["aggregator"]["total_tokens"] > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_streaming_thinking_then_answer(moa_home):
    client = await _client()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:live",
                "messages": [
                    {
                        "role": "user",
                        "content": "In one short sentence: why is the sky blue?",
                    }
                ],
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 4000,
            },
        )
        assert resp.status == 200, await resp.text()
        events = await _read_sse(resp)
        assert events[-1] == "[DONE]"
        reasoning = _reasoning_text(events)
        content = _content_text(events)
        # Both references and the aggregating marker streamed as thinking.
        assert f"— openrouter:{_REF_A}]" in reasoning
        assert f"— openrouter:{_REF_B}]" in reasoning
        assert "[Aggregating — " in reasoning
        assert len(reasoning) > 100  # actual advisory text, not just labels
        assert content.strip()  # the acting answer arrived as plain content
        usage_chunks = [e for e in events if e != "[DONE]" and not e.get("choices")]
        assert usage_chunks and usage_chunks[0]["usage"]["total_tokens"] > 0
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_client_side_tool_round_trip(moa_home):
    """Leg 1: the aggregator must call the client's tool. Leg 2: the client
    returns the tool result and the aggregator must use it in its answer."""
    client = await _client()
    try:
        messages = [
            {
                "role": "user",
                "content": (
                    "What's the weather in Hong Kong right now? "
                    "You MUST use the get_weather tool; do not guess."
                ),
            }
        ]
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:live",
                "messages": messages,
                "tools": [_WEATHER_TOOL],
                "stream": True,
                "max_tokens": 4000,
            },
        )
        assert resp.status == 200, await resp.text()
        events = await _read_sse(resp)
        tool_calls = _assemble_tool_calls(events)
        assert tool_calls, f"aggregator made no tool call; content={_content_text(events)!r}"
        call = tool_calls[0]
        assert call["function"]["name"] == "get_weather"
        args = json.loads(call["function"]["arguments"])
        assert "hong kong" in str(args.get("city", "")).lower()
        finish = [
            e["choices"][0]["finish_reason"]
            for e in events
            if e != "[DONE]" and e.get("choices") and e["choices"][0]["finish_reason"]
        ]
        assert finish[-1] == "tool_calls"

        # Leg 2 — the client executed the tool; feed the result back.
        messages = messages + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call["id"] or "call_live_1",
                        "type": "function",
                        "function": call["function"],
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call["id"] or "call_live_1",
                "content": '{"temperature_c": 31, "condition": "thunderstorms"}',
            },
        ]
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:live",
                "messages": messages,
                "tools": [_WEATHER_TOOL],
                "max_tokens": 4000,
            },
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        answer = (body["choices"][0]["message"]["content"] or "").lower()
        assert "31" in answer or "thunderstorm" in answer, answer
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_evolve_loop_end_to_end(moa_home):
    """The full skills loop against real models: proxied turns write traces,
    `hermes moa evolve` distills them into the moa-aggregation skill, and the
    next turn's aggregator prompt carries the distilled heuristics."""
    from types import SimpleNamespace

    from hermes_cli.moa_evolve import cmd_moa_evolve

    client = await _client()
    try:
        for prompt in (
            "Is a tomato a fruit or a vegetable, botanically? One sentence.",
            "What is 12 * 12? Just the number.",
        ):
            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "moa:live",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 2000,
                },
            )
            assert resp.status == 200, await resp.text()

        trace_dir = moa_home / "moa-traces"
        trace_files = list(trace_dir.glob("*.jsonl"))
        assert trace_files, "proxied turns did not write MoA traces"

        # Distill with a cheap model.
        rc = cmd_moa_evolve(
            SimpleNamespace(
                max_turns=10,
                model=f"openrouter:{_REF_A}",
                trace_dir=None,
                dry_run=False,
            )
        )
        assert rc == 0
        skill_path = moa_home / "skills" / "moa-aggregation" / "SKILL.md"
        content = skill_path.read_text(encoding="utf-8")
        assert "auto_generated: moa-evolve" in content
        assert len(content) > 200, content  # real heuristics, not an empty shell

        # A fresh turn must inject the distilled heuristics into the
        # aggregator guidance — visible in the newly written trace record.
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "moa:live",
                "messages": [
                    {"role": "user", "content": "Name the largest planet. One word."}
                ],
                "max_tokens": 1000,
            },
        )
        assert resp.status == 200, await resp.text()
        records = []
        for path in trace_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        records.sort(key=lambda r: r.get("ts") or 0)
        last_agg_input = records[-1]["aggregator"]["input_messages"]
        joined = "\n".join(
            str(m.get("content")) for m in last_agg_input if isinstance(m, dict)
        )
        assert "[Aggregation heuristics" in joined
    finally:
        await client.close()
