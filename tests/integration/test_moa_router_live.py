"""Live tests for moa:auto routing through the OpenAI-compatible endpoint.

Two classifier deployments are covered, each gated on its credential:

- Cerebras ``gemma-4-31b`` via the Hermes custom-provider path (the reference
  deployment: wafer-speed classification). Needs CEREBRAS_API_KEY.
- An OpenRouter-hosted fast model as the drop-in substitute. Needs
  OPENROUTER_API_KEY (which the MoA presets need anyway).

Run with:  pytest -m integration tests/integration/test_moa_router_live.py
"""

from __future__ import annotations

import json
import os

import pytest

# Captured at import time — the autouse hermetic fixture wipes credential env
# vars before each test; live fixtures re-set them via monkeypatch.
_OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
_CEREBRAS_KEY = os.environ.get("CEREBRAS_API_KEY", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _OPENROUTER_KEY, reason="OPENROUTER_API_KEY not set"),
]

aiohttp_test_utils = pytest.importorskip("aiohttp.test_utils")
TestClient = aiohttp_test_utils.TestClient
TestServer = aiohttp_test_utils.TestServer

_REF = "deepseek/deepseek-v4-flash"
_AGG = "openai/gpt-5.4-mini"
_OPENROUTER_CLASSIFIER = "google/gemma-4-31b-it"


def _routed_config(classifier_provider: str, classifier_model: str) -> str:
    providers_block = ""
    if classifier_provider.startswith("custom:"):
        providers_block = (
            "providers:\n"
            "  cerebras:\n"
            "    api: https://api.cerebras.ai/v1\n"
            "    name: cerebras\n"
            "    default_model: gemma-4-31b\n"
        )
    return (
        providers_block
        + f"""
moa:
  default_preset: general
  save_traces: true
  router:
    enabled: true
    classifier:
      provider: {classifier_provider}
      model: {classifier_model}
    default: general
    timeout_s: 20
  presets:
    coding:
      route:
        description: writing or debugging code, refactors, shell commands, programming questions
      reference_models:
        - provider: openrouter
          model: {_REF}
      aggregator:
        provider: openrouter
        model: {_AGG}
      reference_max_tokens: 300
    general:
      route:
        description: general knowledge, writing, analysis — everything that is not code and not trivial
      reference_models:
        - provider: openrouter
          model: {_REF}
      aggregator:
        provider: openrouter
        model: {_AGG}
      reference_max_tokens: 300
""".strip()
    )


def _make_home(tmp_path, monkeypatch, provider: str, model: str):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        _routed_config(provider, model), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OPENROUTER_API_KEY", _OPENROUTER_KEY)
    if _CEREBRAS_KEY:
        monkeypatch.setenv("CEREBRAS_API_KEY", _CEREBRAS_KEY)
    import hermes_cli.proxy.moa_router as moa_router
    import hermes_cli.proxy.moa_server as moa_server

    moa_server._ref_cache.clear()
    moa_router.sticky_clear()
    return home


@pytest.fixture()
def openrouter_home(monkeypatch, tmp_path):
    return _make_home(tmp_path, monkeypatch, "openrouter", _OPENROUTER_CLASSIFIER)


@pytest.fixture()
def cerebras_home(monkeypatch, tmp_path):
    return _make_home(tmp_path, monkeypatch, "custom:cerebras", "gemma-4-31b")


async def _client():
    from hermes_cli.proxy.moa_server import create_moa_app

    server = TestServer(create_moa_app())
    client = TestClient(server)
    await client.start_server()
    return client


async def _post_auto(client, content, session=None):
    headers = {"x-hermes-session-id": session} if session else {}
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "moa:auto",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 600,
        },
        headers=headers,
    )
    assert resp.status == 200, await resp.text()
    return await resp.json()


@pytest.mark.asyncio
async def test_live_openrouter_classifier_routes_coding(openrouter_home):
    client = await _client()
    try:
        body = await _post_auto(
            client,
            "Debug this Python: `def f(x): return x +` — fix the syntax error.",
        )
        routing = body["usage"]["moa"]["routing"]
        assert routing["method"] == "classified", routing
        assert body["usage"]["moa"]["routed_preset"] == "coding", routing
        assert body["model"] == "moa:coding"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_openrouter_classifier_self_answers_trivial(openrouter_home):
    client = await _client()
    try:
        body = await _post_auto(client, "hi! how are you today?")
        routing = body["usage"]["moa"]["routing"]
        assert routing["method"] == "classified", routing
        assert body["usage"]["moa"]["routed_preset"] == "self", routing
        # SELF class: no reference fan-out ran, and the fast model answered.
        assert body["usage"]["moa"]["references"] == []
        assert (body["choices"][0]["message"]["content"] or "").strip()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_sticky_reuses_decision(openrouter_home):
    client = await _client()
    try:
        first = await _post_auto(
            client, "Write a bash one-liner to count files.", session="live-conv"
        )
        second = await _post_auto(
            client, "Now explain what it does.", session="live-conv"
        )
        assert (
            second["usage"]["moa"]["routing"]["method"] == "sticky"
        ), second["usage"]["moa"]["routing"]
        assert (
            second["usage"]["moa"]["routed_preset"]
            == first["usage"]["moa"]["routed_preset"]
        )
    finally:
        await client.close()


@pytest.mark.skipif(not _CEREBRAS_KEY, reason="CEREBRAS_API_KEY not set")
@pytest.mark.asyncio
async def test_live_cerebras_classifier_end_to_end(cerebras_home):
    """The reference deployment: Cerebras gemma-4-31b classifies, and for the
    SELF class also acts. Asserts the added routing latency stays in the fast
    lane (classifier_ms budget from the backlog: low hundreds of ms; allow
    generous headroom for cold TLS)."""
    client = await _client()
    try:
        body = await _post_auto(client, "thanks, that's all for today!")
        routing = body["usage"]["moa"]["routing"]
        assert routing["method"] == "classified", routing
        assert routing["classifier_ms"] is not None
        assert routing["classifier_ms"] < 5000, routing
        assert body["usage"]["moa"]["routed_preset"] == "self", routing

        body = await _post_auto(
            client, "Refactor this JavaScript function to use async/await."
        )
        assert body["usage"]["moa"]["routed_preset"] == "coding", body["usage"]["moa"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_live_routing_recorded_in_traces(openrouter_home):
    client = await _client()
    try:
        await _post_auto(client, "hello there!", session="trace-conv")
        trace_dir = openrouter_home / "moa-traces"
        records = []
        for path in trace_dir.glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    records.append(json.loads(line))
        assert records, "routed turn wrote no trace"
        routed = [r for r in records if r.get("routing")]
        assert routed, "trace record missing routing info"
        assert routed[-1]["routing"]["requested"] == "auto"
        assert routed[-1]["routing"]["routed_preset"]
    finally:
        await client.close()
