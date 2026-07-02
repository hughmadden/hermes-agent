"""Unit tests for moa:auto routing (hermes_cli/proxy/moa_router.py).

All classifier calls are faked at the agent.moa_loop.call_llm seam — the same
one the rest of the MoA stack patches. Live routing coverage (real Cerebras/
OpenRouter classifier) lives in tests/integration/test_moa_router_live.py.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

import hermes_cli.proxy.moa_router as moa_router
from hermes_cli.moa_config import normalize_moa_config
from hermes_cli.proxy.moa_router import (
    RouteDecision,
    is_auto_model,
    route_request,
    self_answer_preset,
    sticky_clear,
    sticky_key,
)

ROUTED_CFG = {
    "default_preset": "general",
    "router": {
        "enabled": True,
        "classifier": {"provider": "openrouter", "model": "fast-classifier"},
        "default": "general",
        "timeout_s": 2,
    },
    "presets": {
        "coding": {
            "route": {"description": "code writing, debugging, refactors, shell"},
            "reference_models": [{"provider": "openrouter", "model": "r1"}],
            "aggregator": {"provider": "openrouter", "model": "a1"},
        },
        "math": {
            "route": {"description": "calculation, proofs, quantitative puzzles"},
            "reference_models": [{"provider": "openrouter", "model": "r2"}],
            "aggregator": {"provider": "openrouter", "model": "a2"},
        },
        "general": {
            "reference_models": [{"provider": "openrouter", "model": "r3"}],
            "aggregator": {"provider": "openrouter", "model": "a3"},
        },
    },
}


def _cfg(**router_overrides):
    raw = {
        **ROUTED_CFG,
        "router": {**ROUTED_CFG["router"], **router_overrides},
    }
    return normalize_moa_config(raw)


def _response(text):
    message = SimpleNamespace(content=text, tool_calls=[], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=None,
    )


@pytest.fixture(autouse=True)
def clean_sticky():
    sticky_clear()
    yield
    sticky_clear()


@pytest.fixture()
def fake_classifier(monkeypatch):
    calls: list[dict] = []
    state = {"reply": "coding", "raise": None, "delay": 0.0}

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if state["delay"]:
            time.sleep(state["delay"])
        if state["raise"]:
            raise state["raise"]
        return _response(state["reply"])

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    state["calls"] = calls
    return state


def _messages(text="write a python function"):
    return [{"role": "user", "content": text}]


# ---------------------------------------------------------------------------
# Config normalization
# ---------------------------------------------------------------------------


def test_router_config_normalization():
    cfg = _cfg()
    router = cfg["router"]
    assert router["enabled"] is True
    assert router["classifier"] == {"provider": "openrouter", "model": "fast-classifier"}
    assert router["routable_presets"] == ["coding", "math"]  # general: no route block
    assert router["default"] == "general"
    assert router["self_answer"] is True
    assert router["self_answer_model"] == router["classifier"]


def test_router_disabled_without_classifier():
    cfg = _cfg(classifier=None)
    assert cfg["router"]["enabled"] is False


def test_router_disabled_without_routable_presets():
    raw = {
        "router": {
            "enabled": True,
            "classifier": {"provider": "openrouter", "model": "fast"},
        },
        "presets": {
            "general": {
                "reference_models": [{"provider": "openrouter", "model": "r"}],
                "aggregator": {"provider": "openrouter", "model": "a"},
            }
        },
    }
    assert normalize_moa_config(raw)["router"]["enabled"] is False


def test_router_default_falls_back_to_first_routable():
    cfg = _cfg(default="nonexistent")
    assert cfg["router"]["default"] == "coding"


# ---------------------------------------------------------------------------
# Model-field detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["moa:auto", "moa/auto", "auto", "MOA:AUTO"])
def test_is_auto_model_variants(field):
    assert is_auto_model(field) is True


@pytest.mark.parametrize("field", ["moa:coding", "coding", "", None, "moa", "autopilot"])
def test_is_auto_model_negatives(field):
    assert is_auto_model(field) is False


# ---------------------------------------------------------------------------
# Classification outcomes
# ---------------------------------------------------------------------------


def test_route_classifies_to_preset(fake_classifier):
    fake_classifier["reply"] = "coding"
    decision = asyncio.run(route_request(_cfg(), _messages()))
    assert decision.preset_name == "coding"
    assert decision.is_self is False
    assert decision.method == "classified"
    assert decision.classifier_ms is not None
    # Classifier saw the preset descriptions and the self class.
    prompt = fake_classifier["calls"][0]["messages"][0]["content"]
    assert "coding:" in prompt and "math:" in prompt and "self:" in prompt


def test_route_self_answer_class(fake_classifier):
    fake_classifier["reply"] = "self"
    decision = asyncio.run(route_request(_cfg(), _messages("hi!")))
    assert decision.is_self is True
    assert decision.preset_name == moa_router.SELF_CLASS


def test_route_self_disabled_excludes_class(fake_classifier):
    fake_classifier["reply"] = "self"
    decision = asyncio.run(route_request(_cfg(self_answer=False), _messages("hi!")))
    # "self" is no longer a valid label -> fallback to default.
    assert decision.is_self is False
    assert decision.preset_name == "general"
    assert decision.method == "fallback"
    prompt = fake_classifier["calls"][0]["messages"][0]["content"]
    assert "self:" not in prompt


def test_route_tolerant_label_parsing(fake_classifier):
    fake_classifier["reply"] = '  "Coding".  '
    decision = asyncio.run(route_request(_cfg(), _messages()))
    assert decision.preset_name == "coding"


def test_route_unknown_label_falls_back(fake_classifier):
    fake_classifier["reply"] = "quantum-basket-weaving"
    decision = asyncio.run(route_request(_cfg(), _messages()))
    assert decision.preset_name == "general"
    assert decision.method == "fallback"


def test_route_classifier_error_falls_back(fake_classifier):
    fake_classifier["raise"] = RuntimeError("classifier exploded")
    decision = asyncio.run(route_request(_cfg(), _messages()))
    assert decision.preset_name == "general"
    assert decision.method == "fallback"
    assert "error" in decision.reason


def test_route_classifier_timeout_falls_back(fake_classifier):
    fake_classifier["delay"] = 0.5
    decision = asyncio.run(route_request(_cfg(timeout_s=0.05), _messages()))
    assert decision.preset_name == "general"
    assert decision.method == "fallback"
    assert "timeout" in decision.reason


# ---------------------------------------------------------------------------
# Sticky sessions
# ---------------------------------------------------------------------------


def test_sticky_by_session_id(fake_classifier):
    fake_classifier["reply"] = "math"
    first = asyncio.run(route_request(_cfg(), _messages("integrate x^2"), session_id="s1"))
    assert first.method == "classified"
    fake_classifier["reply"] = "coding"  # would flip if re-classified
    second = asyncio.run(
        route_request(_cfg(), _messages("now do the next step"), session_id="s1")
    )
    assert second.preset_name == "math"
    assert second.method == "sticky"
    assert len(fake_classifier["calls"]) == 1


def test_sticky_by_first_user_message(fake_classifier):
    fake_classifier["reply"] = "coding"
    convo = [{"role": "user", "content": "fix this bug"}]
    first = asyncio.run(route_request(_cfg(), convo))
    # Tool loop appends results; first user message unchanged.
    convo_later = convo + [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "t1"}]},
        {"role": "tool", "content": "ran"},
    ]
    second = asyncio.run(route_request(_cfg(), convo_later))
    assert first.preset_name == second.preset_name == "coding"
    assert second.method == "sticky"
    assert len(fake_classifier["calls"]) == 1


def test_sticky_key_prefers_session_id():
    assert sticky_key([], "abc") == "sid:abc"
    k1 = sticky_key([{"role": "user", "content": "hello"}], None)
    k2 = sticky_key([{"role": "user", "content": "hello"}], None)
    k3 = sticky_key([{"role": "user", "content": "different"}], None)
    assert k1 == k2 != k3


def test_fallback_decision_is_sticky_too(fake_classifier):
    fake_classifier["raise"] = RuntimeError("down")
    first = asyncio.run(route_request(_cfg(), _messages(), session_id="s2"))
    assert first.method == "fallback"
    fake_classifier["raise"] = None
    fake_classifier["reply"] = "coding"
    second = asyncio.run(route_request(_cfg(), _messages(), session_id="s2"))
    # Conversation keeps its routing; no mid-conversation preset flip.
    assert second.preset_name == first.preset_name
    assert second.method == "sticky"


# ---------------------------------------------------------------------------
# Self-answer pseudo-preset
# ---------------------------------------------------------------------------


def test_self_answer_preset_shape():
    router = _cfg()["router"]
    preset = self_answer_preset(router)
    assert preset["enabled"] is False
    assert preset["reference_models"] == []
    assert preset["aggregator"] == router["classifier"]


def test_self_answer_preset_custom_model():
    router = _cfg(self_answer_model={"provider": "openrouter", "model": "bigger"})["router"]
    assert self_answer_preset(router)["aggregator"] == {
        "provider": "openrouter",
        "model": "bigger",
    }


def test_route_decision_trace_shape():
    decision = RouteDecision(
        preset_name="coding", is_self=False, method="classified",
        reason="classified in 120ms", classifier_ms=120,
    )
    trace = decision.as_trace()
    assert trace["requested"] == "auto"
    assert trace["routed_preset"] == "coding"
    assert trace["method"] == "classified"
    assert trace["classifier_ms"] == 120
