"""Quorum straggler-dropping in the MoA reference fan-out."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import agent.moa_loop as moa_loop
from agent.moa_loop import _run_references_parallel


def _response(text):
    message = SimpleNamespace(content=text, tool_calls=[], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
    )


@pytest.fixture()
def fake_llm(monkeypatch):
    delays = {}

    def fake_call_llm(**kwargs):
        model = kwargs.get("model")
        time.sleep(delays.get(model, 0))
        return _response(f"advice from {model}")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    return delays


SLOTS = [
    {"provider": "openrouter", "model": "fast-a"},
    {"provider": "openrouter", "model": "fast-b"},
    {"provider": "openrouter", "model": "slow-c"},
]
MSGS = [{"role": "user", "content": "q"}]


def test_no_quorum_waits_for_all(fake_llm):
    fake_llm["slow-c"] = 0.4
    started = time.time()
    outputs = _run_references_parallel(SLOTS, MSGS)
    assert time.time() - started >= 0.4
    assert [t for _, t, _ in outputs] == [
        "advice from fast-a", "advice from fast-b", "advice from slow-c"
    ]


def test_quorum_drops_straggler(fake_llm):
    fake_llm["fast-a"] = 0.05
    fake_llm["fast-b"] = 0.05
    fake_llm["slow-c"] = 30  # would dominate the turn
    started = time.time()
    outputs = _run_references_parallel(SLOTS, MSGS, quorum_grace=0.5)
    elapsed = time.time() - started
    assert elapsed < 5, f"straggler was not dropped ({elapsed:.1f}s)"
    labels_texts = {label: text for label, text, _ in outputs}
    assert labels_texts["openrouter:fast-a"] == "advice from fast-a"
    assert "dropped" in labels_texts["openrouter:slow-c"]
    # Order preserved
    assert [label for label, _, _ in outputs] == [
        "openrouter:fast-a", "openrouter:fast-b", "openrouter:slow-c"
    ]


def test_quorum_grace_lets_close_straggler_finish(fake_llm):
    fake_llm["fast-a"] = 0.1
    fake_llm["fast-b"] = 0.1
    fake_llm["slow-c"] = 0.6  # within grace: 0.1s quorum + max(1s floor) grace
    outputs = _run_references_parallel(SLOTS, MSGS, quorum_grace=0.5)
    assert [t for _, t, _ in outputs][2] == "advice from slow-c"


def test_quorum_ignored_for_single_reference(fake_llm):
    fake_llm["slow-c"] = 0.3
    outputs = _run_references_parallel([SLOTS[2]], MSGS, quorum_grace=0.5)
    assert outputs[0][1] == "advice from slow-c"


def test_per_slot_max_tokens_override(fake_llm, monkeypatch):
    """A slot's own max_tokens overrides the fan-out-level cap (thinking
    voters need caps >= their reasoning budget)."""
    import agent.moa_loop as moa_loop

    captured = []
    real = moa_loop.call_llm

    def spy(**kwargs):
        captured.append(kwargs.get("max_tokens"))
        return real(**kwargs)

    monkeypatch.setattr(moa_loop, "call_llm", spy)
    slots = [
        {"provider": "openrouter", "model": "fast-a"},
        {"provider": "openrouter", "model": "thinker", "max_tokens": 16000},
    ]
    moa_loop._run_references_parallel(slots, MSGS, max_tokens=800)
    assert sorted(captured) == [800, 16000]
