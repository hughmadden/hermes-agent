"""Quorum straggler-dropping in the MoA reference fan-out."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

import agent.moa_loop as moa_loop
from agent.moa_loop import _run_references_parallel


def _response(text, usage=None):
    message = SimpleNamespace(content=text, tool_calls=[], reasoning_content=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
    )


def _usage(prompt=10, completion=5):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, prompt_tokens_details=None
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


# ---------------------------------------------------------------------------
# Addendum v1.3 -- RLM voter slots (iteration 31)
#
# `agent: "rlm"` on a reference slot runs the reason->python->observe loop
# (mirroring scripts/moa_rlm_bench.py) INSIDE `_run_reference` when the call
# is `direct=True` (cascade voters); a bare fan-out advisory call (direct
# defaults to False) ignores the flag entirely. Python fences are executed
# via `hermes_cli.proxy.moa_cascade.run_rlm_exec`, monkeypatched here as
# instructed by the addendum rather than exercising a real subprocess.
# ---------------------------------------------------------------------------


def test_rlm_agent_loop_executes_python_fence_then_returns_final(monkeypatch):
    """A python-fence turn gets executed via run_rlm_exec and fed back as an
    OUTPUT user turn; the loop's returned text is the FINAL assistant
    message; usage sums across BOTH loop turns into one accounting entry;
    the label gains a " [rlm]" suffix."""
    calls = {"n": 0}
    seen_messages = []

    def fake_call_llm(**kwargs):
        calls["n"] += 1
        seen_messages.append([dict(m) for m in kwargs["messages"]])
        assert kwargs.get("task") == "moa_reference"
        if calls["n"] == 1:
            return _response(
                "Let's compute.\n```python\nprint(2 + 2)\n```", usage=_usage(10, 5)
            )
        return _response("Verified.\nFINAL: 4", usage=_usage(20, 5))

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    exec_calls = []

    def fake_run_rlm_exec(code):
        exec_calls.append(code)
        return "4\n"

    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_rlm_exec", fake_run_rlm_exec, raising=False
    )

    slot = {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm"}
    label, text, acct = moa_loop._run_reference(
        slot,
        [{"role": "user", "content": "what is 2+2?"}],
        temperature=0.4,
        max_tokens=None,
        timeout=None,
        direct=True,
    )

    assert calls["n"] == 2
    assert exec_calls == ["print(2 + 2)\n"]
    assert "FINAL: 4" in text
    assert label == "custom:gemma-4-31b [rlm]"

    # The second call's messages must carry the sandbox output as an OUTPUT
    # user turn -- the loop feeds observations back, not just the raw reply.
    second_call_messages = seen_messages[1]
    output_turns = [
        m
        for m in second_call_messages
        if m.get("role") == "user" and "OUTPUT" in str(m.get("content"))
    ]
    assert output_turns, f"expected an OUTPUT user turn, got: {second_call_messages!r}"
    assert any("4" in str(m.get("content")) for m in output_turns)

    # Usage sums across both loop turns into a SINGLE accounting entry (not
    # just the last turn's usage) -- token counts add, not just the final call.
    assert acct.usage.prompt_tokens == 30
    assert acct.usage.output_tokens == 10


def test_rlm_agent_loop_forces_final_on_round_cap(monkeypatch):
    """The model never emits a python fence or FINAL within the configured
    `rlm_rounds` budget -- the loop must force one extra no-tool final round
    (mirroring moa_rlm_bench.run_rlm) rather than giving up with no answer."""
    calls = {"n": 0}

    def fake_call_llm(**kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return _response("Still thinking, no answer yet.")
        # The forced final round: no code fence should be honored even if
        # present, and the answer must come back as FINAL.
        return _response("FINAL: 9")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    monkeypatch.setattr(
        "hermes_cli.proxy.moa_cascade.run_rlm_exec", lambda code: "unused", raising=False
    )

    slot = {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm", "rlm_rounds": 2}
    label, text, acct = moa_loop._run_reference(
        slot, [{"role": "user", "content": "4+5?"}], direct=True,
    )

    # rlm_rounds=2 normal rounds + 1 forced final round = 3 calls total.
    assert calls["n"] == 3
    assert "FINAL: 9" in text


def test_rlm_agent_loop_infra_failure_returns_failed_note(monkeypatch):
    """A loop-infrastructure exception (e.g. the upstream call_llm raising)
    must degrade to the standard "[failed: ...]" note -- exactly like a
    plain (non-rlm) reference failure -- never break the fan-out."""

    def raising_call_llm(**kwargs):
        raise RuntimeError("upstream boom")

    monkeypatch.setattr(moa_loop, "call_llm", raising_call_llm)

    slot = {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm"}
    label, text, acct = moa_loop._run_reference(
        slot, [{"role": "user", "content": "q"}], direct=True,
    )
    assert text.startswith("[failed:")
    assert "upstream boom" in text


def test_rlm_agent_flag_ignored_when_not_direct(monkeypatch):
    """Advisory fan-out (direct=False, the default) ignores `agent: rlm`
    entirely -- the loop is for cascade voters ANSWERING the client's
    request, not for advice, which is prose handed to an aggregator."""
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        return _response("plain advisory answer")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)

    slot = {"provider": "custom", "model": "gemma-4-31b", "agent": "rlm"}
    label, text, acct = moa_loop._run_reference(
        slot, [{"role": "user", "content": "q"}], direct=False,
    )

    assert len(calls) == 1  # no loop -- exactly one normal advisory call
    assert calls[0]["messages"][0]["role"] == "system"
    assert calls[0]["messages"][0]["content"] == moa_loop._REFERENCE_SYSTEM_PROMPT
    assert text == "plain advisory answer"
    assert label == "custom:gemma-4-31b"  # no " [rlm]" suffix outside the loop
