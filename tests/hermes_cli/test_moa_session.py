"""Tests for the voter-history projection + session registry
(``hermes_cli/proxy/moa_session.py``).

Conventions mirror ``tests/hermes_cli/test_moa_cascade.py``: plain pytest,
no server/aiohttp fixtures needed since both units under test are pure
Python (``project_history_for_voters``) or in-memory-only
(``SessionRegistry`` — no I/O, no background threads).
"""

from __future__ import annotations

import hashlib
import threading

import pytest

from hermes_cli.proxy.moa_session import (
    SessionRegistry,
    project_history_for_voters,
)


# ---------------------------------------------------------------------------
# project_history_for_voters — identity / plain passthrough
# ---------------------------------------------------------------------------


def test_plain_history_projects_byte_identically():
    """A message with role in {system,user,assistant}, plain string content,
    and no opaque fields passes through UNCHANGED (same dict value)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "what is 6*7?"},
        {"role": "assistant", "content": "42"},
    ]
    projected = project_history_for_voters(messages)
    assert projected == messages
    # Identity, not just equality: plain messages are returned as the same
    # object, not a copy.
    for orig, proj in zip(messages, projected):
        assert orig is proj


def test_plain_message_with_name_field_kept():
    messages = [{"role": "user", "content": "hi", "name": "hugh"}]
    projected = project_history_for_voters(messages)
    assert projected == messages
    assert projected[0] is messages[0]


# ---------------------------------------------------------------------------
# Content-list flattening
# ---------------------------------------------------------------------------


def test_content_list_text_parts_joined_with_newline():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first part"},
                {"type": "text", "text": "second part"},
            ],
        }
    ]
    projected = project_history_for_voters(messages)
    assert projected == [{"role": "user", "content": "first part\nsecond part"}]


def test_content_list_non_text_part_becomes_bracketed_note():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://example/x.png"}},
            ],
        }
    ]
    projected = project_history_for_voters(messages)
    assert projected[0]["content"] == "look at this\n[non-text content: image_url]"


def test_content_list_all_non_text_still_produces_content():
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]
    projected = project_history_for_voters(messages)
    assert projected[0]["content"] == "[non-text content: image_url]"


# ---------------------------------------------------------------------------
# tool_calls rendering
# ---------------------------------------------------------------------------


def test_assistant_tool_calls_rendered_and_field_dropped():
    messages = [
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q": "weather"}'},
                }
            ],
        }
    ]
    projected = project_history_for_voters(messages)
    out = projected[0]
    assert out["role"] == "assistant"
    assert "tool_calls" not in out
    assert "Let me check that." in out["content"]
    assert '[tool call lookup({"q": "weather"})]' in out["content"]


def test_assistant_tool_calls_no_text_content_still_renders_calls():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "ping", "arguments": "{}"}}
            ],
        }
    ]
    projected = project_history_for_voters(messages)
    assert projected[0]["content"] == "[tool call ping({})]"


def test_multiple_tool_calls_each_rendered_as_own_line():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "a", "arguments": "{}"}},
                {"function": {"name": "b", "arguments": "{}"}},
            ],
        }
    ]
    projected = project_history_for_voters(messages)
    lines = projected[0]["content"].splitlines()
    assert lines == ["[tool call a({})]", "[tool call b({})]"]


def test_tool_call_arguments_capped_at_500_chars():
    big_args = '{"data": "' + ("x" * 1000) + '"}'
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "write", "arguments": big_args}}],
        }
    ]
    projected = project_history_for_voters(messages)
    content = projected[0]["content"]
    assert "...[truncated]" in content
    # The rendered args text itself (inside the parens) must be capped.
    inner = content[len("[tool call write(") : -len(")]")]
    assert len(inner) <= 500 + len("...[truncated]")


# ---------------------------------------------------------------------------
# tool-role -> user-role conversion + budget trimming
# ---------------------------------------------------------------------------


def test_tool_result_becomes_user_role_with_call_id():
    messages = [
        {"role": "tool", "tool_call_id": "call_42", "content": "the answer is 7"}
    ]
    projected = project_history_for_voters(messages)
    assert projected == [
        {"role": "user", "content": "[tool result call_42]\nthe answer is 7"}
    ]


def test_tool_result_missing_call_id_uses_question_mark():
    messages = [{"role": "tool", "content": "result text"}]
    projected = project_history_for_voters(messages)
    assert projected[0]["content"].startswith("[tool result ?]\n")


def test_tool_result_over_budget_head_tail_trimmed():
    """A 10k-char tool result is budgeted to head 2500 + marker + tail 1500."""
    head_marker = "HEAD" * 700  # >2500 chars of distinguishable head content
    tail_marker = "TAIL" * 500  # >1500 chars of distinguishable tail content
    middle = "M" * (10_000 - len(head_marker) - len(tail_marker))
    big_result = head_marker + middle + tail_marker
    assert len(big_result) == 10_000

    messages = [{"role": "tool", "tool_call_id": "c1", "content": big_result}]
    projected = project_history_for_voters(messages)
    content = projected[0]["content"]

    prefix = "[tool result c1]\n"
    assert content.startswith(prefix)
    body = content[len(prefix):]

    assert "[... " in body and " chars trimmed ...]" in body
    marker_start = body.index("[... ")
    marker_end = body.index(" chars trimmed ...]") + len(" chars trimmed ...]")
    head_part = body[: marker_start - 1]  # trailing \n before marker
    tail_part = body[marker_end + 1:]  # leading \n after marker

    assert len(head_part) == 2500
    assert head_part == big_result[:2500]
    assert len(tail_part) == 1500
    assert tail_part == big_result[-1500:]

    trimmed_n = int(body[marker_start + len("[... ") : body.index(" chars trimmed")])
    assert trimmed_n == 10_000 - 2500 - 1500


def test_tool_result_under_budget_not_trimmed():
    messages = [{"role": "tool", "tool_call_id": "c1", "content": "short result"}]
    projected = project_history_for_voters(messages)
    assert "trimmed" not in projected[0]["content"]


def test_tool_result_empty_content_becomes_empty_marker():
    messages = [{"role": "tool", "tool_call_id": "c1", "content": ""}]
    projected = project_history_for_voters(messages)
    # Body is empty, but the "[tool result c1]\n" prefix itself is non-empty
    # content, so the [empty] fallback does not apply here.
    assert projected[0]["content"] == "[tool result c1]\n"


# ---------------------------------------------------------------------------
# Opaque-field stripping
# ---------------------------------------------------------------------------


def test_opaque_fields_stripped_from_assistant_message():
    messages = [
        {
            "role": "assistant",
            "content": "final answer",
            "reasoning": "internal chain of thought",
            "reasoning_content": "more cot",
            "thinking": "hmm",
            "signature": "abc123",
            "cache_control": {"type": "ephemeral"},
            "refusal": None,
            "audio": {"id": "audio1"},
            "function_call": {"name": "old_style"},
        }
    ]
    projected = project_history_for_voters(messages)
    out = projected[0]
    for field in (
        "reasoning",
        "reasoning_content",
        "thinking",
        "signature",
        "cache_control",
        "refusal",
        "audio",
        "function_call",
    ):
        assert field not in out
    assert out["content"] == "final answer"


def test_opaque_field_triggers_non_identity_path_even_with_plain_content():
    """A message that otherwise looks 'plain' (string content, known role)
    but carries an opaque field must NOT be returned unchanged."""
    messages = [{"role": "assistant", "content": "hi", "cache_control": {"type": "ephemeral"}}]
    projected = project_history_for_voters(messages)
    assert projected[0] == {"role": "assistant", "content": "hi"}
    assert projected[0] is not messages[0]


# ---------------------------------------------------------------------------
# Per-message / append-only prefix property
# ---------------------------------------------------------------------------


def _mixed_history() -> list[dict]:
    return [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "do the thing"},
        {
            "role": "assistant",
            "content": "on it",
            "tool_calls": [
                {"function": {"name": "run", "arguments": '{"x": 1}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "tool output here"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "reasoning": "internal",
        },
        {"role": "user", "content": "thanks"},
    ]


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 6])
def test_projection_is_append_only_prefix_stable(n):
    """project(msgs[:n]) == project(msgs)[:n] for every prefix length n —
    the per-message contract stated in the module docstring."""
    full = _mixed_history()
    prefix = full[:n]
    assert project_history_for_voters(prefix) == project_history_for_voters(full)[:n]


# ---------------------------------------------------------------------------
# Unknown role degrades to user
# ---------------------------------------------------------------------------


def test_unknown_role_degrades_to_user_with_bracketed_note():
    messages = [{"role": "developer", "content": "some directive"}]
    projected = project_history_for_voters(messages)
    out = projected[0]
    assert out["role"] == "user"
    assert "developer" in out["content"]
    assert "some directive" in out["content"]
    assert out["content"].startswith("[")


def test_unknown_role_with_empty_content_becomes_empty_marker():
    messages = [{"role": "funky", "content": ""}]
    projected = project_history_for_voters(messages)
    assert projected[0] == {"role": "user", "content": "[role=funky] [empty]"}


def test_unknown_role_content_list_flattened_too():
    messages = [{"role": "weird", "content": [{"type": "text", "text": "hello"}]}]
    projected = project_history_for_voters(messages)
    assert "hello" in projected[0]["content"]
    assert projected[0]["role"] == "user"


# ---------------------------------------------------------------------------
# Consecutive-role safety (no merging)
# ---------------------------------------------------------------------------


def test_consecutive_tool_results_not_merged():
    messages = [
        {"role": "tool", "tool_call_id": "a", "content": "result a"},
        {"role": "tool", "tool_call_id": "b", "content": "result b"},
    ]
    projected = project_history_for_voters(messages)
    assert len(projected) == 2
    assert projected[0]["content"] == "[tool result a]\nresult a"
    assert projected[1]["content"] == "[tool result b]\nresult b"


def test_message_count_preserved_for_mixed_history():
    full = _mixed_history()
    assert len(project_history_for_voters(full)) == len(full)


# ---------------------------------------------------------------------------
# SessionRegistry
# ---------------------------------------------------------------------------


def _msgs(text="hello"):
    return [{"role": "user", "content": text}]


def test_resolve_uses_explicit_header_as_key():
    reg = SessionRegistry()
    record = reg.resolve("client-session-1", _msgs("anything"))
    assert record["key"] == "client-session-1"


def test_resolve_auto_key_is_stable_hash_of_first_user_message():
    reg = SessionRegistry()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the first user turn"},
    ]
    record = reg.resolve(None, messages)
    expected_digest = hashlib.sha256(
        "the first user turn".encode("utf-8", "replace")
    ).hexdigest()[:16]
    assert record["key"] == f"auto-{expected_digest}"


def test_resolve_auto_key_stable_across_turns():
    """Repeated resolves with the same growing history (first user message
    unchanged, per OpenAI-style resend-the-whole-conversation) land on the
    same session key."""
    reg = SessionRegistry()
    turn1 = [{"role": "user", "content": "start of conversation"}]
    turn2 = turn1 + [
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "follow up"},
    ]
    r1 = reg.resolve(None, turn1)
    r2 = reg.resolve(None, turn2)
    assert r1["key"] == r2["key"]


def test_resolve_increments_turn_count():
    reg = SessionRegistry()
    r1 = reg.resolve("sess-a", _msgs())
    assert r1["turns"] == 1
    r2 = reg.resolve("sess-a", _msgs())
    assert r2["turns"] == 2
    r3 = reg.resolve("sess-a", _msgs())
    assert r3["turns"] == 3


def test_cache_key_stable_and_formatted():
    reg = SessionRegistry()
    r1 = reg.resolve("sess-a", _msgs())
    r2 = reg.resolve("sess-a", _msgs())
    assert r1["cache_key"] == r2["cache_key"]
    assert r1["cache_key"].startswith("moa-sess-")
    expected = "moa-sess-" + hashlib.sha256(b"sess-a").hexdigest()[:24]
    assert r1["cache_key"] == expected


def test_cache_key_differs_across_sessions():
    reg = SessionRegistry()
    r1 = reg.resolve("sess-a", _msgs())
    r2 = reg.resolve("sess-b", _msgs())
    assert r1["cache_key"] != r2["cache_key"]
    assert r1["key"] != r2["key"]


def test_note_mode_updates_last_mode():
    reg = SessionRegistry()
    record = reg.resolve("sess-a", _msgs())
    assert record["last_mode"] is None
    reg.note_mode("sess-a", "tier0")
    updated = reg.resolve("sess-a", _msgs())
    assert updated["last_mode"] == "tier0"


def test_note_mode_on_unknown_key_is_a_noop():
    reg = SessionRegistry()
    # Should not raise even though "ghost" was never resolved.
    reg.note_mode("ghost", "tier1")


def test_ttl_expiry_evicts_stale_session(monkeypatch):
    import hermes_cli.proxy.moa_session as moa_session

    clock = {"t": 1000.0}
    monkeypatch.setattr(moa_session.time, "monotonic", lambda: clock["t"])

    reg = SessionRegistry(ttl_s=60)
    first = reg.resolve("sess-a", _msgs())
    assert first["turns"] == 1

    # Advance past the TTL without touching the session again.
    clock["t"] += 61
    # A resolve for a DIFFERENT session should sweep the expired one, and a
    # fresh resolve for "sess-a" must start a new record (turns resets to 1).
    reg.resolve("sess-b", _msgs())
    revived = reg.resolve("sess-a", _msgs())
    assert revived["turns"] == 1
    assert revived["created_at"] == clock["t"]


def test_ttl_not_expired_within_window(monkeypatch):
    import hermes_cli.proxy.moa_session as moa_session

    clock = {"t": 1000.0}
    monkeypatch.setattr(moa_session.time, "monotonic", lambda: clock["t"])

    reg = SessionRegistry(ttl_s=60)
    reg.resolve("sess-a", _msgs())
    clock["t"] += 30
    again = reg.resolve("sess-a", _msgs())
    assert again["turns"] == 2


def test_lru_eviction_bounds_session_count():
    reg = SessionRegistry(max_sessions=3)
    for i in range(5):
        reg.resolve(f"sess-{i}", _msgs())
    # Only the 3 most recently touched sessions survive.
    assert len(reg._sessions) == 3
    assert set(reg._sessions.keys()) == {"sess-2", "sess-3", "sess-4"}


def test_lru_touch_refreshes_recency():
    reg = SessionRegistry(max_sessions=2)
    reg.resolve("sess-a", _msgs())
    reg.resolve("sess-b", _msgs())
    # Touch sess-a again so it becomes MRU; sess-b is now the least recent.
    reg.resolve("sess-a", _msgs())
    reg.resolve("sess-c", _msgs())
    assert set(reg._sessions.keys()) == {"sess-a", "sess-c"}


def test_thread_safety_smoke_20_threads_x_50_resolves():
    reg = SessionRegistry(max_sessions=64)
    errors: list[BaseException] = []

    def worker(idx: int):
        try:
            for turn in range(50):
                key = f"thread-{idx % 8}"  # some overlap across threads
                reg.resolve(key, _msgs(f"turn {turn}"))
                reg.note_mode(key, "tier0" if turn % 2 == 0 else "tier1")
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(reg._sessions) <= 64


def test_sanitize_acting_messages_strips_response_only_fields():
    """The acting-lane sanitizer removes reasoning_content /
    provider_specific_fields / reasoning / refusal (response-only annotations
    a strict endpoint 400s on replay) while PRESERVING tool_calls, tool ids,
    content, role — the acting lane is mid an agentic loop. Clean messages
    pass through as the same object (zero-copy)."""
    from hermes_cli.proxy.moa_session import sanitize_acting_messages

    dirty = {
        "role": "assistant",
        "content": "ok",
        "reasoning_content": "internal chain of thought",
        "provider_specific_fields": {"x": 1},
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    }
    clean = {"role": "user", "content": "hi"}
    out = sanitize_acting_messages([dirty, clean])
    assert "reasoning_content" not in out[0]
    assert "provider_specific_fields" not in out[0]
    assert out[0]["tool_calls"] == dirty["tool_calls"]  # preserved
    assert out[0]["content"] == "ok"
    assert out[1] is clean  # untouched, same object
    # signature/thinking are NOT stripped (native Anthropic tool continuation
    # needs them) — only the four response-only annotations go.
    signed = {"role": "assistant", "content": "x", "signature": "sig", "thinking": "t"}
    out2 = sanitize_acting_messages([signed])
    assert out2[0]["signature"] == "sig" and out2[0]["thinking"] == "t"
