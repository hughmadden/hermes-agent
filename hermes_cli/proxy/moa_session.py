"""Provider-agnostic history projection + a session registry for the MoA
proxy (`hermes moa serve`, ``hermes_cli/proxy/moa_server.py``).

**Why this exists.** A real agent session (see ``agent/moa_loop.py``) carries
``tool_calls``/``tool``-role messages and provider-opaque fields — reasoning
traces, Anthropic-style content-block lists, cache_control markers,
signatures — accumulated by whichever model produced each turn. Cascade MoA
fans that history out to VOTER models that may be on a completely different
provider (an Anthropic-shaped ``tool_calls`` array, or a ``reasoning``
field another vendor never emitted, routinely 400s a strict endpoint or just
confuses a model that never asked the question). ``moa_loop._reference_messages``
already solves this for the acting agent's OWN advisory fan-out; this module
solves the same problem one layer down, for voter models that need a
plain-text-only projection of a client-supplied OpenAI-format conversation,
not just the acting agent's live turn.

**The projection contract.** ``project_history_for_voters`` is PURE and
PER-MESSAGE: message *i*'s projection is a function of message *i* alone,
never of its neighbours. Concretely, for any conversation ``msgs`` and any
prefix length ``n``:

    project_history_for_voters(msgs[:n]) == project_history_for_voters(msgs)[:n]

This is deliberate, not incidental. A live agent session's history is
append-only — each turn only ever adds messages, never edits earlier ones —
so a per-message projection is *also* append-only, which means the plain-text
projection of turn N is always a prefix-extension of the projection of turn
N-1. That in turn preserves upstream prompt caches (OpenAI-style
``prompt_cache_key``, Anthropic-style implicit prefix caching) on the voter
side: a merge/summarize/cross-message transform would invalidate the cached
prefix on every single turn and burn the whole point of caching. Nothing in
this module looks at message ``i-1`` or ``i+1`` while projecting message
``i``; the only "memory" is what ``SessionRegistry`` tracks out-of-band about
the session as a whole (turn count, cache key, last mode).

Plain messages already safe for any provider (role in {system, user,
assistant}, plain string content, no opaque fields) project to themselves
UNCHANGED — same dict object even — so a plain history round-trips through
this function byte-identically and pays zero projection tax.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

# Fields that are provider-opaque (reasoning traces, cache hints, signatures,
# alternate tool-call encodings) and must never reach a voter model that
# didn't produce them — carrying another vendor's cache_control block or a
# thinking-signature is meaningless at best and a strict-provider 400 at
# worst. ``name`` is deliberately NOT in this set: it's a plain, portable
# identifier every provider already understands.
_OPAQUE_FIELDS = (
    "reasoning",
    "reasoning_content",
    "thinking",
    "signature",
    "cache_control",
    "refusal",
    "audio",
    "function_call",
)

_PLAIN_ROLES = ("system", "user", "assistant")

# Per-call argument cap inside a rendered ``[tool call ...]`` block — a
# single pathological call (e.g. a giant inlined file write) must not blow
# out the projected turn's size budget on its own.
_TOOL_CALL_ARGS_CAP = 500

# Same head+tail budget as ``agent.moa_loop._REFERENCE_TOOL_RESULT_BUDGET``
# (4000 chars): a tool result folded into the voter-facing history keeps the
# start (what the call was doing) and the end (its outcome) without paying
# for the whole payload in every voter's context.
_TOOL_RESULT_BUDGET = 4000
_TOOL_RESULT_HEAD = 2500
_TOOL_RESULT_TAIL = 1500


def _is_plain_message(msg: dict) -> bool:
    """True iff ``msg`` needs no projection work at all.

    A message qualifies only when its role is a plain conversational role,
    its content is already a plain string, it carries no ``tool_calls``, and
    none of the opaque fields are present. Returning the identity dict for
    these keeps a plain (non-agentic) history projecting to itself
    byte-identically, with zero copying.
    """
    if msg.get("role") not in _PLAIN_ROLES:
        return False
    if not isinstance(msg.get("content"), str):
        return False
    if msg.get("tool_calls"):
        return False
    return not any(field in msg for field in _OPAQUE_FIELDS)


def _compact_json(value: Any) -> str:
    """Compact string form of a tool-call ``arguments`` payload.

    Arguments arrive as either an already-serialized JSON string (the OpenAI
    wire format) or, occasionally, a pre-parsed dict/list. Either way we want
    one compact single-line string; anything that fails to serialize falls
    back to ``str(value)`` rather than raising — a malformed call is still
    worth showing a voter, just not worth crashing the projection over.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            text = str(value)
    if len(text) > _TOOL_CALL_ARGS_CAP:
        text = text[:_TOOL_CALL_ARGS_CAP] + "...[truncated]"
    return text


def _render_tool_call_block(tool_calls: Any) -> str:
    """Render an assistant turn's ``tool_calls`` as ``[tool call name(args)]``
    lines, one per call, each argument string capped independently (see
    ``_TOOL_CALL_ARGS_CAP``). Mirrors the shape of
    ``agent.moa_loop._render_tool_calls`` but with per-call capping added,
    since this projection has no separate whole-message budget pass."""
    lines: list[str] = []
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        name = fn.get("name") or tc.get("name") or "tool"
        args_text = _compact_json(fn.get("arguments"))
        lines.append(f"[tool call {name}({args_text})]")
    return "\n".join(lines)


def _flatten_content(content: Any) -> str:
    """Flatten string OR content-block-list content into plain text.

    A content list (``[{"type": "text", "text": ...}, {"type": "image_url",
    ...}, ...]``) has its text parts joined with ``"\\n"``; each non-text
    part becomes a short bracketed placeholder (``[non-text content:
    image_url]``) so the voter at least knows something was there, without
    receiving a payload (raw image bytes/URLs, provider-specific blocks) it
    can't use and another provider might reject outright.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(str(block.get("text") or ""))
                else:
                    parts.append(f"[non-text content: {btype or 'unknown'}]")
            else:
                parts.append(f"[non-text content: {type(block).__name__}]")
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _truncate_tool_result(text: str) -> str:
    """Head+tail budget a tool result to ``_TOOL_RESULT_BUDGET`` chars total.

    Same shape as ``agent.moa_loop._truncate_tool_result`` (which this
    mirrors): keep the first ``_TOOL_RESULT_HEAD`` chars and the last
    ``_TOOL_RESULT_TAIL`` chars with a ``[... N chars trimmed ...]`` marker
    between them when the text is over budget, so a voter sees both how the
    tool call started and how it ended.
    """
    if not text or len(text) <= _TOOL_RESULT_BUDGET:
        return text
    trimmed = len(text) - _TOOL_RESULT_HEAD - _TOOL_RESULT_TAIL
    return (
        f"{text[:_TOOL_RESULT_HEAD]}\n"
        f"[... {trimmed} chars trimmed ...]\n"
        f"{text[-_TOOL_RESULT_TAIL:]}"
    )


def _project_one(msg: dict) -> dict:
    """Project a single message. See module docstring for the per-message
    contract this function must uphold."""
    if _is_plain_message(msg):
        return msg

    role = msg.get("role")
    content = msg.get("content")

    if role == "tool":
        call_id = msg.get("tool_call_id") or "?"
        body = _truncate_tool_result(_flatten_content(content))
        text = f"[tool result {call_id}]\n{body}"
        return {"role": "user", "content": text or "[empty]"}

    if role == "assistant":
        text = _flatten_content(content).strip()
        calls_text = _render_tool_call_block(msg.get("tool_calls"))
        parts = [p for p in (text, calls_text) if p]
        merged = "\n".join(parts)
        out = {"role": "assistant", "content": merged or "[empty]"}
        if "name" in msg:
            out["name"] = msg["name"]
        return out

    if role in ("system", "user"):
        text = _flatten_content(content)
        out = {"role": role, "content": text or "[empty]"}
        if "name" in msg:
            out["name"] = msg["name"]
        return out

    # Unknown role: degrade to a user turn with a bracketed note so the
    # voter still gets the content instead of the request silently dropping
    # a turn (which would desync turn-taking for the rest of the history).
    text = _flatten_content(content) or "[empty]"
    return {"role": "user", "content": f"[role={role}] {text}"}


def project_history_for_voters(messages: list[dict]) -> list[dict]:
    """Project an OpenAI-format conversation into plain-text-only messages
    safe to send to any provider's voter model.

    PURE and PER-MESSAGE — see the module docstring for the append-only /
    prefix-cache-preserving contract this guarantees. Each output message's
    role is one of ``system``/``user``/``assistant`` (unknown/``tool`` roles
    degrade to ``user``); content is always a plain string; opaque fields
    (``reasoning``, ``cache_control``, ``signature``, ...) and ``tool_calls``
    never appear in the output. Messages are never merged or reordered —
    consecutive same-role output messages are left as-is, since collapsing
    them would violate the per-message contract above.
    """
    return [_project_one(dict(m)) if not _is_plain_message(m) else m for m in messages]


def window_history(
    messages: list[dict], budget_tokens: int | None
) -> tuple[list[dict], dict]:
    """Bound a (projected) history to a recency window of ~``budget_tokens``.

    Measured motivation (2026-07-08 session-sim run): fanning a real
    session's FULL context to every cascade voter multiplies the per-turn
    token bill by the voter count — a 10k-token session tripped the
    Cerebras per-minute token quota after ONE turn (4 voters + gate +
    aggregator ≈ 5-6x the session length per user turn), and provider-side
    prefix caching does not help because the quota counts cached tokens
    too (measured: 30,848/30,907 tokens cached, full amount billed against
    TPM). Voters answer the CURRENT question; the acting lanes (solo /
    tier-1 aggregator / tier-2 escalate) keep the full verbatim transcript.
    This is the cascade's existing division of labor: tier-0 serves
    self-contained turns, and a question that genuinely needs deep history
    produces voter discord, which escalates to a full-context tier anyway.

    Keeps a leading system message (if any) and the LONGEST suffix of the
    remaining messages that fits ``budget_tokens * 4`` characters (the
    stdlib-only ~4 chars/token heuristic every probe script here uses); the
    newest message is always kept even when oversized. When anything is
    trimmed, ONE deterministic marker message notes how many turns the
    voter is not seeing, so a voter never mistakes the window for the whole
    session. ``budget_tokens`` None/<=0 returns ``(messages, {})``
    unchanged.

    Windowing deliberately trades voter-side prefix-cache reuse (the window
    slides, so voter prefixes change every turn) for a bounded fan-out
    bill: small uncached voter prompts cost less than giant cached ones
    under both quota and price. Returns ``(windowed, stats)`` where stats
    is ``{}`` when nothing was trimmed, else ``{"kept": n, "trimmed": m}``.
    """
    if not budget_tokens or budget_tokens <= 0 or not messages:
        return messages, {}
    budget_chars = budget_tokens * 4

    head: list[dict] = []
    body = messages
    if messages[0].get("role") == "system":
        head = [messages[0]]
        body = messages[1:]
        budget_chars -= len(str(messages[0].get("content") or ""))

    used = 0
    kept: list[dict] = []
    for msg in reversed(body):
        size = len(str(msg.get("content") or ""))
        if kept and used + size > budget_chars:
            break
        kept.append(msg)
        used += size
    kept.reverse()

    trimmed = len(body) - len(kept)
    if trimmed <= 0:
        return messages, {}
    marker = {
        "role": "user",
        "content": (
            f"[voter context window: the {trimmed} earliest conversation "
            "messages are not shown here; the acting agent retains the "
            "full history. Answer the latest request from the visible "
            "context.]"
        ),
    }
    return [*head, marker, *kept], {"kept": len(kept), "trimmed": trimmed}


# ---------------------------------------------------------------------------
# SessionRegistry — thread-safe TTL+LRU session bookkeeping
# ---------------------------------------------------------------------------

_DEFAULT_TTL_S = 6 * 3600


class SessionRegistry:
    """Thread-safe TTL+LRU registry of proxy sessions.

    Tracks just enough per-session state for the MoA proxy to hand voters a
    stable ``prompt_cache_key`` and remember which cascade mode last served a
    session — no I/O, no background threads; eviction happens inline on
    ``resolve()``.

    Keying mirrors ``moa_server._save_proxy_trace``: the client-supplied
    session header wins when present, else a stable hash of the FIRST user
    message's text content (``"auto-" + sha256(...).hexdigest()[:16]``) so a
    whole client conversation — which keeps resending its first turn on every
    request, per the OpenAI chat-completions convention — lands on the same
    session key without the client having to opt in to a header.
    """

    def __init__(self, max_sessions: int = 512, ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._max_sessions = max_sessions
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        # Insertion/access order == LRU order (dict preserves insertion order;
        # resolve() re-inserts on touch to bump recency).
        self._sessions: dict[str, dict] = {}

    @staticmethod
    def _first_user_text(messages: list[dict]) -> str:
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"]
        return ""

    @staticmethod
    def _session_key(session_header: str | None, messages: list[dict]) -> str:
        if session_header:
            return session_header
        first_user = SessionRegistry._first_user_text(messages)
        digest = hashlib.sha256(first_user.encode("utf-8", "replace")).hexdigest()[:16]
        return f"auto-{digest}"

    @staticmethod
    def _cache_key_for(key: str) -> str:
        # Distinct from `key` itself: the raw key may be a client-controlled
        # header (arbitrary length/characters) which is not necessarily a
        # safe/short value to hand upstream as a cache_key.
        return "moa-sess-" + hashlib.sha256(key.encode("utf-8", "replace")).hexdigest()[:24]

    def _evict_expired_locked(self, now: float) -> None:
        expired = [
            k for k, rec in self._sessions.items() if now - rec["last_seen"] > self._ttl_s
        ]
        for k in expired:
            del self._sessions[k]

    def _evict_lru_locked(self) -> None:
        while len(self._sessions) > self._max_sessions:
            oldest_key = next(iter(self._sessions))
            del self._sessions[oldest_key]

    def resolve(self, session_header: str | None, messages: list[dict]) -> dict:
        """Return the session record for this request, creating it if new.

        Increments ``turns`` and refreshes ``last_seen`` on every call
        (including the first, where ``turns`` becomes 1). Expired and
        LRU-overflow entries are swept inline before the record is
        looked up/created.
        """
        key = self._session_key(session_header, messages)
        now = time.monotonic()
        with self._lock:
            self._evict_expired_locked(now)
            record = self._sessions.pop(key, None)
            if record is None:
                record = {
                    "key": key,
                    "cache_key": self._cache_key_for(key),
                    "turns": 0,
                    "last_mode": None,
                    # Addendum v1.6 §B (advisor lane): the latest pending
                    # advisor note, or None. Set by `note_advisor` (the
                    # async advisor worker, after a turn returns) and
                    # consumed exactly once by the acting call that reads it
                    # on the NEXT turn (see moa_server._apply_advisor_note).
                    "advisor_note": None,
                    "created_at": now,
                    "last_seen": now,
                }
            record["turns"] += 1
            record["last_seen"] = now
            # Re-insert to move this key to the MRU end.
            self._sessions[key] = record
            self._evict_lru_locked()
            return dict(record)

    def note_mode(self, key: str, mode: str) -> None:
        """Record the cascade mode last served for session ``key``, if it is
        still tracked (a race with eviction is a harmless no-op)."""
        with self._lock:
            record = self._sessions.get(key)
            if record is not None:
                record["last_mode"] = mode

    def note_advisor(self, key: str, note: dict | None) -> None:
        """Store the latest advisor note for session ``key`` (addendum v1.6
        §B), or clear it with ``note=None``. ``note`` is a ``{"kind":
        "concern"|"blocker", "text": str}`` dict — the shape
        `moa_server._parse_advisor_reply` produces. Like `note_mode`, a race
        with eviction is a harmless no-op: an evicted session has nothing
        left to advise."""
        with self._lock:
            record = self._sessions.get(key)
            if record is not None:
                record["advisor_note"] = note


__all__ = ["project_history_for_voters", "window_history", "SessionRegistry"]
