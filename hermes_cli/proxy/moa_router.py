"""``moa:auto`` request routing for the MoA OpenAI endpoint.

A fast classifier assigns each incoming request either to one of the
configured MoA presets (those carrying a ``route.description``) or to the
special SELF class — trivial requests answered directly by a fast model with
no reference fan-out at all, which is where most of the latency/cost win
lives. Design sketch: docs/plans/moa-proxy-backlog.md item 1.

The classifier slot is any Hermes model slot (``{provider, model}``) resolved
through the same ``_slot_runtime`` chokepoint as every other MoA slot; the
reference deployment uses Cerebras ``gemma-4-31b`` (custom provider,
wafer-speed) with any fast OpenRouter host as a drop-in substitute.

Failure containment: classification is best-effort with a hard timeout — any
error, timeout, or unparseable label falls back to ``router.default``. A
routing failure must never fail the request.

Sticky sessions: one client conversation keeps its first routing decision
(keyed by ``x-hermes-session-id`` when supplied, else a hash of the first
user message) so multi-iteration tool loops don't re-classify — and can't
flip presets — mid-conversation.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

SELF_CLASS = "self"

# Sticky routing decisions per conversation. Server-wide, like the reference
# cache: bounded LRU, thread-safe, safe to clear (worst case: one extra
# classifier call per conversation).
_STICKY_MAX = 512
_sticky: "OrderedDict[str, RouteDecision]" = OrderedDict()
_sticky_lock = threading.Lock()

# Tail of the last user message shown to the classifier. Requests are
# classified by what they ASK, not their full history; the digest keeps the
# classifier call cheap (fast host + tiny prompt = low added latency).
_CLASSIFIER_INPUT_CHARS = 2000
_CLASSIFIER_CONTEXT_CHARS = 400


@dataclass
class RouteDecision:
    """Outcome of routing one request."""

    preset_name: str  # concrete preset, or SELF_CLASS
    is_self: bool
    method: str  # "classified" | "sticky" | "fallback"
    reason: str  # short human-readable note for the reasoning delta
    classifier_ms: Optional[int] = None

    def as_trace(self) -> dict[str, Any]:
        return {
            "requested": "auto",
            "routed_preset": self.preset_name,
            "method": self.method,
            "reason": self.reason,
            "classifier_ms": self.classifier_ms,
        }


def is_auto_model(model_field: Any) -> bool:
    """True when the request's ``model`` selects the routed virtual model."""
    raw = str(model_field or "").strip().lower()
    for prefix in ("moa:", "moa/"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    return raw == "auto"


def sticky_key(messages: list, session_id: str | None) -> str:
    """Conversation identity for sticky routing.

    The client's ``x-hermes-session-id`` header wins. Otherwise the FIRST user
    message identifies the conversation — it is the one prefix every request
    of a tool loop shares (later requests append tool results after it).
    """
    if session_id:
        return f"sid:{session_id}"
    first_user = next(
        (
            m.get("content")
            for m in (messages or [])
            if isinstance(m, dict)
            and m.get("role") == "user"
            and isinstance(m.get("content"), str)
        ),
        "",
    )
    return "msg:" + hashlib.sha256(first_user.encode("utf-8", "replace")).hexdigest()[:24]


def sticky_get(key: str) -> Optional[RouteDecision]:
    with _sticky_lock:
        decision = _sticky.get(key)
        if decision is not None:
            _sticky.move_to_end(key)
        return decision


def sticky_put(key: str, decision: RouteDecision) -> None:
    with _sticky_lock:
        _sticky[key] = decision
        _sticky.move_to_end(key)
        while len(_sticky) > _STICKY_MAX:
            _sticky.popitem(last=False)


def sticky_clear() -> None:
    """Test hook / config-reload hook."""
    with _sticky_lock:
        _sticky.clear()


def _failure_signal(messages: list, patterns: list[str]) -> bool:
    """True when the newest tool/user feedback carries a failure marker.

    Only the LATEST feedback message counts — a failure earlier in the
    conversation that was already fixed must not re-trigger escalation.
    Assistant messages never count (the model describing a failure it is
    fixing is not an observed failure).
    """
    for m in reversed(messages or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "assistant":
            continue
        if role in {"tool", "user"}:
            content = m.get("content")
            if not isinstance(content, str) or not content.strip():
                return False
            return any(p in content for p in patterns)
    return False


def _classifier_prompt(
    presets: dict[str, Any], router: dict[str, Any], messages: list
) -> list[dict[str, str]]:
    classes = []
    for name in router.get("routable_presets") or []:
        description = ((presets.get(name) or {}).get("route") or {}).get("description", "")
        classes.append(f"- {name}: {description}")
    if router.get("self_answer"):
        classes.append(
            f"- {SELF_CLASS}: trivial requests a small fast model handles alone — "
            "greetings, small talk, thanks, simple lookups or single-fact "
            "questions, trivial formatting or rephrasing"
        )

    last_user = ""
    prior_context = ""
    for m in reversed(messages or []):
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            continue
        if m.get("role") == "user" and not last_user:
            last_user = m["content"]
        elif not prior_context and m.get("role") in {"user", "assistant"}:
            prior_context = m["content"]
        if last_user and prior_context:
            break

    parts = []
    if prior_context:
        parts.append(
            "Earlier context (tail): "
            + prior_context[-_CLASSIFIER_CONTEXT_CHARS:]
        )
    parts.append("Request: " + last_user[-_CLASSIFIER_INPUT_CHARS:])

    system = (
        "You route requests to the best handler class. Classes:\n"
        + "\n".join(classes)
        + "\n\nReply with EXACTLY one class name from the list — one word, "
        "no punctuation, no explanation."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _parse_label(text: str, router: dict[str, Any]) -> Optional[str]:
    """Map classifier output to a known class, tolerantly."""
    valid = set(router.get("routable_presets") or [])
    if router.get("self_answer"):
        valid.add(SELF_CLASS)
    cleaned = str(text or "").strip().strip("\"'`.,:;!").lower()
    if not cleaned:
        return None
    by_lower = {v.lower(): v for v in valid}
    if cleaned in by_lower:
        return by_lower[cleaned]
    # First word, then substring rescue ("the coding preset" -> coding).
    first = cleaned.split()[0].strip("\"'`.,:;!")
    if first in by_lower:
        return by_lower[first]
    hits = [v for low, v in by_lower.items() if low in cleaned]
    if len(hits) == 1:
        return hits[0]
    return None


def _classify_sync(
    presets: dict[str, Any], router: dict[str, Any], messages: list
) -> Optional[str]:
    """One small completion → class label (or None). Runs in a worker thread."""
    # Same call seam as the rest of the MoA stack: tests patch
    # agent.moa_loop.call_llm once and cover routing too.
    from agent import moa_loop

    response = moa_loop.call_llm(
        task="moa_router",
        messages=_classifier_prompt(presets, router, messages),
        temperature=0.0,
        max_tokens=10,
        **moa_loop._slot_runtime(router["classifier"]),
    )
    return _parse_label(moa_loop._extract_text(response), router)


async def route_request(
    moa_cfg: dict[str, Any],
    messages: list,
    *,
    session_id: str | None = None,
) -> RouteDecision:
    """Resolve ``moa:auto`` to a concrete preset (or SELF), never raising.

    ``moa_cfg`` is the normalized MoA config (``normalize_moa_config``).
    """
    router = moa_cfg.get("router") or {}
    presets = moa_cfg.get("presets") or {}
    default = router.get("default") or moa_cfg.get("default_preset") or ""

    key = sticky_key(messages, session_id)
    previous = sticky_get(key)
    if previous is not None:
        # Failure-gated escalation: a conversation stuck on a cheaper lane
        # whose latest tool/user feedback shows a failure signal is re-routed
        # to the configured escalation preset — strong-solo-first, frontier
        # only on observed failure. The escalated decision becomes the new
        # sticky state, so a conversation escalates at most once.
        escalation = router.get("escalation")
        if (
            escalation
            and previous.preset_name != escalation["preset"]
            and _failure_signal(messages, escalation["on_patterns"])
        ):
            decision = RouteDecision(
                preset_name=escalation["preset"],
                is_self=False,
                method="escalated",
                reason=(
                    f"failure signal after '{previous.preset_name}' — "
                    f"escalating to '{escalation['preset']}'"
                ),
                classifier_ms=None,
            )
            sticky_put(key, decision)
            return decision
        decision = RouteDecision(
            preset_name=previous.preset_name,
            is_self=previous.is_self,
            method="sticky",
            reason=f"reusing this conversation's routing ({previous.preset_name})",
            classifier_ms=None,
        )
        return decision

    started = time.time()
    label: Optional[str] = None
    failure = ""
    try:
        label = await asyncio.wait_for(
            asyncio.to_thread(_classify_sync, presets, router, messages),
            timeout=float(router.get("timeout_s") or 8.0),
        )
        if label is None:
            failure = "unrecognized classifier label"
    except asyncio.TimeoutError:
        failure = f"classifier timeout ({router.get('timeout_s')}s)"
    except Exception as exc:
        failure = f"classifier error: {exc}"
        logger.warning("moa:auto classifier failed: %s", exc)
    elapsed_ms = int((time.time() - started) * 1000)

    if label is None:
        decision = RouteDecision(
            preset_name=default,
            is_self=False,
            method="fallback",
            reason=f"{failure or 'no label'} — using default preset '{default}'",
            classifier_ms=elapsed_ms,
        )
    else:
        decision = RouteDecision(
            preset_name=label,
            is_self=label == SELF_CLASS,
            method="classified",
            reason=f"classified in {elapsed_ms}ms",
            classifier_ms=elapsed_ms,
        )
    sticky_put(key, decision)
    return decision


def self_answer_preset(router: dict[str, Any]) -> dict[str, Any]:
    """Pseudo-preset for the SELF class: fast model acting alone.

    Shaped like a normalized preset with the fan-out disabled, so the server
    handles it through the exact solo code path used by ``enabled: false``
    presets — no special-casing downstream.
    """
    slot = router.get("self_answer_model") or router.get("classifier") or {}
    return {
        "enabled": False,
        "reference_models": [],
        "aggregator": dict(slot),
        "route": None,
        "reference_temperature": 0.6,
        "aggregator_temperature": 0.4,
        "max_tokens": 4096,
        "reference_max_tokens": None,
    }


__all__ = [
    "SELF_CLASS",
    "RouteDecision",
    "is_auto_model",
    "route_request",
    "self_answer_preset",
    "sticky_clear",
    "sticky_key",
]
