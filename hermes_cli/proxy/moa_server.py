"""OpenAI-compatible HTTP endpoint for Hermes Mixture-of-Agents presets.

``hermes moa serve`` exposes every configured MoA preset as a model on a local
``/v1/chat/completions`` surface, so *other* agents (OpenCode, OpenClaw, other
Hermes instances, any OpenAI-SDK client) can use Hermes MoA as their upstream
model. The critical inversion versus the in-process MoA loop: here the CALLING
CLIENT owns tool execution and turn termination — the endpoint never runs
tools. It runs the reference fan-out, injects the synthesized reference
context, forwards the client's ``tools`` to the acting aggregator, and streams
the aggregator's ``tool_calls`` back for the client to execute. On the
client's next request (with the tool results appended) the references judge
the advanced state again, exactly like the internal per-tool-iteration MoA
loop but across the wire.

Model naming: the request's ``model`` field selects a preset — ``moa:review``,
``moa/review``, or bare ``review``; empty/``default``/``moa`` selects the
configured default preset. ``GET /v1/models`` lists the presets.

Streaming: reference-model output is advisory, so it streams to the client as
*reasoning* deltas (``delta.reasoning`` and ``delta.reasoning_content``, both
set for DeepSeek-style and OpenRouter-style consumers) with a labelled header
per reference, followed by the aggregator's own reasoning deltas, then the
aggregator's acting ``content`` / ``tool_calls`` deltas verbatim. A client
that renders reasoning shows the whole MoA process live; a client that
ignores unknown delta fields still gets a clean final answer.

Reference calls stream live when the client streams: reference 1's deltas are
forwarded as they arrive while later references buffer in their own queues,
each flushed in order as its predecessor finishes — live first-token latency
without interleaving unlabelled text from concurrent models.

Usage: the top-level ``usage`` object is the SUM across every upstream call
(all references + aggregator) — that is the true cost of the request, which
is what a billing-aware client wants. The per-slot split lives under
``usage.moa`` so nothing is hidden.

Reuses the canonical MoA turn machinery from ``agent.moa_loop`` (advisory
view construction, slot→runtime resolution, reference fan-out, guidance
format) so a proxied turn behaves byte-for-byte like an in-process ``/moa``
turn; the private-name imports are deliberate shared internals, not a fork.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import threading
import time
import uuid
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via cmd_moa_serve guard
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from agent.auxiliary_client import call_llm
from agent.moa_loop import (
    _MAX_REFERENCE_WORKERS,
    _maybe_apply_moa_cache_control,
    _REFERENCE_SYSTEM_PROMPT,
    _RefAccounting,
    _attach_reference_guidance,
    _reference_messages,
    _run_reference,
    _run_references_parallel,
    _slot_label,
    _slot_runtime,
    aggregation_skill_block,
)
from hermes_cli.proxy import moa_cascade
from hermes_cli.proxy.moa_session import (
    SessionRegistry,
    project_history_for_voters,
    window_history,
)
from hermes_cli.proxy.moa_cascade import (
    agrees,
    consensus,
    extract_candidate,
    extract_python_block,
    is_boilerplate,
    normalize_candidate,
)

logger = logging.getLogger(__name__)

# Session bookkeeping for real-world serving: stable per-session cache keys
# for OpenAI-family acting lanes + last-served-mode observability. TTL+LRU,
# in-memory only — the proxy stays restart-safe (a lost registry only means
# a fresh cache key / turn counter, never a wrong answer).
_session_registry = SessionRegistry()

DEFAULT_MOA_HOST = "127.0.0.1"
DEFAULT_MOA_PORT = 8646  # existing credential proxy default is 8645

# Client request-body fields forwarded to the aggregator via extra_body.
# call_llm has first-class kwargs only for messages/temperature/max_tokens/
# tools/stream; everything else OpenAI-compatible rides through extra_body so
# the aggregator honors the client's sampling and tool-choice intent.
_PASSTHROUGH_FIELDS = (
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "top_p",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "logit_bias",
    "user",
)

# Server-wide LRU of reference fan-outs keyed by advisory-view signature.
# Same semantics as MoAChatCompletions' per-instance cache: a repeated request
# over identical task state (client retry, duplicated turn) reuses the advice
# instead of re-billing the whole fan-out. New state (new user turn, new tool
# result) changes the advisory view, so it is always a MISS and re-runs.
_REF_CACHE_MAX = 16
_ref_cache: "OrderedDict[tuple, list[tuple[str, str, Any]]]" = OrderedDict()
_ref_cache_lock = threading.Lock()


def _ref_cache_get(key: tuple) -> Optional[list]:
    with _ref_cache_lock:
        outputs = _ref_cache.get(key)
        if outputs is not None:
            _ref_cache.move_to_end(key)
        return list(outputs) if outputs is not None else None


def _ref_cache_put(key: tuple, outputs: list) -> None:
    with _ref_cache_lock:
        _ref_cache[key] = list(outputs)
        _ref_cache.move_to_end(key)
        while len(_ref_cache) > _REF_CACHE_MAX:
            _ref_cache.popitem(last=False)


def _advisory_signature(preset_name: str, ref_messages: list, reference_models: list) -> tuple:
    sig = hashlib.sha256(
        "\u0000".join(f"{m.get('role')}:{m.get('content')}" for m in ref_messages).encode(
            "utf-8", "replace"
        )
    ).hexdigest()
    return (preset_name, sig, tuple(_slot_label(s) for s in reference_models))


# ---------------------------------------------------------------------------
# Request parsing / preset resolution
# ---------------------------------------------------------------------------


def resolve_preset_name(model_field: Any, config: dict) -> str:
    """Map an OpenAI ``model`` field to a MoA preset name.

    Accepts ``moa:<preset>``, ``moa/<preset>``, a bare preset name, and the
    aliases ``""``/``"moa"``/``"default"`` for the configured default preset.
    Raises KeyError (with the requested name) when no such preset exists.
    """
    from hermes_cli.moa_config import normalize_moa_config

    cfg = normalize_moa_config((config or {}).get("moa") or {})
    raw = str(model_field or "").strip()
    for prefix in ("moa:", "moa/"):
        if raw.lower().startswith(prefix):
            raw = raw[len(prefix):].strip()
            break
    if raw.lower() in {"", "moa", "default"}:
        return cfg["default_preset"]
    if raw in cfg["presets"]:
        return raw
    raise KeyError(raw)


def _json_error(status: int, message: str, *, err_type: str = "invalid_request_error", code: str | None = None):
    body = {"error": {"message": message, "type": err_type, "code": code or err_type}}
    return web.json_response(body, status=status)


def _usage_to_openai(usage: Any) -> dict[str, int]:
    """CanonicalUsage → OpenAI usage dict.

    Uses the ``prompt_tokens`` property (input + cache read/write) so cached
    prefixes still count toward the billed prompt figure, matching how the
    upstream provider reported them.
    """
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "output_tokens", 0) or 0)
    out = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    # Surface upstream cache activity (OpenAI wire shape) so real-world
    # sessions can SEE whether provider-side prompt caching is working —
    # cached_tokens is the sum of every slot's cache reads this turn.
    # Emitted unconditionally: a hard 0 on a 100k-token warm turn is the
    # signal that a lane is not caching, and hiding it would make that
    # indistinguishable from "not measured".
    cache_read = int(getattr(usage, "cache_read_tokens", 0) or 0)
    cache_write = int(getattr(usage, "cache_write_tokens", 0) or 0)
    out["prompt_tokens_details"] = {"cached_tokens": cache_read}
    if cache_write:
        out["cache_write_tokens"] = cache_write
    return out


def _normalize_chunk_usage(raw_usage: Any, runtime: dict) -> Any:
    from agent.usage_pricing import CanonicalUsage, normalize_usage

    if not raw_usage:
        return CanonicalUsage()
    try:
        return normalize_usage(
            raw_usage, provider=runtime.get("provider"), api_mode=runtime.get("api_mode")
        )
    except Exception:  # pragma: no cover - defensive
        return CanonicalUsage()


def _reference_guidance(preset_name: str, aggregator: dict, reference_outputs: list) -> str:
    """Same guidance block MoAChatCompletions injects, with one addition: the
    aggregator is told the CLIENT executes any tools it calls."""
    joined = "\n\n".join(
        f"Reference {idx} — {label}:\n{text}"
        for idx, (label, text, _acct) in enumerate(reference_outputs, start=1)
    )
    return (
        "[Mixture of Agents reference context]\n"
        f"Preset: {preset_name}\n"
        f"Aggregator/acting model: {_slot_label(aggregator)}\n"
        f"References: {', '.join(label for label, _, _ in reference_outputs)}\n\n"
        "Use the reference responses below as private context. You are the "
        "aggregator and acting model: answer the user directly or call tools "
        "as needed. Any tool you call is executed by the calling client, which "
        "will return the result in the next request."
        f"{aggregation_skill_block(preset_name)}\n\n"
        f"{joined}"
    )


def _cascade_vote_counts(candidates: list[str | None]) -> int:
    """Size of the largest normalized-candidate group (0 if none extracted)."""
    from collections import Counter

    counts = Counter(normalize_candidate(c) for c in candidates if c is not None)
    return max(counts.values()) if counts else 0


def _cascade_tool_names(tools: Any) -> list[str]:
    """Function names from a client's OpenAI-style ``tools`` list — used to
    build the addendum v1.5 VOTER GATE's tool-awareness system line.
    Malformed entries are skipped defensively (a client tool list is
    untrusted input)."""
    names = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):
            names.append(str(fn["name"]))
    return names


def _cascade_tool_awareness_line(tool_names: list[str]) -> str:
    """Addendum v1.5's per-voter tool-awareness system line: appended AFTER
    the client's messages for every VOTER GATE voter (direct-mode voters get
    no advisory prompt otherwise), so a voter can vote "TOOL_TURN" instead
    of guessing at an answer only a client tool call could actually
    determine. ``normalize_candidate`` already lowercases, so an extracted
    ``ANSWER: TOOL_TURN`` candidate compares equal to the literal
    ``"tool_turn"`` this module checks for — no extra normalization needed.
    """
    joined = ", ".join(tool_names) if tool_names else "(unnamed tools)"
    return (
        f"The client has external tools available: {joined}. You cannot "
        "call them. If a correct answer requires using those tools rather "
        "than reasoning or knowledge, reply exactly: ANSWER: TOOL_TURN — "
        "otherwise answer the request directly."
    )


# Addendum v1.1 (judge gate): a cheap LLM consistency check for freeform
# voter output that never produces a comparable exact candidate (prose,
# explanations, ...). Only reached when exact-match consensus already missed.
_JUDGE_SYSTEM_PROMPT = (
    "You compare answers for substantive agreement. Reply with exactly one "
    "word: CONSISTENT if they give the same answer/conclusion, DIFFERENT "
    "otherwise."
)
_JUDGE_REQUEST_CHARS = 1500
_JUDGE_ANSWER_CHARS = 2000


def _judge_messages(messages: list, answer_a: str, answer_b: str) -> list[dict]:
    last_user = next(
        (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    request_tail = str(last_user or "")[-_JUDGE_REQUEST_CHARS:]
    user = (
        f"{request_tail}\n\nAnswer A:\n{str(answer_a or '')[:_JUDGE_ANSWER_CHARS]}"
        f"\n\nAnswer B:\n{str(answer_b or '')[:_JUDGE_ANSWER_CHARS]}"
    )
    return [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# Addendum v1.2 (verified cascade): an independent sandboxed check of an
# exact-candidate answer. Orthogonal to agreement — it catches the residual
# "confident consensus/aggregator, wrong anyway" failure mode that voter
# agreement alone cannot see.
_VERIFIER_SYSTEM_PROMPT = (
    "You write a short standalone Python 3 program that CHECKS a candidate "
    "answer. The program must recompute or verify the answer independently "
    "and print exactly one final line: VERDICT: CORRECT or VERDICT: WRONG. "
    "If the claim cannot be checked by computation, print VERDICT: "
    "UNCHECKABLE. No network, no files, stdlib only, under 5 seconds of "
    "compute."
)
_VERIFIER_PROBLEM_CHARS = 4000


def _verifier_messages(messages: list, candidate: str) -> list[dict]:
    last_user = next(
        (m.get("content") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    problem = str(last_user or "")[:_VERIFIER_PROBLEM_CHARS]
    user = f"{problem}\n\nCandidate answer: {candidate}"
    return [
        {"role": "system", "content": _VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _verifier_note_text(verdict: str, on: str, candidate: str) -> str:
    """Human-readable summary of a verifier run, folded into reference
    context so the aggregator sees WHY a consensus/tier-1 answer was struck
    (or confirmed) rather than just a bare tier flip."""
    if verdict == "wrong":
        return (
            f"Automated check REJECTED the {on} answer {candidate}: "
            "independent verification returned WRONG."
        )
    if verdict == "correct":
        return f"Automated check CONFIRMED the {on} answer {candidate}."
    return f"Automated check was inconclusive for the {on} answer {candidate}."


def _run_cascade_verifier(
    *,
    messages: list,
    candidate: str,
    verifier_slot: dict,
    slot_timeout: float | None,
    on: str,
) -> tuple[str, tuple[str, str, Any] | None]:
    """One verifier LLM call (writes a standalone check script) plus a
    sandboxed execution of that script for ``candidate``. Returns
    ``(verdict, reference_entry)``:

    - ``verdict`` is always one of ``"correct"``/``"wrong"``/``"inconclusive"``
      — an LLM failure degrades to ``"inconclusive"`` the same as an
      execution failure (``moa_cascade.run_verification`` never raises), so a
      broken verifier never blocks the cascade.
    - ``reference_entry`` is a ``(label, text, _RefAccounting)`` tuple folding
      the verifier LLM's real usage into billing (labelled
      ``"verifier — <slot>"``), or ``None`` when the LLM call itself never
      returned (nothing to bill).
    """
    verifier_runtime = _slot_runtime(verifier_slot)
    verifier_timeout = min(60.0, slot_timeout) if slot_timeout else 60.0
    verifier_msgs = _verifier_messages(messages, candidate)
    try:
        response = call_llm(
            task="moa_verifier",
            messages=_maybe_apply_moa_cache_control(verifier_msgs, verifier_runtime),
            temperature=0.0,
            max_tokens=2000,
            timeout=verifier_timeout,
            **verifier_runtime,
        )
    except Exception as exc:
        logger.warning("MoA cascade verifier call failed: %s", exc)
        return "inconclusive", None

    reply = _extract_message_fields(response).get("content") or ""
    usage = _normalize_chunk_usage(getattr(response, "usage", None), verifier_runtime)
    code = extract_python_block(reply)
    verdict = moa_cascade.run_verification(code)
    entry = (
        f"verifier — {_slot_label(verifier_slot)}",
        _verifier_note_text(verdict, on, candidate),
        _RefAccounting(
            usage,
            messages=verifier_msgs,
            output=reply,
            model=verifier_slot.get("model"),
            provider=verifier_runtime.get("provider") or verifier_slot.get("provider"),
            temperature=0.0,
        ),
    )
    return verdict, entry


def _cascade_last_non_system_role(messages: list) -> str | None:
    """Role of the last non-system message in a raw client message list —
    addendum v1.5's mid-loop signal. "tool" or "assistant" means the client
    is already mid an agentic tool loop (the latest thing in the
    conversation is a tool result, or a prior assistant turn it is still
    acting on); "user" (or nothing at all) means a fresh turn for the VOTER
    GATE to evaluate.
    """
    for msg in reversed(messages):
        role = msg.get("role")
        if role != "system":
            return role
    return None


def _cascade_bypass_mode(common: dict) -> str | None:
    """Reasons a cascade request skips the voter pool for an acting-solo call.

    Checked in this order — context-solo wins first, regardless of tools:

    - "context-solo": the estimated request size exceeds
      ``cascade.max_context_tokens`` — voter slots are ~128k-context
      wafer/local models, so an oversized request would error through the
      whole pool; the acting slot (typically a 400k-class plan model) takes
      it solo instead. Estimate: total message chars / 4.
    - "tool-solo": the client sent tools AND ``cascade.tool_turns ==
      "solo"`` — the addendum v1.4 behavior, kept as an explicit opt-out:
      every tool-carrying request bypasses voters unconditionally (voters
      cannot drive a client tool loop, and advisory context measurably
      hurts tool work).
    - "tool-solo-mid-loop": tools present, ``cascade.tool_turns ==
      "detect"`` (the addendum v1.5 default), and the session is already
      mid an agentic tool loop — the last non-system message is "tool" or
      "assistant". There is nothing to vote on: the loop is already
      running, so the acting model continues solo (surfaced with reason
      "mid-loop").
    - ``None``: no immediate bypass. Either the request is tool-free (the
      normal cascade tier-0/1/2 flow applies), or it carries tools with
      ``tool_turns == "detect"`` and the last non-system message is "user"
      — a fresh turn the addendum v1.5 VOTER GATE must decide per-turn (see
      `_run_cascade_tool_turn_gate` / `_stream_cascade_tool_turn_gate`); the
      caller checks ``common["tools"]`` itself once this returns ``None``
      to route there instead of the plain tool-free cascade turn.
    """
    cascade_cfg = common.get("cascade") or {}
    cap = cascade_cfg.get("max_context_tokens") or 0
    if cap:
        est = sum(len(str(m.get("content") or "")) for m in common["messages"]) // 4
        if est > cap:
            return "context-solo"
    if common["tools"]:
        if (cascade_cfg.get("tool_turns") or "detect") == "solo":
            return "tool-solo"
        if _cascade_last_non_system_role(common["messages"]) in {"tool", "assistant"}:
            return "tool-solo-mid-loop"
        return None
    return None


def _cascade_solo_turn_for_bypass(common: dict, bypass: str) -> dict:
    """Turn a `_cascade_bypass_mode` sentinel into the actual solo-turn call.

    Addendum v1.5's "tool-solo-mid-loop" sentinel becomes
    ``{"mode": "tool-solo", "reason": "mid-loop"}``; the plain "tool-solo"
    (config ``tool_turns: "solo"``) and "context-solo" sentinels pass
    through unchanged, so ``usage.moa.cascade`` stays byte-identical to
    addendum v1.4 for those two modes.
    """
    if bypass == "tool-solo-mid-loop":
        return _run_cascade_solo_turn(
            common, "tool-solo", cascade_extra={"reason": "mid-loop"}
        )
    return _run_cascade_solo_turn(common, bypass)


def _run_cascade_solo_turn(
    common: dict,
    mode: str = "tool-solo",
    *,
    reference_outputs: list | None = None,
    cascade_extra: dict | None = None,
) -> dict:
    """Addendum v1.4 §A: a tool-carrying request on a cascade preset runs the
    acting aggregator SOLO — no advisory guidance attached, ever, on this
    path. Measured basis: advisory context hurts tool/code work, so a
    cascade preset should behave like a plain solo model whenever the
    client is driving tool use. Returns the same turn-result dict shape as
    the other ``_run_turn`` branches; ``usage.moa.cascade`` surfaces
    ``{"tier": None, "mode": mode}`` (plus any ``cascade_extra`` keys) so a
    client/trace can tell this apart from a fanout call.

    ``reference_outputs`` (addendum v1.5): reference entries to fold into
    this turn's billing/trace even though the acting call itself runs
    solo — used by the VOTER GATE, where the voters DID run (to decide
    whether to re-engage cascade) even though the winning outcome is still
    an acting-solo call. ``None`` (default) means no voters ran, preserving
    the addendum v1.4 shape (``reference_outputs: []``) exactly.

    ``cascade_extra`` (addendum v1.5): additional keys folded into the
    ``cascade`` usage dict beyond ``tier``/``mode`` (``reason``/``votes``).
    ``None`` (default) adds nothing, so the plain "tool-solo"/"context-solo"
    callers get the exact ``{"tier": None, "mode": mode}`` dict addendum
    v1.4 always returned.
    """
    messages = common["messages"]
    agg_messages = [dict(m) for m in messages]
    # Prompt-cache decoration (see agent/moa_loop._maybe_apply_moa_cache_control):
    # the solo lane re-sends the whole growing session every tool iteration, so
    # on cache-honoring routes this is the single biggest cache win in the
    # proxy. Applied to the outgoing copy only — traces keep the plain shape.
    response = call_llm(
        task="moa_aggregator",
        messages=_maybe_apply_moa_cache_control(agg_messages, _slot_runtime(common["aggregator"])),
        temperature=common["aggregator_temperature"],
        max_tokens=common["max_tokens"],
        tools=common["tools"],
        extra_body=_acting_extra_body(common, common["aggregator"]),
        timeout=common["slot_timeout"],
        **_slot_runtime(common["aggregator"]),
    )
    runtime = _slot_runtime(common["aggregator"])
    agg_usage = _normalize_chunk_usage(getattr(response, "usage", None), runtime)
    cascade_result: dict[str, Any] = {"tier": None, "mode": mode}
    if cascade_extra:
        cascade_result.update(cascade_extra)
    return {
        "reference_outputs": list(reference_outputs) if reference_outputs else [],
        "refs_from_cache": False,
        "agg_messages": agg_messages,
        "response": response,
        "agg_usage": agg_usage,
        "acting_slot": common["aggregator"],
        "cascade": cascade_result,
        "winner_text": None,
    }


def _effective_min_consensus(cascade_cfg: dict, voters: list) -> int:
    """min_consensus, degraded to the LIVE voter count when
    ``cascade.degraded_consensus`` is on (floor 2).

    Measured motivation (2026-07-08 session-sim): upstream quota bursts can
    kill half the fan-out (2 of 4 voters 429'd) while the survivors agree on
    the right answer — counting dead voters in the denominator turned a
    clean 2-of-2 live consensus into acting-solo. Under full health this is
    byte-identical to the configured min_consensus; it only bends during
    partial voter outages, and never below 2 independent agreeing voters.
    Default OFF: benchmark presets keep the strict measured semantics.
    """
    base = int(cascade_cfg.get("min_consensus") or 2)
    if not cascade_cfg.get("degraded_consensus"):
        return base
    live = sum(
        1 for _label, text, _acct in voters if not moa_cascade.is_boilerplate(text)
    )
    return max(2, min(base, live))


def _cascade_tier0_gate(common: dict, messages: list, voters: list) -> dict:
    """Tier-0 voter-agreement gate shared by the non-streaming cascade turn
    (`_run_cascade_turn`) and the streaming cascade turn (addendum v1.4 §B,
    `_stream_cascade_turn`): exact consensus (+ optional verifier strike),
    then the judge gate for freeform agreement. See
    docs/plans/moa-cascade-spec.md for the full tier-0/1/2 semantics.

    Returns ``{"result": <turn dict>}`` when a tier-0 hit already resolves
    the request (ready to return as-is from the non-streaming path, or to
    emit as a single content delta from the streaming path). Otherwise
    returns ``{"tier1": <bundle>}`` — every piece of state
    `_cascade_after_tier1` needs to run tier 1 (and possibly tier 2),
    including the tier-1 ``agg_messages`` with guidance already attached.
    """
    from agent.usage_pricing import CanonicalUsage

    cascade_cfg = common["cascade"] or {}
    min_consensus = _effective_min_consensus(cascade_cfg, voters)
    reference_models = common["reference_models"]

    candidates = [extract_candidate(text) for _label, text, _acct in voters]
    votes = _cascade_vote_counts(candidates)
    cons = consensus(candidates, min_consensus)
    normalized_candidates = [
        normalize_candidate(c) if c is not None else None for c in candidates
    ]
    voter_labels = [label for label, _text, _acct in voters]

    # Addendum v1.2 (verified cascade): resolved once, reused at both the
    # tier-0 consensus check below and the tier-1 aggregator-candidate check
    # in `_cascade_after_tier1`. `verify` is only ever "python" once a
    # verifier slot has actually resolved (config normalization nulls it out
    # otherwise), so `verify_active` alone gates every verifier call in this
    # turn.
    verify_mode = cascade_cfg.get("verify")
    verifier_slot = cascade_cfg.get("verifier")
    verify_when = cascade_cfg.get("verify_when") or "weak"
    verify_active = verify_mode == "python" and bool(verifier_slot)
    verify_extra: list[tuple[str, str, Any]] = []
    struck_consensus: str | None = None
    verify_surface: dict[str, Any] = {"ran": False, "verdict": None, "on": None}

    if cons is not None:
        # Exact consensus is tried first regardless of gate — it's free — and
        # always wins tier 0 with gate_used "exact" (addendum v1.1 §1).
        winner_idx = next(
            idx
            for idx, c in enumerate(candidates)
            if c is not None and normalize_candidate(c) == cons
        )
        _winner_label, winner_text, _winner_acct = voters[winner_idx]

        if verify_active:
            # "weak" (default) only pays for verification when the consensus
            # isn't unanimous — a unanimous vote across every configured
            # reference slot is the strongest signal the cheap path already
            # has. "always" verifies every consensus regardless.
            should_verify = verify_when == "always" or votes < len(reference_models)
            if should_verify:
                verdict, entry = _run_cascade_verifier(
                    messages=messages,
                    candidate=cons,
                    verifier_slot=verifier_slot,
                    slot_timeout=common["slot_timeout"],
                    on="consensus",
                )
                verify_surface = {"ran": True, "verdict": verdict, "on": "consensus"}
                if entry is not None:
                    verify_extra = [entry]
                if verdict == "wrong":
                    struck_consensus = cons

        if struck_consensus is None:
            cascade_result = {
                "tier": 0,
                "consensus": cons,
                "votes": votes,
                "voters": voter_labels,
                "candidates": normalized_candidates,
                "gate_used": "exact",
            }
            if verify_mode == "python":
                cascade_result["verify"] = verify_surface
                cascade_result["struck_consensus"] = None
            return {
                "result": {
                    "reference_outputs": voters + verify_extra,
                    "refs_from_cache": False,
                    "agg_messages": [dict(m) for m in messages],
                    "response": None,
                    "agg_usage": CanonicalUsage(),
                    "acting_slot": reference_models[winner_idx],
                    "cascade": cascade_result,
                    "winner_text": winner_text,
                }
            }
        # Verdict "wrong": the consensus answer is struck — fall through to
        # the same no-consensus path (judge gate, then tier 1) as if the
        # voters had never agreed at all. `verify_extra` (the verifier LLM's
        # billing entry) and `struck_consensus` carry forward into whichever
        # tier ultimately returns.

    # Addendum v1.1: judge gate for freeform traffic. Exact consensus just
    # missed (no comparable short candidates agreed) — with gate == "judge",
    # try ONE cheap consistency check on the first two substantive voter
    # outputs before paying for full aggregation. gate_used stays None
    # (surfaced as JSON null) whenever the judge never ran at all — plain
    # exact-mode no-consensus flows, or judge-mode with too few substantive
    # voters to compare.
    gate = cascade_cfg.get("gate") or "exact"
    gate_used: str | None = None
    judge_entry: tuple[str, str, Any] | None = None

    if gate == "judge":
        judge_slot = cascade_cfg.get("judge")
        substantive = [
            (idx, text) for idx, (_label, text, _acct) in enumerate(voters)
            if not is_boilerplate(text)
        ]
        if judge_slot and len(substantive) >= min(2, min_consensus):
            idx_a, text_a = substantive[0]
            _idx_b, text_b = substantive[1]
            try:
                judge_runtime = _slot_runtime(judge_slot)
                judge_timeout = (
                    min(30.0, common["slot_timeout"]) if common["slot_timeout"] else 30.0
                )
                judge_response = call_llm(
                    task="moa_router",
                    messages=_maybe_apply_moa_cache_control(
                        _judge_messages(messages, text_a, text_b), judge_runtime
                    ),
                    temperature=0.0,
                    max_tokens=8,
                    timeout=judge_timeout,
                    **judge_runtime,
                )
            except Exception as exc:
                logger.warning("MoA cascade judge call failed: %s", exc)
                gate_used = "judge-error"
            else:
                judge_reply = _extract_message_fields(judge_response).get("content") or ""
                judge_usage = _normalize_chunk_usage(
                    getattr(judge_response, "usage", None), judge_runtime
                )
                judge_entry = (
                    f"consensus-judge — {_slot_label(judge_slot)}",
                    judge_reply,
                    _RefAccounting(
                        judge_usage,
                        messages=_judge_messages(messages, text_a, text_b),
                        output=judge_reply,
                        model=judge_slot.get("model"),
                        provider=judge_runtime.get("provider") or judge_slot.get("provider"),
                        temperature=0.0,
                    ),
                )
                if judge_reply.strip().upper().startswith("CONSISTENT"):
                    return {
                        "result": {
                            "reference_outputs": voters + [judge_entry],
                            "refs_from_cache": False,
                            "agg_messages": [dict(m) for m in messages],
                            "response": None,
                            "agg_usage": CanonicalUsage(),
                            "acting_slot": reference_models[idx_a],
                            "cascade": {
                                "tier": 0,
                                "consensus": None,
                                "votes": votes,
                                "voters": voter_labels,
                                "candidates": normalized_candidates,
                                "gate_used": "judge",
                            },
                            "winner_text": text_a,
                        }
                    }
                gate_used = "judge-different"

    # Any judge call that actually ran is billed regardless of its verdict
    # (addendum v1.1: "fold ... ONLY when the judge ran"), folded into
    # whichever reference_outputs tier 1/2 below returns. It does NOT feed the
    # aggregator's guidance text — a one-word CONSISTENT/DIFFERENT verdict
    # adds no synthesis-useful context beyond the full voter texts already
    # attached.
    judge_extra = [judge_entry] if judge_entry is not None else []
    # Addendum v1.1 §4: tier-2 escalation stays exact-candidate-only — a
    # judge-gated freeform "discord" (DIFFERENT verdict, or the judge call
    # itself erroring) never escalates in v1; it settles at tier 1 same as an
    # aggregator that simply agreed with a voter.
    judge_discord = gate_used in {"judge-different", "judge-error"}

    # Tier 1: no consensus among voters — run the preset's own aggregator with
    # the voter outputs attached as reference context (existing guidance
    # builder), tool-free (cascade only serves tool-free requests). A
    # tier-0 strike's verifier note (addendum v1.2) rides along here too —
    # unlike the judge entry above, the aggregator MUST see why its exact
    # consensus was rejected, not just have it billed.
    agg_messages = [dict(m) for m in messages]
    clean_arbiter = bool(cascade_cfg.get("clean_arbiter"))
    if clean_arbiter:
        # Clean arbitration: the aggregator re-solves from scratch. Voter
        # context anchors arbiters (measured: a frontier arbiter scored 7/9
        # on disagreements vs ~98% solo); only a verifier strike note is
        # ever attached (it names no candidate answers beyond the struck one).
        if verify_extra:
            _attach_reference_guidance(
                agg_messages,
                _reference_guidance(
                    common["preset_name"], common["aggregator"], verify_extra
                ),
            )
    else:
        guidance = _reference_guidance(
            common["preset_name"], common["aggregator"], voters + verify_extra
        )
        _attach_reference_guidance(agg_messages, guidance)

    return {
        "tier1": {
            "agg_messages": agg_messages,
            "voters": voters,
            "judge_extra": judge_extra,
            "verify_extra": verify_extra,
            "struck_consensus": struck_consensus,
            "gate_used": gate_used,
            "votes": votes,
            "voter_labels": voter_labels,
            "candidates": normalized_candidates,
            "candidates_raw": candidates,
            "verify_mode": verify_mode,
            "verifier_slot": verifier_slot,
            "verify_active": verify_active,
            "verify_surface": verify_surface,
            "judge_discord": judge_discord,
            "clean_arbiter": clean_arbiter,
        }
    }


def _cascade_tool_turn_gate(common: dict, messages: list, voters: list) -> dict:
    """Addendum v1.5 VOTER GATE decision, given an already-completed
    tool-aware voter fan-out (voters answered with the tool-awareness system
    line appended — see `_cascade_tool_awareness_line`,
    `_run_cascade_tool_turn_gate`, `_stream_cascade_tool_turn_gate`). Shared
    by both the non-streaming and streaming tool-turn gates so the
    tool_turn / real-answer / no-consensus decision is identical whichever
    surface ran the fan-out.

    Returns ``{"result": <turn dict>}`` when the voters converged on a REAL
    answer — the session has reverted to cascade, tier 0 exactly as the
    tool-free path (`_cascade_tier0_gate`), including the judge gate and
    verifier, EXCEPT a "tool_turn" consensus never reaches
    `_cascade_tier0_gate` at all (so it can never be sent to the verifier).
    Otherwise returns ``{"solo": {"reason": ..., "extra": {...},
    "reference_outputs": [...]}}`` — acting-solo w/ tools is required
    instead, either because the voters voted "tool_turn" (reason
    "tool_turn_vote", ``extra={"votes": n}``) or no consensus formed at all
    (reason "no-consensus" — this also covers a struck exact consensus or a
    judge-different/error freeform result: anything `_cascade_tier0_gate`
    would otherwise have escalated to the tool-free tier-1 aggregator,
    which cannot forward tools). ``reference_outputs`` folds in the voters
    (and any judge/verifier calls that ran) so the solo turn's billing
    still reflects the vote that happened.
    """
    cascade_cfg = common["cascade"] or {}
    min_consensus = _effective_min_consensus(cascade_cfg, voters)
    candidates = [extract_candidate(text) for _label, text, _acct in voters]
    cons = consensus(candidates, min_consensus)

    if cons == "tool_turn":
        return {
            "solo": {
                "reason": "tool_turn_vote",
                "extra": {"votes": _cascade_vote_counts(candidates)},
                "reference_outputs": voters,
            }
        }

    gate = _cascade_tier0_gate(common, messages, voters)
    if gate.get("result") is not None:
        return gate

    tier1_bundle = gate["tier1"]
    reference_outputs = (
        tier1_bundle["voters"] + tier1_bundle["judge_extra"] + tier1_bundle["verify_extra"]
    )
    return {
        "solo": {
            "reason": "no-consensus",
            "extra": {},
            "reference_outputs": reference_outputs,
        }
    }


def _cascade_after_tier1(
    common: dict,
    messages: list,
    tier1_bundle: dict,
    *,
    agg_text: str,
    agg_usage: Any,
    agg_runtime: dict,
    agg_response: Any,
    agg_messages: list,
) -> dict:
    """Tier-1 candidate decision + optional tier-2 escalation, shared by the
    non-streaming cascade turn and the streaming cascade turn (addendum v1.4
    §B). ``agg_text``/``agg_usage``/``agg_runtime``/``agg_response`` describe
    an ALREADY-COMPLETED tier-1 aggregator call — obtained via a blocking
    ``call_llm`` (non-streaming path, and the streaming path whenever an
    escalate preset is configured) or accumulated from a live token stream
    (streaming path with no escalate preset, where ``agg_response`` is
    ``None`` since there is no SDK response object to hand back).

    A tier-2 escalation, when it fires, is always a single blocking
    ``call_llm`` regardless of whether the caller is streaming — the client
    only ever sees the FINAL chosen answer for tier 2, never a live token
    stream of it.
    """
    agg_candidate = extract_candidate(agg_text)
    tier1_candidate_norm = (
        normalize_candidate(agg_candidate) if agg_candidate is not None else None
    )

    escalate_preset = common.get("cascade_escalate_preset")
    judge_discord = tier1_bundle["judge_discord"]
    agrees_any = any(agrees(agg_candidate, c) for c in tier1_bundle["candidates_raw"])

    verify_extra = tier1_bundle["verify_extra"]
    verify_surface = tier1_bundle["verify_surface"]
    verify_mode = tier1_bundle["verify_mode"]
    struck_consensus = tier1_bundle["struck_consensus"]

    # Addendum v1.2: verify the aggregator's own candidate. Exact-gate flows
    # only — a judge-gated freeform "discord" never had a comparable
    # candidate in the first place, and there is nothing to verify when the
    # aggregator produced none. Unlike the tier-0 knob, tier 1 ALWAYS
    # verifies when active: tier 1 is already the slow path, so
    # `verify_when` only guards the fast (tier-0) path.
    tier1_verify_wrong = False
    if tier1_bundle["verify_active"] and not judge_discord and agg_candidate is not None:
        verdict, entry = _run_cascade_verifier(
            messages=messages,
            candidate=agg_candidate,
            verifier_slot=tier1_bundle["verifier_slot"],
            slot_timeout=common["slot_timeout"],
            on="tier1",
        )
        verify_surface = {"ran": True, "verdict": verdict, "on": "tier1"}
        if entry is not None:
            verify_extra = verify_extra + [entry]
        tier1_verify_wrong = verdict == "wrong"

    # A tier-1 "wrong" verdict escalates regardless of voter agreement, same
    # as the existing no-voter-agreement trigger — both need an escalate
    # preset configured and no judge-gated discord already settling at tier 1.
    escalate_now = (
        bool(escalate_preset)
        and not judge_discord
        and (not agrees_any or tier1_verify_wrong)
    )

    voters = tier1_bundle["voters"]
    judge_extra = tier1_bundle["judge_extra"]
    votes = tier1_bundle["votes"]
    voter_labels = tier1_bundle["voter_labels"]
    normalized_candidates = tier1_bundle["candidates"]
    gate_used = tier1_bundle["gate_used"]

    if not escalate_now:
        cascade_result = {
            "tier": 1,
            "consensus": None,
            "votes": votes,
            "voters": voter_labels,
            "candidates": normalized_candidates,
            "tier1_candidate": tier1_candidate_norm,
            "gate_used": gate_used,
        }
        if verify_mode == "python":
            cascade_result["verify"] = verify_surface
            cascade_result["struck_consensus"] = struck_consensus
        return {
            "reference_outputs": voters + judge_extra + verify_extra,
            "refs_from_cache": False,
            "agg_messages": agg_messages,
            "response": agg_response,
            "agg_usage": agg_usage,
            "acting_slot": None,  # preset aggregator acted — default is right
            "cascade": cascade_result,
            "winner_text": None,
        }

    # Tier 2: the aggregator agreed with NO voter (or a verifier struck its
    # candidate) and an escalate preset is configured — call THAT preset's
    # aggregator solo, with the voters PLUS the tier-1 aggregator's own
    # output (and any verifier note) attached as reference context.
    tier1_label = f"tier1-aggregator — {_slot_label(common['aggregator'])}"
    tier1_acct = _RefAccounting(
        agg_usage,
        messages=agg_messages,
        output=agg_text,
        model=common["aggregator"].get("model"),
        provider=agg_runtime.get("provider") or common["aggregator"].get("provider"),
        temperature=common["aggregator_temperature"],
    )
    reference_outputs = voters + judge_extra + verify_extra + [(tier1_label, agg_text, tier1_acct)]

    escalate_aggregator = escalate_preset.get("aggregator") or {}
    escalate_messages = [dict(m) for m in messages]
    if not tier1_bundle["clean_arbiter"]:
        escalate_guidance = _reference_guidance(
            common["preset_name"], escalate_aggregator, reference_outputs
        )
        _attach_reference_guidance(escalate_messages, escalate_guidance)
    escalate_response = call_llm(
        task="moa_aggregator",
        messages=_maybe_apply_moa_cache_control(
            escalate_messages, _slot_runtime(escalate_aggregator)
        ),
        extra_body=_acting_extra_body(common, escalate_aggregator),
        temperature=escalate_preset.get("aggregator_temperature", 0.4),
        max_tokens=common["max_tokens"],
        timeout=common["slot_timeout"],
        **_slot_runtime(escalate_aggregator),
    )
    escalate_runtime = _slot_runtime(escalate_aggregator)
    escalate_usage = _normalize_chunk_usage(
        getattr(escalate_response, "usage", None), escalate_runtime
    )
    cascade_result = {
        "tier": 2,
        "consensus": None,
        "votes": votes,
        "voters": voter_labels,
        "candidates": normalized_candidates,
        "tier1_candidate": tier1_candidate_norm,
        "gate_used": gate_used,
    }
    if verify_mode == "python":
        cascade_result["verify"] = verify_surface
        cascade_result["struck_consensus"] = struck_consensus
    return {
        "reference_outputs": reference_outputs,
        "refs_from_cache": False,
        "agg_messages": escalate_messages,
        "response": escalate_response,
        "agg_usage": escalate_usage,
        "acting_slot": escalate_aggregator,
        "cascade": cascade_result,
        "winner_text": None,
    }


def _run_cascade_tier1_blocking(common: dict, messages: list, tier1_bundle: dict) -> dict:
    """Blocking tier-1 aggregator call + `_cascade_after_tier1` decision.

    Shared by `_run_cascade_turn` (non-streaming) and the streaming cascade
    turn (addendum v1.4 §B step 4) whenever an escalate preset is
    configured — that answer must be inspected before the client sees it, so
    it always runs non-streaming even for a streaming request.
    """
    agg_messages = tier1_bundle["agg_messages"]
    aggregator = common["aggregator"]
    agg_runtime = _slot_runtime(aggregator)
    cascade_cfg = common.get("cascade") or {}
    if aggregator.get("agent") == "rlm" and cascade_cfg.get("clean_arbiter"):
        # An RLM arbiter re-solves from scratch with the voter loop
        # (reason -> python -> observe). Measured basis (iter 38): the loop
        # lifts open arbiters massively (deepseek-v4-pro 53% -> 90%), and a
        # clean arbiter gets client messages only — exactly the voter-shaped
        # call `_run_reference(direct=True)` already implements.
        from agent.moa_loop import _run_reference

        _label, rlm_text, rlm_acct = _run_reference(
            aggregator,
            [dict(m) for m in common["messages"]],
            temperature=common["aggregator_temperature"],
            max_tokens=common["max_tokens"],
            timeout=common["slot_timeout"],
            direct=True,
        )
        agg_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=rlm_text, reasoning_content=None, tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )
        agg_usage = getattr(rlm_acct, "usage", None) or _normalize_chunk_usage(
            None, agg_runtime
        )
        agg_text = rlm_text or ""
        return _cascade_after_tier1(
            common,
            messages,
            tier1_bundle,
            agg_text=agg_text,
            agg_usage=agg_usage,
            agg_runtime=agg_runtime,
            agg_response=agg_response,
            agg_messages=agg_messages,
        )
    agg_response = call_llm(
        task="moa_aggregator",
        messages=_maybe_apply_moa_cache_control(
            agg_messages, _slot_runtime(common["aggregator"])
        ),
        temperature=common["aggregator_temperature"],
        max_tokens=common["max_tokens"],
        tools=None,
        extra_body=common["extra_body"] or None,
        timeout=common["slot_timeout"],
        **_slot_runtime(common["aggregator"]),
    )
    agg_usage = _normalize_chunk_usage(getattr(agg_response, "usage", None), agg_runtime)
    agg_text = _extract_message_fields(agg_response).get("content") or ""
    return _cascade_after_tier1(
        common,
        messages,
        tier1_bundle,
        agg_text=agg_text,
        agg_usage=agg_usage,
        agg_runtime=agg_runtime,
        agg_response=agg_response,
        agg_messages=agg_messages,
    )


def _acting_extra_body(common: dict, slot: dict) -> dict | None:
    """extra_body for an ACTING call to ``slot``: the client's passthrough
    fields plus, on OpenAI-family slots, a session-stable
    ``prompt_cache_key`` (upstream routes requests sharing a key to the
    same cache shard — without it, a multi-lane proxy session load-balances
    away from its own cached prefix). Identity for every other provider.
    Never overrides a client-supplied prompt_cache_key.
    """
    extra = dict(common.get("extra_body") or {})
    sess = common.get("session_info")
    provider = str(slot.get("provider") or "").strip().lower()
    if sess and provider in {"openai", "openai-codex"} and "prompt_cache_key" not in extra:
        extra["prompt_cache_key"] = sess["cache_key"]
    return extra or None


def _cascade_voter_view(common: dict, messages: list) -> list:
    """The voter-facing view of the client conversation (cascade fan-outs
    only): provider-agnostic plain-text projection + a bounded recency
    window. The ACTING lanes (solo, tier-1 aggregator, tier-2 escalate)
    always keep the full verbatim transcript — see
    hermes_cli/proxy/moa_session.py for both contracts and the measured
    motivations (cross-provider 400s on replayed tool/opaque content; the
    Cerebras TPM quota tripped by full-context fan-out after one 10k turn).

    Window stats (when anything was trimmed) are recorded on
    ``common["voter_view"]`` so usage.moa can surface what the voters saw.
    """
    view = project_history_for_voters([dict(m) for m in messages])
    budget = (common.get("cascade") or {}).get("voter_context_tokens")
    view, stats = window_history(view, budget)
    if stats:
        common["voter_view"] = stats
    return view


def _run_cascade_turn(common: dict) -> dict:
    """Lazy MoA turn: tier-0 wafer-voter consensus, tier-1 aggregator, tier-2
    escalation. See docs/plans/moa-cascade-spec.md.

    Only reached for a non-streaming, tool-free request on a preset with
    ``mode == "cascade"`` and >=2 reference slots (config normalization
    already guarantees the slot count). Returns the same turn-result dict
    shape as the fanout/draft_review path in ``_run_turn`` above. The tier-0
    gate and tier-1/2 decision are factored into `_cascade_tier0_gate` /
    `_cascade_after_tier1` so the streaming cascade turn (addendum v1.4 §B,
    `_stream_cascade_turn`) can reuse the exact same logic.
    """
    messages = common["messages"]
    reference_models = common["reference_models"]

    # Tier 0: every voter answers the client's ACTUAL request directly (no
    # advisory system prompt) in parallel, on the projected + windowed voter
    # view (`_cascade_voter_view`) — acting lanes keep the full transcript.
    voters = _run_references_parallel(
        reference_models,
        _cascade_voter_view(common, messages),
        temperature=common["reference_temperature"],
        max_tokens=common["reference_max_tokens"] or common["max_tokens"],
        timeout=common["slot_timeout"],
        quorum_grace=common["preset"].get("reference_quorum_grace"),
        direct=True,
    )
    gate = _cascade_tier0_gate(common, messages, voters)
    if gate.get("result") is not None:
        return gate["result"]
    return _run_cascade_tier1_blocking(common, messages, gate["tier1"])


def _run_cascade_tool_turn_gate(common: dict) -> dict:
    """Addendum v1.5 VOTER GATE (non-streaming): reached when the resolved
    cascade preset carries tools, ``cascade.tool_turns == "detect"`` (the
    default), and the last non-system message is "user" —
    `_cascade_bypass_mode` already ruled out the immediate bypasses
    (context-solo, the config ``tool_turns: "solo"`` opt-out, and
    mid-loop). Runs the tier-0 voter fan-out with ONE extra tool-awareness
    system line appended to what each voter sees, then applies the shared
    `_cascade_tool_turn_gate` decision. See docs/plans/moa-cascade-spec.md
    addendum v1.5.
    """
    messages = common["messages"]
    reference_models = common["reference_models"]
    voter_messages = _cascade_voter_view(common, messages) + [
        {
            "role": "system",
            "content": _cascade_tool_awareness_line(_cascade_tool_names(common["tools"])),
        }
    ]
    voters = _run_references_parallel(
        reference_models,
        voter_messages,
        temperature=common["reference_temperature"],
        max_tokens=common["reference_max_tokens"] or common["max_tokens"],
        timeout=common["slot_timeout"],
        quorum_grace=common["preset"].get("reference_quorum_grace"),
        direct=True,
    )
    gate = _cascade_tool_turn_gate(common, messages, voters)
    if gate.get("result") is not None:
        return gate["result"]
    solo = gate["solo"]
    return _run_cascade_solo_turn(
        common,
        "tool-solo",
        reference_outputs=solo["reference_outputs"],
        cascade_extra={"reason": solo["reason"], **solo["extra"]},
    )


def _run_cascade_voters_with_progress(
    reference_models: list[dict],
    ref_messages: list,
    *,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float | None,
    quorum_grace: float | None,
    on_complete,
) -> list[tuple[str, str, Any]]:
    """Same fan-out `_run_references_parallel(..., direct=True)` performs
    (dispatch, quorum-grace straggler dropping, per-slot RLM voters), but
    calls ``on_complete(label)`` synchronously as each voter's result lands
    — including a straggler dropped by the quorum deadline — so the
    streaming cascade turn (addendum v1.4 §B step 2) can emit a live
    "[voter <label>: done]" reasoning delta per voter instead of waiting for
    the whole fan-out. Kept as a server-local sibling of
    `_run_references_parallel` because that function has no progress-
    callback hook and `agent/moa_loop.py` is shared, non-proxy-specific
    runtime code.
    """
    import time as _time
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    from agent.usage_pricing import CanonicalUsage

    if not reference_models:
        return []

    results: list[tuple[str, str, Any] | None] = [None] * len(reference_models)
    futures = {}
    workers = min(_MAX_REFERENCE_WORKERS, len(reference_models))
    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        for idx, slot in enumerate(reference_models):
            if slot.get("provider") == "moa":
                results[idx] = (
                    _slot_label(slot),
                    "[skipped: MoA presets cannot recursively reference MoA]",
                    _RefAccounting(CanonicalUsage()),
                )
                on_complete(results[idx][0])
                continue
            futures[
                executor.submit(
                    _run_reference,
                    slot,
                    ref_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    direct=True,
                )
            ] = idx

        if quorum_grace is None or len(futures) < 2:
            for future, idx in futures.items():
                results[idx] = future.result()
                on_complete(results[idx][0])
        else:
            started = _time.time()
            pending = set(futures)
            deadline = None  # armed once the quorum (all but one) is in
            while pending:
                wait_timeout = None
                if deadline is not None:
                    wait_timeout = max(0.05, deadline - _time.time())
                done, pending = wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)
                for future in done:
                    idx = futures[future]
                    results[idx] = future.result()
                    on_complete(results[idx][0])
                if not pending:
                    break
                if deadline is None and len(pending) == 1:
                    deadline = _time.time() + max(
                        1.0, (_time.time() - started) * float(quorum_grace)
                    )
                elif deadline is not None and _time.time() >= deadline and not done:
                    for future in pending:
                        idx = futures[future]
                        slot = reference_models[idx]
                        logger.info(
                            "MoA quorum: dropping straggler reference %s", _slot_label(slot)
                        )
                        results[idx] = (
                            _slot_label(slot),
                            "[dropped: reference exceeded the quorum deadline]",
                            _RefAccounting(CanonicalUsage()),
                        )
                        on_complete(results[idx][0])
                    pending = set()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return [r for r in results if r is not None]


def _attach_session_usage(usage: dict, common: dict, cascade: Any) -> None:
    """Fold session continuity + voter-window observability into usage.moa
    and remember the mode that served this turn (drives nothing yet; it is
    the session-level answer to "did this conversation get pinned to solo,
    or did it revert to cascade?" without trawling logs)."""
    moa = usage.get("moa")
    if not isinstance(moa, dict):
        return
    if common.get("voter_view"):
        moa["voter_view"] = common["voter_view"]
    sess = common.get("session_info")
    if not sess:
        return
    mode = None
    if isinstance(cascade, dict):
        tier = cascade.get("tier")
        mode = cascade.get("mode") or (f"tier{tier}" if tier is not None else None)
    moa["session"] = {
        "key": sess["key"],
        "turns": sess["turns"],
        "last_mode": sess["last_mode"],
    }
    if mode:
        _session_registry.note_mode(sess["key"], str(mode))


def _cascade_stream_usage(turn: dict, common: dict) -> dict:
    """Final-chunk ``usage`` for the streaming cascade turn (addendum v1.4
    §B): the same shape/derivation as the non-streaming path's usage
    assembly in ``_handle_non_streaming`` — cascade never serves references
    from the advisory cache, so ``refs_from_cache`` is always False here.
    """
    from agent.usage_pricing import CanonicalUsage

    reference_outputs = turn["reference_outputs"]
    agg_usage = turn["agg_usage"]
    ref_usage = CanonicalUsage()
    for _label, _text, acct in reference_outputs:
        if isinstance(getattr(acct, "usage", None), CanonicalUsage):
            ref_usage = ref_usage + acct.usage
    usage = _usage_to_openai(agg_usage + ref_usage)
    usage["moa"] = _usage_breakdown(reference_outputs, agg_usage, False, common.get("routing"))
    usage["moa"]["cascade"] = turn["cascade"]
    _attach_session_usage(usage, common, turn["cascade"])
    return usage


def _extract_message_fields(response: Any) -> dict[str, Any]:
    """Pull content / tool_calls / reasoning off a complete SDK response."""
    out: dict[str, Any] = {"role": "assistant", "content": None}
    try:
        message = response.choices[0].message
    except Exception:
        return out
    if isinstance(message, dict):
        get = message.get
    else:
        def get(name, default=None):
            value = getattr(message, name, default)
            if value is None and hasattr(message, "model_extra"):
                extra = message.model_extra
                if isinstance(extra, dict):
                    value = extra.get(name, default)
            return value

    content = get("content")
    out["content"] = content if isinstance(content, str) else (str(content) if content else None)
    for reasoning_key in ("reasoning_content", "reasoning"):
        reasoning = get(reasoning_key)
        if isinstance(reasoning, str) and reasoning.strip():
            out["reasoning_content"] = reasoning
            break
    tool_calls = get("tool_calls")
    if tool_calls:
        rendered = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function") or {}
                rendered.append(
                    {
                        "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": tc.get("type") or "function",
                        "function": {
                            "name": fn.get("name") or "",
                            "arguments": fn.get("arguments") or "",
                        },
                    }
                )
            else:
                fn = getattr(tc, "function", None)
                rendered.append(
                    {
                        "id": getattr(tc, "id", None) or f"call_{uuid.uuid4().hex[:24]}",
                        "type": getattr(tc, "type", None) or "function",
                        "function": {
                            "name": getattr(fn, "name", None) or "",
                            "arguments": getattr(fn, "arguments", None) or "",
                        },
                    }
                )
        out["tool_calls"] = rendered
    return out


def _finish_reason(response: Any) -> str:
    try:
        reason = response.choices[0].finish_reason
        if reason:
            return str(reason)
    except Exception:
        pass
    return "tool_calls" if _extract_message_fields(response).get("tool_calls") else "stop"


# ---------------------------------------------------------------------------
# Streaming reference / aggregator workers (threads — call_llm is sync)
# ---------------------------------------------------------------------------

_DONE = object()



def _as_chunk_stream(response: Any):
    """Adapt a call_llm(stream=True) return value into a chunk iterator.

    Some providers (openai-codex plan OAuth, measured 2026-07-05) ignore
    stream=True and hand back one complete response object; iterating it
    raises "'types.SimpleNamespace' object is not iterable" and the turn
    surfaces as an empty completion to streaming clients. When the return
    value is not iterable, synthesize a single stream chunk carrying the
    full message as a delta (content + reasoning + tool_calls) plus usage.
    """
    if hasattr(response, "__iter__"):
        return response
    choices = getattr(response, "choices", None) or []
    message = getattr(choices[0], "message", None) if choices else None
    delta = SimpleNamespace(
        content=getattr(message, "content", None),
        reasoning_content=getattr(message, "reasoning_content", None),
        tool_calls=getattr(message, "tool_calls", None) or None,
    )
    finish = getattr(choices[0], "finish_reason", None) if choices else "stop"
    chunk = SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish or "stop")],
        usage=getattr(response, "usage", None),
    )
    return iter([chunk])

def _delta_reasoning_text(delta: Any) -> str | None:
    """Reasoning text from a stream delta, across provider spellings."""
    for key in ("reasoning_content", "reasoning"):
        value = getattr(delta, key, None)
        if value is None and hasattr(delta, "model_extra"):
            extra = delta.model_extra
            if isinstance(extra, dict):
                value = extra.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _reference_stream_worker(
    slot: dict,
    ref_messages: list,
    *,
    temperature: float | None,
    max_tokens: int | None,
    timeout: float | None,
    push,
    abort: threading.Event,
) -> tuple[str, str, Any]:
    """Stream one reference model, pushing its thinking + advice text as it
    arrives. Returns the same ``(label, text, _RefAccounting)`` tuple shape as
    the non-streaming fan-out so traces and usage accounting are uniform.

    Never raises — a failed reference becomes a labelled note, matching
    ``_run_reference`` semantics, so the aggregator still acts on partial
    context.
    """
    from agent.usage_pricing import CanonicalUsage

    label = _slot_label(slot)
    runtime = _slot_runtime(slot)
    messages = [{"role": "system", "content": _REFERENCE_SYSTEM_PROMPT}, *ref_messages]
    messages = _maybe_apply_moa_cache_control(messages, runtime)
    parts: list[str] = []
    usage = CanonicalUsage()
    try:
        stream = call_llm(
            task="moa_reference",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
            stream_options={"include_usage": True},
            **runtime,
        )
        for chunk in _as_chunk_stream(stream):
            if abort.is_set():
                break
            raw_usage = getattr(chunk, "usage", None)
            if raw_usage:
                usage = _normalize_chunk_usage(raw_usage, runtime)
            choices = getattr(chunk, "choices", None) or []
            delta = getattr(choices[0], "delta", None) if choices else None
            if delta is None:
                continue
            reasoning = _delta_reasoning_text(delta)
            if reasoning:
                push(reasoning)
            content = getattr(delta, "content", None)
            if content:
                parts.append(content)
                push(content)
    except Exception as exc:
        logger.warning("MoA proxy reference %s failed: %s", label, exc)
        note = f"[failed: {exc}]"
        parts.append(note)
        push(note)
    text = "".join(parts).strip() or "(empty response)"
    acct = _RefAccounting(
        usage,
        messages=messages,
        output=text,
        model=slot.get("model"),
        provider=runtime.get("provider") or slot.get("provider"),
        temperature=temperature,
    )
    return label, text, acct


# ---------------------------------------------------------------------------
# Streaming cascade turn (addendum v1.4 §B, docs/plans/moa-cascade-spec.md)
# ---------------------------------------------------------------------------


async def _stream_cascade_tier1_live(
    common: dict,
    messages: list,
    tier1_bundle: dict,
    *,
    send_chunk,
    send_reasoning,
    loop: asyncio.AbstractEventLoop,
    abort: threading.Event,
) -> tuple[dict, str | None]:
    """Stream the cascade tier-1 aggregator live, token by token (addendum
    v1.4 §B step 4) — only reached when no escalate preset is configured, so
    a tier-1 "wrong" verifier verdict has nowhere to escalate to and is
    simply surfaced rather than hidden behind an inspect-then-emit call.

    Same event-queue shape as the generic aggregator streaming block in
    `_handle_streaming` (tools are always ``None`` here — cascade only ever
    streams tool-free turns this way, so tool_call deltas never occur).
    Returns the tier-1 turn-result dict from `_cascade_after_tier1` (its
    ``response`` is always ``None`` — the text was already streamed to the
    client) plus the streamed text for trace persistence.
    """
    agg_messages = tier1_bundle["agg_messages"]
    runtime = _slot_runtime(common["aggregator"])
    agg_queue: asyncio.Queue = asyncio.Queue()

    def _push(kind: str, value: Any) -> None:
        loop.call_soon_threadsafe(agg_queue.put_nowait, (kind, value))

    def _worker() -> None:
        try:
            stream = call_llm(
                task="moa_aggregator",
                messages=_maybe_apply_moa_cache_control(agg_messages, runtime),
                temperature=common["aggregator_temperature"],
                max_tokens=common["max_tokens"],
                tools=None,
                extra_body=common["extra_body"] or None,
                timeout=common["slot_timeout"],
                stream=True,
                stream_options={"include_usage": True},
                **runtime,
            )
            for chunk in _as_chunk_stream(stream):
                if abort.is_set():
                    break
                raw_usage = getattr(chunk, "usage", None)
                if raw_usage:
                    _push("usage", raw_usage)
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    reasoning = _delta_reasoning_text(delta)
                    if reasoning:
                        _push("reasoning", reasoning)
                    content = getattr(delta, "content", None)
                    if content:
                        _push("content", content)
                finish = getattr(choice, "finish_reason", None)
                if finish:
                    _push("finish", str(finish))
        except Exception as exc:
            _push("error", str(exc))
        finally:
            _push("done", None)

    future = loop.run_in_executor(None, _worker)

    finish_reason: str | None = None
    raw_usage: Any = None
    content_parts: list[str] = []
    while True:
        kind, value = await agg_queue.get()
        if kind == "done":
            break
        if kind == "reasoning":
            await send_reasoning(value)
        elif kind == "content":
            content_parts.append(value)
            await send_chunk({"content": value})
        elif kind == "finish":
            finish_reason = value
        elif kind == "usage":
            raw_usage = value
        elif kind == "error":
            await send_reasoning(f"\n\n[aggregator failed: {value}]")
            finish_reason = finish_reason or "stop"
    await future

    agg_text = "".join(content_parts)
    agg_usage = _normalize_chunk_usage(raw_usage, runtime)
    # `_cascade_after_tier1` can itself perform blocking I/O (a tier-1
    # verifier `call_llm` + sandboxed subprocess, addendum v1.2) when
    # `verify` is configured on a no-escalate preset — run it off the event
    # loop thread like every other blocking cascade call, so a verifier call
    # never freezes the whole aiohttp server for other concurrent requests.
    turn = await asyncio.to_thread(
        _cascade_after_tier1,
        common,
        messages,
        tier1_bundle,
        agg_text=agg_text,
        agg_usage=agg_usage,
        agg_runtime=runtime,
        agg_response=None,
        agg_messages=agg_messages,
    )
    await send_chunk(None, finish_reason=finish_reason or "stop")
    return turn, (agg_text or None)


async def _stream_cascade_turn(
    common: dict,
    messages: list,
    reference_models: list,
    *,
    send_chunk,
    send_reasoning,
    resp: "web.StreamResponse",
    loop: asyncio.AbstractEventLoop,
    abort: threading.Event,
    include_usage: bool,
) -> None:
    """Streaming cascade turn (addendum v1.4 §B): a live voter fan-out with
    progress reasoning deltas, then the exact same tier-0/1/2 gate
    `_run_cascade_turn` uses (`_cascade_tier0_gate` / `_cascade_after_tier1`
    / `_run_cascade_tier1_blocking`), surfaced to the client per the
    addendum's rules. Writes SSE chunks via ``send_chunk``/``send_reasoning``
    and the ``[DONE]`` sentinel directly; does not call ``resp.write_eof()``
    — the caller (`_handle_streaming`) owns that so its abort/finally
    handling stays identical for every streaming path.
    """
    await send_reasoning(f"[cascade: {len(reference_models)} voters answering…]\n")

    voter_queue: asyncio.Queue = asyncio.Queue()

    def _on_voter_done(label: str) -> None:
        loop.call_soon_threadsafe(voter_queue.put_nowait, label)

    def _voters_worker():
        try:
            return _run_cascade_voters_with_progress(
                reference_models,
                _cascade_voter_view(common, messages),
                temperature=common["reference_temperature"],
                max_tokens=common["reference_max_tokens"] or common["max_tokens"],
                timeout=common["slot_timeout"],
                quorum_grace=common["preset"].get("reference_quorum_grace"),
                on_complete=_on_voter_done,
            )
        finally:
            loop.call_soon_threadsafe(voter_queue.put_nowait, _DONE)

    voters_future = loop.run_in_executor(None, _voters_worker)
    while True:
        item = await voter_queue.get()
        if item is _DONE:
            break
        await send_reasoning(f"\n\n[voter {item}: done]")
    voters = await voters_future

    # `_cascade_tier0_gate` can perform blocking I/O of its own (a judge-gate
    # `call_llm`, and/or a verifier `call_llm` + sandboxed subprocess) when
    # those features are configured — run it off the event loop thread like
    # every other blocking cascade call in this turn (the voter fan-out above
    # and the tier-1/2 blocking call below), so a judge/verifier call never
    # freezes the whole aiohttp server for other concurrent requests.
    gate = await asyncio.to_thread(_cascade_tier0_gate, common, messages, voters)

    if gate.get("result") is not None:
        # Tier 0: emit the winning voter's full text as ONE content delta —
        # no live token-by-token streaming, the answer is already complete.
        turn = gate["result"]
        await send_chunk({"content": turn["winner_text"]})
        await send_chunk(None, finish_reason="stop")
        if include_usage:
            await send_chunk(None, usage=_cascade_stream_usage(turn, common), empty_choices=True)
        await resp.write(b"data: [DONE]\n\n")
        _save_proxy_trace(
            common,
            turn["reference_outputs"],
            turn["agg_messages"],
            turn["winner_text"],
            acting_slot=turn.get("acting_slot"),
        )
        return

    tier1_bundle = gate["tier1"]
    if common.get("cascade_escalate_preset"):
        # The tier-1 (and possible tier-2) answer must be inspected before
        # the client sees it, so both run non-streaming; only the chosen
        # final text is ever emitted to the client, as one content delta.
        turn = await asyncio.to_thread(
            _run_cascade_tier1_blocking, common, messages, tier1_bundle
        )
        message = _extract_message_fields(turn["response"])
        finish_reason = _finish_reason(turn["response"])
        await send_chunk({"content": message.get("content")})
        await send_chunk(None, finish_reason=finish_reason)
        if include_usage:
            await send_chunk(None, usage=_cascade_stream_usage(turn, common), empty_choices=True)
        await resp.write(b"data: [DONE]\n\n")
        _save_proxy_trace(
            common,
            turn["reference_outputs"],
            turn["agg_messages"],
            message.get("content"),
            acting_slot=turn.get("acting_slot"),
        )
        return

    # No escalate preset configured (e.g. a clean-arbiter preset): stream the
    # acting aggregator live, token by token.
    turn, streamed_text = await _stream_cascade_tier1_live(
        common,
        messages,
        tier1_bundle,
        send_chunk=send_chunk,
        send_reasoning=send_reasoning,
        loop=loop,
        abort=abort,
    )
    if include_usage:
        await send_chunk(None, usage=_cascade_stream_usage(turn, common), empty_choices=True)
    await resp.write(b"data: [DONE]\n\n")
    _save_proxy_trace(
        common,
        turn["reference_outputs"],
        turn["agg_messages"],
        streamed_text,
        acting_slot=turn.get("acting_slot"),
    )


# ---------------------------------------------------------------------------
# Streaming tool-turn VOTER GATE (addendum v1.5, docs/plans/moa-cascade-spec.md)
# ---------------------------------------------------------------------------


async def _stream_cascade_tool_turn_gate(
    common: dict,
    messages: list,
    reference_models: list,
    *,
    send_chunk,
    send_reasoning,
    resp: "web.StreamResponse",
    loop: asyncio.AbstractEventLoop,
    abort: threading.Event,
    include_usage: bool,
) -> dict | None:
    """Addendum v1.5 VOTER GATE (streaming): a tool-carrying cascade request
    whose last non-system message is "user" (`_cascade_bypass_mode` already
    ruled out the immediate solo bypasses) runs the tier-0 voter fan-out
    LIVE — the same progress reasoning deltas as the tool-free streaming
    cascade turn (`_stream_cascade_turn`) — with ONE extra tool-awareness
    system line appended to what each voter sees (direct-mode voters get no
    advisory prompt otherwise, so it rides at the end of the client
    messages), then applies the shared `_cascade_tool_turn_gate` decision.

    Returns ``None`` when the voters converged on a REAL answer and the
    session has reverted to cascade — the tier-0 winner text was already
    emitted as one content delta, the final chunk/usage written, and the
    trace saved (mirrors `_stream_cascade_turn`'s tier-0 branch). Otherwise
    returns ``{"reference_outputs": [...], "reason": ..., "extra": {...}}``
    so the caller (`_handle_streaming`) can fall through to the generic
    aggregator-streaming block with tools forwarded, the voter fan-out
    already billed, and no guidance attached — the acting model streams
    live with tools exactly like any other tool-carrying request.
    """
    await send_reasoning(f"[cascade: {len(reference_models)} voters answering…]\n")

    tool_names = _cascade_tool_names(common["tools"])
    voter_messages = _cascade_voter_view(common, messages) + [
        {"role": "system", "content": _cascade_tool_awareness_line(tool_names)}
    ]

    voter_queue: asyncio.Queue = asyncio.Queue()

    def _on_voter_done(label: str) -> None:
        loop.call_soon_threadsafe(voter_queue.put_nowait, label)

    def _voters_worker():
        try:
            return _run_cascade_voters_with_progress(
                reference_models,
                voter_messages,
                temperature=common["reference_temperature"],
                max_tokens=common["reference_max_tokens"] or common["max_tokens"],
                timeout=common["slot_timeout"],
                quorum_grace=common["preset"].get("reference_quorum_grace"),
                on_complete=_on_voter_done,
            )
        finally:
            loop.call_soon_threadsafe(voter_queue.put_nowait, _DONE)

    voters_future = loop.run_in_executor(None, _voters_worker)
    while True:
        item = await voter_queue.get()
        if item is _DONE:
            break
        await send_reasoning(f"\n\n[voter {item}: done]")
    voters = await voters_future

    gate = await asyncio.to_thread(_cascade_tool_turn_gate, common, messages, voters)

    if gate.get("result") is not None:
        turn = gate["result"]
        await send_chunk({"content": turn["winner_text"]})
        await send_chunk(None, finish_reason="stop")
        if include_usage:
            await send_chunk(None, usage=_cascade_stream_usage(turn, common), empty_choices=True)
        await resp.write(b"data: [DONE]\n\n")
        _save_proxy_trace(
            common,
            turn["reference_outputs"],
            turn["agg_messages"],
            turn["winner_text"],
            acting_slot=turn.get("acting_slot"),
        )
        return None

    solo = gate["solo"]
    return {
        "reference_outputs": solo["reference_outputs"],
        "reason": solo["reason"],
        "extra": solo.get("extra") or {},
    }


# ---------------------------------------------------------------------------
# The aiohttp application
# ---------------------------------------------------------------------------


def create_moa_app(*, api_key: str | None = None) -> "web.Application":
    """Build the OpenAI-compatible MoA application.

    ``api_key``: when set, every /v1 request must carry
    ``Authorization: Bearer <api_key>``; /health stays open.
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError(
            "aiohttp is required for `hermes moa serve`. Install with: "
            "pip install 'hermes-agent[messaging]' or `pip install aiohttp`."
        )

    app = web.Application()

    def _check_auth(request: "web.Request"):
        if not api_key:
            return None
        header = request.headers.get("Authorization", "")
        if header.strip() == f"Bearer {api_key}":
            return None
        return _json_error(
            401, "Invalid or missing API key.", err_type="authentication_error"
        )

    async def handle_health(request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "service": "hermes-moa-proxy"})

    async def handle_models(request: "web.Request") -> "web.Response":
        denied = _check_auth(request)
        if denied is not None:
            return denied
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config((load_config() or {}).get("moa") or {})
        data = []
        router_cfg = cfg.get("router") or {}
        if router_cfg.get("enabled"):
            data.append(
                {
                    "id": "moa:auto",
                    "object": "model",
                    "created": 0,
                    "owned_by": "hermes-moa",
                    "moa": {
                        "router": True,
                        "classifier": _slot_label(router_cfg.get("classifier") or {}),
                        "routable_presets": router_cfg.get("routable_presets") or [],
                        "self_answer": bool(router_cfg.get("self_answer")),
                        "default": router_cfg.get("default"),
                    },
                }
            )
        for name, preset in cfg["presets"].items():
            if not preset.get("enabled", True):
                continue
            data.append(
                {
                    "id": f"moa:{name}",
                    "object": "model",
                    "created": 0,
                    "owned_by": "hermes-moa",
                    # Non-standard but useful metadata; OpenAI clients ignore it.
                    "moa": {
                        "default": name == cfg["default_preset"],
                        "aggregator": _slot_label(preset.get("aggregator") or {}),
                        "references": [
                            _slot_label(s) for s in preset.get("reference_models") or []
                        ],
                    },
                }
            )
        return web.json_response({"object": "list", "data": data})

    async def handle_chat_completions(request: "web.Request") -> "web.StreamResponse":
        denied = _check_auth(request)
        if denied is not None:
            return denied

        try:
            body = await request.json()
        except Exception:
            return _json_error(400, "Request body must be valid JSON.")
        if not isinstance(body, dict):
            return _json_error(400, "Request body must be a JSON object.")

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return _json_error(400, "'messages' must be a non-empty array.")

        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config, resolve_moa_preset
        from hermes_cli.proxy import moa_router

        config = load_config() or {}
        session_id = request.headers.get("x-hermes-session-id") or None
        routing: moa_router.RouteDecision | None = None

        if moa_router.is_auto_model(body.get("model")):
            cfg_norm = normalize_moa_config(config.get("moa") or {})
            router_cfg = cfg_norm.get("router") or {}
            if not router_cfg.get("enabled"):
                return _json_error(
                    404,
                    "moa:auto requires moa.router.enabled with a classifier "
                    "slot and at least one preset carrying route.description.",
                    code="model_not_found",
                )
            routing = await moa_router.route_request(
                cfg_norm, messages, session_id=session_id
            )
            if routing.is_self:
                preset_name = moa_router.SELF_CLASS
                preset = moa_router.self_answer_preset(router_cfg)
            else:
                preset_name = routing.preset_name
                preset = cfg_norm["presets"].get(preset_name)
                if preset is None:  # stale sticky entry after a config change
                    preset_name = cfg_norm["default_preset"]
                    preset = cfg_norm["presets"][preset_name]
        else:
            try:
                preset_name = resolve_preset_name(body.get("model"), config)
                preset = resolve_moa_preset(config.get("moa") or {}, preset_name)
            except KeyError as exc:
                return _json_error(
                    404,
                    f"Unknown MoA preset {exc}. See GET /v1/models for available presets.",
                    code="model_not_found",
                )

        reference_models = list(preset.get("reference_models") or [])
        if not preset.get("enabled", True):
            reference_models = []
        aggregator = preset.get("aggregator") or {}
        reference_temperature = float(preset.get("reference_temperature", 0.6) or 0.6)
        aggregator_temperature = float(
            preset.get("aggregator_temperature", body.get("temperature") or 0.4) or 0.4
        )
        reference_max_tokens = preset.get("reference_max_tokens")

        extra_body: dict[str, Any] = {}
        for field in _PASSTHROUGH_FIELDS:
            if body.get(field) is not None:
                extra_body[field] = body[field]

        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        model_name = f"moa:{preset_name}"

        from hermes_cli.moa_config import normalize_moa_config as _norm_cfg

        slot_timeout = float(
            _norm_cfg((config or {}).get("moa") or {}).get("slot_timeout_s") or 0
        )

        # Cascade mode (see docs/plans/moa-cascade-spec.md): pre-resolve the
        # escalate-to preset once here (config lookups don't belong in the
        # hot per-tier turn logic). None whenever mode != "cascade", when no
        # escalate_to is configured, or when it names an unknown preset (the
        # config-normalization post-pass already nulls the latter, but a
        # stale/hand-built preset dict could still reach here).
        cascade_cfg = preset.get("cascade")
        cascade_escalate_preset = None
        if cascade_cfg and cascade_cfg.get("escalate_to"):
            try:
                cascade_escalate_preset = resolve_moa_preset(
                    config.get("moa") or {}, cascade_cfg["escalate_to"]
                )
            except KeyError:
                cascade_escalate_preset = None

        session_info = _session_registry.resolve(session_id, messages)

        common = {
            "routing": routing,
            "session_info": session_info,
            "slot_timeout": slot_timeout if slot_timeout > 0 else None,
            "request_id": request_id,
            "created": created,
            "model_name": model_name,
            "preset_name": preset_name,
            "preset": preset,
            "reference_models": reference_models,
            "aggregator": aggregator,
            "reference_temperature": reference_temperature,
            "aggregator_temperature": aggregator_temperature,
            "reference_max_tokens": reference_max_tokens,
            "messages": messages,
            "tools": body.get("tools"),
            "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
            "extra_body": extra_body,
            "session_id": session_id,
            "cascade": cascade_cfg,
            "cascade_escalate_preset": cascade_escalate_preset,
        }

        if body.get("stream"):
            include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            return await _handle_streaming(request, common, include_usage=include_usage)
        return await _handle_non_streaming(common)

    # ------------------------------------------------------------------
    # Non-streaming turn
    # ------------------------------------------------------------------

    async def _handle_non_streaming(common: dict) -> "web.Response":
        from agent.usage_pricing import CanonicalUsage

        def _run_turn():
            messages = common["messages"]
            reference_models = common["reference_models"]

            if common["preset"].get("mode") == "cascade":
                bypass = _cascade_bypass_mode(common)
                if bypass:
                    # Addendum v1.4 §A (+context guard) / v1.5 mid-loop:
                    # tool-carrying or oversized requests run the acting
                    # model solo — no voters, no guidance.
                    return _cascade_solo_turn_for_bypass(common, bypass)
                if common["tools"] and reference_models:
                    # Addendum v1.5 VOTER GATE: tools present, tool_turns ==
                    # "detect", and the last non-system message is "user" —
                    # `_cascade_bypass_mode` returned None precisely for
                    # this case. Ask the voters whether the turn even needs
                    # tools before falling back to acting-solo.
                    return _run_cascade_tool_turn_gate(common)
                # Cascade mode (docs/plans/moa-cascade-spec.md) only applies
                # to tool-free turns on a preset with >=2 reference slots
                # (config normalization guarantees the slot count whenever
                # mode == "cascade"; a disabled preset empties
                # reference_models, which falls through to the existing
                # fanout path below).
                if reference_models:
                    return _run_cascade_turn(common)

            draft_review = (
                common["preset"].get("mode") == "draft_review" and reference_models
            )
            draft_text = None
            if draft_review:
                # Inverted MoA: the aggregator drafts SOLO first (preserving
                # solo precision on exact-format work), references then only
                # REVIEW the draft, and the aggregator revises. Motivated by
                # the measured pass@1 drop when advisory context precedes
                # precise code edits.
                draft_response = call_llm(
                    task="moa_aggregator",
                    messages=_maybe_apply_moa_cache_control(
                        [dict(m) for m in messages], _slot_runtime(common["aggregator"])
                    ),
                    temperature=common["aggregator_temperature"],
                    max_tokens=common["max_tokens"],
                    extra_body=common["extra_body"] or None,
                    timeout=common["slot_timeout"],
                    **_slot_runtime(common["aggregator"]),
                )
                draft_text = _extract_message_fields(draft_response).get("content") or ""

            ref_messages = _reference_messages(messages)
            if draft_review:
                ref_messages = ref_messages + [
                    {
                        "role": "user",
                        "content": (
                            "[Draft answer under review]\n"
                            "The acting model produced the draft below. Review it "
                            "critically: identify concrete bugs, spec violations, or "
                            "missed requirements, citing the exact spot. If it is "
                            "correct, say APPROVE and nothing else. Do NOT rewrite "
                            "the whole answer.\n\n" + draft_text
                        ),
                    }
                ]
            cache_key = _advisory_signature(
                common["preset_name"], ref_messages, reference_models
            )
            reference_outputs = _ref_cache_get(cache_key)
            refs_from_cache = reference_outputs is not None
            if reference_outputs is None:
                reference_outputs = _run_references_parallel(
                    reference_models,
                    ref_messages,
                    temperature=common["reference_temperature"],
                    max_tokens=common["reference_max_tokens"],
                    timeout=common["slot_timeout"],
                    quorum_grace=common["preset"].get("reference_quorum_grace"),
                )
                _ref_cache_put(cache_key, reference_outputs)

            agg_messages = [dict(m) for m in messages]
            if reference_outputs:
                if draft_review:
                    joined = "\n\n".join(
                        f"Reviewer {idx} — {label}:\n{text}"
                        for idx, (label, text, _acct) in enumerate(reference_outputs, start=1)
                    )
                    guidance = (
                        "[Draft-review context]\n"
                        "You drafted the answer below; independent reviewers then "
                        "checked it. If every reviewer approved, return the draft "
                        "essentially unchanged. Otherwise fix ONLY the concrete "
                        "problems reviewers identified — do not rewrite working "
                        "parts.\n\n[Your draft]\n" + (draft_text or "") + "\n\n" + joined
                    )
                else:
                    guidance = _reference_guidance(
                        common["preset_name"], common["aggregator"], reference_outputs
                    )
                _attach_reference_guidance(agg_messages, guidance)
            response = call_llm(
                task="moa_aggregator",
                messages=_maybe_apply_moa_cache_control(
                    agg_messages, _slot_runtime(common["aggregator"])
                ),
                temperature=common["aggregator_temperature"],
                max_tokens=common["max_tokens"],
                tools=common["tools"],
                extra_body=common["extra_body"] or None,
                timeout=common["slot_timeout"],
                **_slot_runtime(common["aggregator"]),
            )
            runtime = _slot_runtime(common["aggregator"])
            agg_usage = _normalize_chunk_usage(getattr(response, "usage", None), runtime)
            return {
                "reference_outputs": reference_outputs,
                "refs_from_cache": refs_from_cache,
                "agg_messages": agg_messages,
                "response": response,
                "agg_usage": agg_usage,
                "cascade": None,
                "winner_text": None,
            }

        try:
            turn = await asyncio.to_thread(_run_turn)
        except Exception as exc:
            logger.warning("MoA proxy turn failed: %s", exc)
            return _json_error(502, f"MoA aggregator call failed: {exc}", err_type="api_error")

        reference_outputs = turn["reference_outputs"]
        refs_from_cache = turn["refs_from_cache"]
        agg_messages = turn["agg_messages"]
        response = turn["response"]
        agg_usage = turn["agg_usage"]

        if response is not None:
            message = _extract_message_fields(response)
            finish_reason = _finish_reason(response)
        else:
            # Cascade tier 0: a voter's own text IS the final answer — no
            # aggregator ever ran for this turn.
            message = {"role": "assistant", "content": turn.get("winner_text")}
            finish_reason = "stop"

        ref_usage = CanonicalUsage()
        if not refs_from_cache:
            for _label, _text, acct in reference_outputs:
                if isinstance(getattr(acct, "usage", None), CanonicalUsage):
                    ref_usage = ref_usage + acct.usage
        usage = _usage_to_openai(agg_usage + ref_usage)
        usage["moa"] = _usage_breakdown(
            reference_outputs, agg_usage, refs_from_cache, common.get("routing")
        )
        if turn.get("cascade") is not None:
            usage["moa"]["cascade"] = turn["cascade"]
        _attach_session_usage(usage, common, turn.get("cascade"))

        _save_proxy_trace(
            common,
            reference_outputs,
            agg_messages,
            message.get("content"),
            acting_slot=turn.get("acting_slot"),
        )

        return web.json_response(
            {
                "id": common["request_id"],
                "object": "chat.completion",
                "created": common["created"],
                "model": common["model_name"],
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": usage,
            }
        )

    # ------------------------------------------------------------------
    # Streaming turn
    # ------------------------------------------------------------------

    async def _handle_streaming(
        request: "web.Request", common: dict, *, include_usage: bool
    ) -> "web.StreamResponse":
        from agent.usage_pricing import CanonicalUsage

        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
        await resp.prepare(request)

        first_delta_sent = False

        async def send_chunk(
            delta: dict | None = None,
            *,
            finish_reason: str | None = None,
            usage: dict | None = None,
            empty_choices: bool = False,
        ) -> None:
            nonlocal first_delta_sent
            if delta is not None and not first_delta_sent:
                delta = {"role": "assistant", **delta}
                first_delta_sent = True
            payload: dict[str, Any] = {
                "id": common["request_id"],
                "object": "chat.completion.chunk",
                "created": common["created"],
                "model": common["model_name"],
                "choices": []
                if empty_choices
                else [
                    {
                        "index": 0,
                        "delta": delta if delta is not None else {},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            if usage is not None:
                payload["usage"] = usage
            await resp.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))

        async def send_reasoning(text: str) -> None:
            await send_chunk({"reasoning": text, "reasoning_content": text})

        abort = threading.Event()
        loop = asyncio.get_running_loop()

        if common.get("routing") is not None:
            routing = common["routing"]
            await send_reasoning(
                f"[moa:auto → '{routing.preset_name}' — {routing.reason}]\n"
            )

        messages = common["messages"]
        reference_models = common["reference_models"]

        # Addendum v1.4 (docs/plans/moa-cascade-spec.md): real-world cascade
        # streaming. A tool-carrying request on a cascade preset runs the
        # acting aggregator SOLO (§A) — forcing reference_models empty makes
        # the generic fan-out/guidance code below a no-op, so only the final
        # usage surface needs the tool-solo marker (see the include_usage
        # block further down). A tool-free cascade request with >=2
        # reference slots gets a dedicated streaming turn (§B) that never
        # falls through to the generic path below. Addendum v1.5 adds a
        # third possibility for a tool-carrying request whose last
        # non-system message is "user": the VOTER GATE
        # (`_stream_cascade_tool_turn_gate`) decides per-turn whether the
        # session reverts to cascade or falls through to the generic path.
        cascade_mode = common["preset"].get("mode") == "cascade"
        cascade_bypass = _cascade_bypass_mode(common) if cascade_mode else None
        cascade_tool_solo = bool(cascade_bypass)
        cascade_streaming = (
            cascade_mode
            and not cascade_bypass
            and not common["tools"]
            and bool(reference_models)
        )
        cascade_tool_turn_gate = (
            cascade_mode
            and not cascade_bypass
            and bool(common["tools"])
            and bool(reference_models)
        )
        cascade_usage_extra: dict | None = None
        if cascade_tool_solo:
            reference_models = []
            if cascade_bypass == "tool-solo-mid-loop":
                cascade_usage_extra = {
                    "tier": None,
                    "mode": "tool-solo",
                    "reason": "mid-loop",
                }
            else:
                cascade_usage_extra = {"tier": None, "mode": cascade_bypass}

        if cascade_streaming:
            try:
                await _stream_cascade_turn(
                    common,
                    messages,
                    reference_models,
                    send_chunk=send_chunk,
                    send_reasoning=send_reasoning,
                    resp=resp,
                    loop=loop,
                    abort=abort,
                    include_usage=include_usage,
                )
            except (ConnectionResetError, asyncio.CancelledError):
                abort.set()
                raise
            finally:
                abort.set()
            await resp.write_eof()
            return resp

        reference_outputs_preset: list | None = None
        cascade_no_guidance = False

        try:
            if cascade_tool_turn_gate:
                # Addendum v1.5 VOTER GATE.
                outcome = await _stream_cascade_tool_turn_gate(
                    common,
                    messages,
                    reference_models,
                    send_chunk=send_chunk,
                    send_reasoning=send_reasoning,
                    resp=resp,
                    loop=loop,
                    abort=abort,
                    include_usage=include_usage,
                )
                if outcome is None:
                    # Tier 0 hit: the session reverted to cascade and
                    # `_stream_cascade_tool_turn_gate` already wrote the
                    # full SSE response (content delta, finish, usage,
                    # [DONE]) and saved the trace.
                    await resp.write_eof()
                    return resp
                # "solo" outcome (tool_turn vote or no consensus): fall
                # through to the generic aggregator-streaming block below,
                # with the voter fan-out already billed and no guidance
                # attached — the acting model streams live with tools.
                reference_models = []
                reference_outputs_preset = outcome["reference_outputs"]
                cascade_no_guidance = True
                cascade_usage_extra = {
                    "tier": None,
                    "mode": "tool-solo",
                    "reason": outcome["reason"],
                    **outcome["extra"],
                }

            ref_messages = _reference_messages(messages)
            cache_key = _advisory_signature(common["preset_name"], ref_messages, reference_models)
            reference_outputs: list[tuple[str, str, Any]] = list(reference_outputs_preset or [])
            refs_from_cache = False

            if reference_outputs_preset is None:
                cached = _ref_cache_get(cache_key)
                if cached is not None:
                    # Replay the cached advice as reasoning so the client still
                    # sees what the aggregator is acting on — without re-billing.
                    refs_from_cache = True
                    reference_outputs = cached
                    for idx, (label, text, _acct) in enumerate(cached, start=1):
                        await send_reasoning(
                            f"\n\n[Reference {idx}/{len(cached)} — {label} (cached)]\n{text}"
                        )
                elif reference_models:
                    # One queue per reference: each worker streams into its own
                    # queue; we drain queue 0 live while 1..n buffer, then flush
                    # each in order. Live tokens, stable labelled ordering.
                    queues: list[asyncio.Queue] = [asyncio.Queue() for _ in reference_models]

                    def _push(q: asyncio.Queue):
                        def push(text: str) -> None:
                            loop.call_soon_threadsafe(q.put_nowait, text)

                        return push

                    def _worker(slot: dict, q: asyncio.Queue):
                        try:
                            return _reference_stream_worker(
                                slot,
                                ref_messages,
                                temperature=common["reference_temperature"],
                                max_tokens=common["reference_max_tokens"],
                                timeout=common["slot_timeout"],
                                push=_push(q),
                                abort=abort,
                            )
                        finally:
                            loop.call_soon_threadsafe(q.put_nowait, _DONE)

                    futures = [
                        loop.run_in_executor(None, _worker, slot, queue)
                        for slot, queue in zip(reference_models, queues)
                    ]
                    for idx, (slot, queue) in enumerate(zip(reference_models, queues), start=1):
                        await send_reasoning(
                            f"\n\n[Reference {idx}/{len(reference_models)} — {_slot_label(slot)}]\n"
                        )
                        while True:
                            item = await queue.get()
                            if item is _DONE:
                                break
                            await send_reasoning(item)
                    reference_outputs = list(await asyncio.gather(*futures))
                    _ref_cache_put(cache_key, reference_outputs)

            agg_messages = [dict(m) for m in messages]
            if reference_outputs and not cascade_no_guidance:
                _attach_reference_guidance(
                    agg_messages,
                    _reference_guidance(
                        common["preset_name"], common["aggregator"], reference_outputs
                    ),
                )
                await send_reasoning(
                    f"\n\n[Aggregating — {_slot_label(common['aggregator'])} acting on "
                    f"{len(reference_outputs)} reference(s)]\n"
                )

            # Aggregator stream: forwarded via a queue of typed events so the
            # sync SDK iteration lives in a thread and the SSE writes stay on
            # the event loop.
            agg_queue: asyncio.Queue = asyncio.Queue()
            runtime = _slot_runtime(common["aggregator"])

            def _agg_push(kind: str, value: Any) -> None:
                loop.call_soon_threadsafe(agg_queue.put_nowait, (kind, value))

            def _agg_worker():
                try:
                    stream = call_llm(
                        task="moa_aggregator",
                        messages=_maybe_apply_moa_cache_control(agg_messages, runtime),
                        temperature=common["aggregator_temperature"],
                        max_tokens=common["max_tokens"],
                        tools=common["tools"],
                        extra_body=common["extra_body"] or None,
                        timeout=common["slot_timeout"],
                        stream=True,
                        stream_options={"include_usage": True},
                        **runtime,
                    )
                    for chunk in _as_chunk_stream(stream):
                        if abort.is_set():
                            break
                        raw_usage = getattr(chunk, "usage", None)
                        if raw_usage:
                            _agg_push("usage", raw_usage)
                        choices = getattr(chunk, "choices", None) or []
                        if not choices:
                            continue
                        choice = choices[0]
                        delta = getattr(choice, "delta", None)
                        if delta is not None:
                            reasoning = _delta_reasoning_text(delta)
                            if reasoning:
                                _agg_push("reasoning", reasoning)
                            content = getattr(delta, "content", None)
                            if content:
                                _agg_push("content", content)
                            for tc in getattr(delta, "tool_calls", None) or []:
                                _agg_push("tool_call", _tool_call_delta_dict(tc))
                        finish = getattr(choice, "finish_reason", None)
                        if finish:
                            _agg_push("finish", str(finish))
                except Exception as exc:
                    _agg_push("error", str(exc))
                finally:
                    _agg_push("done", None)

            agg_future = loop.run_in_executor(None, _agg_worker)

            finish_reason: str | None = None
            saw_tool_call = False
            agg_raw_usage: Any = None
            content_parts: list[str] = []
            while True:
                kind, value = await agg_queue.get()
                if kind == "done":
                    break
                if kind == "reasoning":
                    await send_reasoning(value)
                elif kind == "content":
                    content_parts.append(value)
                    await send_chunk({"content": value})
                elif kind == "tool_call":
                    saw_tool_call = True
                    await send_chunk({"tool_calls": [value]})
                elif kind == "finish":
                    finish_reason = value
                elif kind == "usage":
                    agg_raw_usage = value
                elif kind == "error":
                    await send_reasoning(f"\n\n[aggregator failed: {value}]")
                    finish_reason = finish_reason or "stop"
            await agg_future

            if finish_reason is None:
                finish_reason = "tool_calls" if saw_tool_call else "stop"
            await send_chunk(None, finish_reason=finish_reason)

            if include_usage:
                agg_usage = _normalize_chunk_usage(agg_raw_usage, runtime)
                ref_usage = CanonicalUsage()
                if not refs_from_cache:
                    for _label, _text, acct in reference_outputs:
                        if isinstance(getattr(acct, "usage", None), CanonicalUsage):
                            ref_usage = ref_usage + acct.usage
                usage = _usage_to_openai(agg_usage + ref_usage)
                usage["moa"] = _usage_breakdown(
                    reference_outputs, agg_usage, refs_from_cache, common.get("routing")
                )
                if cascade_usage_extra is not None:
                    # Addendum v1.4 §A / v1.5: acting-solo turn (config
                    # opt-out, mid-loop, tool_turn vote, or no consensus).
                    usage["moa"]["cascade"] = cascade_usage_extra
                await send_chunk(None, usage=usage, empty_choices=True)

            await resp.write(b"data: [DONE]\n\n")

            _save_proxy_trace(
                common, reference_outputs, agg_messages, "".join(content_parts) or None
            )
        except (ConnectionResetError, asyncio.CancelledError):
            # Client went away: signal workers to stop pulling upstream tokens.
            abort.set()
            raise
        finally:
            abort.set()

        await resp.write_eof()
        return resp

    app.router.add_get("/health", handle_health)
    app.router.add_get("/v1/models", handle_models)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    return app


def _tool_call_delta_dict(tc: Any) -> dict[str, Any]:
    """SDK tool-call delta → OpenAI wire dict, preserving index/id/fragments."""
    fn = getattr(tc, "function", None)
    out: dict[str, Any] = {
        "index": getattr(tc, "index", 0) or 0,
        "type": getattr(tc, "type", None) or "function",
        "function": {},
    }
    tc_id = getattr(tc, "id", None)
    if tc_id:
        out["id"] = tc_id
    name = getattr(fn, "name", None)
    if name:
        out["function"]["name"] = name
    arguments = getattr(fn, "arguments", None)
    if arguments:
        out["function"]["arguments"] = arguments
    return out


def _usage_breakdown(
    reference_outputs: list, agg_usage: Any, refs_from_cache: bool, routing: Any = None
) -> dict:
    """Per-slot usage split for the ``usage.moa`` extension field."""
    refs = []
    for label, _text, acct in reference_outputs:
        usage = getattr(acct, "usage", None)
        refs.append(
            {
                "label": label,
                "cached": refs_from_cache,
                **(_usage_to_openai(usage) if usage is not None else {}),
            }
        )
    out = {"references": refs, "aggregator": _usage_to_openai(agg_usage)}
    if routing is not None:
        out["routed_preset"] = routing.preset_name
        out["routing"] = routing.as_trace()
    return out


def _save_proxy_trace(
    common: dict,
    reference_outputs: list,
    agg_messages: list,
    aggregator_output: str | None,
    acting_slot: dict | None = None,
) -> None:
    """Persist the full proxied MoA turn via the canonical trace writer.

    Gated by ``moa.save_traces`` inside ``save_moa_turn`` — off by default,
    best-effort always. Proxy sessions are keyed by the client-supplied
    ``x-hermes-session-id`` header when present, else a stable hash of the
    first user message so one client conversation lands in one trace file.

    ``acting_slot``: the slot that actually produced the returned text. On
    cascade turns this is a tier-0 voter or the tier-2 escalate aggregator —
    NOT the preset's own aggregator — and `hermes moa evolve` grades traces
    on this attribution, so mislabelling would corrupt distilled heuristics.
    Defaults to the preset aggregator (correct for fanout/draft_review/tier 1).
    """
    try:
        from agent.moa_trace import save_moa_turn

        session_id = common.get("session_id")
        if not session_id:
            first_user = next(
                (
                    m.get("content")
                    for m in common["messages"]
                    if m.get("role") == "user" and isinstance(m.get("content"), str)
                ),
                "",
            )
            digest = hashlib.sha256(first_user.encode("utf-8", "replace")).hexdigest()[:16]
            session_id = f"moa-proxy-{digest}"
        routing = common.get("routing")
        acting = acting_slot or common["aggregator"] or {}
        save_moa_turn(
            session_id=session_id,
            preset_name=common["preset_name"],
            reference_outputs=reference_outputs,
            aggregator_label=_slot_label(acting),
            aggregator_model=acting.get("model"),
            aggregator_provider=acting.get("provider"),
            aggregator_temperature=common["aggregator_temperature"],
            aggregator_input_messages=agg_messages,
            aggregator_output=aggregator_output,
            aggregator_streamed=aggregator_output is None,
            routing=routing.as_trace() if routing is not None else None,
        )
    except Exception as exc:  # pragma: no cover - tracing must never break a turn
        logger.debug("MoA proxy trace write failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI entrypoint — dispatched from `hermes moa serve`
# ---------------------------------------------------------------------------


def cmd_moa_serve(args: Any) -> int:
    """Run the OpenAI-compatible MoA endpoint in the foreground."""
    if not AIOHTTP_AVAILABLE:
        print(
            "hermes moa serve requires aiohttp. Install one of:\n"
            "  pip install 'hermes-agent[messaging]'\n"
            "  pip install aiohttp",
            file=sys.stderr,
        )
        return 1

    import os

    host = getattr(args, "host", None) or DEFAULT_MOA_HOST
    port = getattr(args, "port", None) or DEFAULT_MOA_PORT
    api_key = getattr(args, "api_key", None) or os.environ.get("HERMES_MOA_API_KEY") or None

    if not api_key and host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNING: binding {host} without --api-key exposes your configured "
            "provider credentials to anyone who can reach this port.",
            file=sys.stderr,
        )

    from hermes_cli.config import load_config
    from hermes_cli.moa_config import normalize_moa_config

    cfg = normalize_moa_config((load_config() or {}).get("moa") or {})
    preset_lines = "\n".join(
        f"    moa:{name}"
        + ("  (default)" if name == cfg["default_preset"] else "")
        for name, preset in cfg["presets"].items()
        if preset.get("enabled", True)
    )
    print(
        f"Starting Hermes MoA OpenAI-compatible endpoint\n"
        f"  Listening on: http://{host}:{port}/v1\n"
        f"  Auth:         {'Bearer token required' if api_key else 'none (loopback)'}\n"
        f"  Models (presets):\n{preset_lines}\n"
        f"\n"
        f"Point any OpenAI-compatible client at base_url=http://{host}:{port}/v1 "
        f"with model=moa:<preset>. Tools are executed by the client.\n"
        f"Press Ctrl+C to stop.",
        file=sys.stderr,
    )

    async def _run() -> None:
        app = create_moa_app(api_key=api_key)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, host=host, port=port)
        await site.start()
        try:
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print("\nmoa serve: stopped", file=sys.stderr)
    except OSError as exc:
        print(f"moa serve: failed to bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = [
    "AIOHTTP_AVAILABLE",
    "DEFAULT_MOA_HOST",
    "DEFAULT_MOA_PORT",
    "cmd_moa_serve",
    "create_moa_app",
    "resolve_preset_name",
]
