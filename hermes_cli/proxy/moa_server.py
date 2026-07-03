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
from typing import Any, Optional

try:
    from aiohttp import web

    AIOHTTP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via cmd_moa_serve guard
    web = None  # type: ignore[assignment]
    AIOHTTP_AVAILABLE = False

from agent.auxiliary_client import call_llm
from agent.moa_loop import (
    _REFERENCE_SYSTEM_PROMPT,
    _RefAccounting,
    _attach_reference_guidance,
    _reference_messages,
    _run_references_parallel,
    _slot_label,
    _slot_runtime,
    aggregation_skill_block,
)

logger = logging.getLogger(__name__)

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
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


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
        f"{aggregation_skill_block()}\n\n"
        f"{joined}"
    )


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
        for chunk in stream:
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

        common = {
            "routing": routing,
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
            ref_messages = _reference_messages(messages)
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
                _attach_reference_guidance(
                    agg_messages,
                    _reference_guidance(
                        common["preset_name"], common["aggregator"], reference_outputs
                    ),
                )
            response = call_llm(
                task="moa_aggregator",
                messages=agg_messages,
                temperature=common["aggregator_temperature"],
                max_tokens=common["max_tokens"],
                tools=common["tools"],
                extra_body=common["extra_body"] or None,
                timeout=common["slot_timeout"],
                **_slot_runtime(common["aggregator"]),
            )
            return reference_outputs, refs_from_cache, agg_messages, response

        try:
            (
                reference_outputs,
                refs_from_cache,
                agg_messages,
                response,
            ) = await asyncio.to_thread(_run_turn)
        except Exception as exc:
            logger.warning("MoA proxy turn failed: %s", exc)
            return _json_error(502, f"MoA aggregator call failed: {exc}", err_type="api_error")

        message = _extract_message_fields(response)
        runtime = _slot_runtime(common["aggregator"])
        agg_usage = _normalize_chunk_usage(getattr(response, "usage", None), runtime)
        ref_usage = CanonicalUsage()
        if not refs_from_cache:
            for _label, _text, acct in reference_outputs:
                if isinstance(getattr(acct, "usage", None), CanonicalUsage):
                    ref_usage = ref_usage + acct.usage
        usage = _usage_to_openai(agg_usage + ref_usage)
        usage["moa"] = _usage_breakdown(
            reference_outputs, agg_usage, refs_from_cache, common.get("routing")
        )

        _save_proxy_trace(common, reference_outputs, agg_messages, message.get("content"))

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
                        "finish_reason": _finish_reason(response),
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
        ref_messages = _reference_messages(messages)
        cache_key = _advisory_signature(common["preset_name"], ref_messages, reference_models)
        reference_outputs: list[tuple[str, str, Any]] = []
        refs_from_cache = False

        try:
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
            if reference_outputs:
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
                        messages=agg_messages,
                        temperature=common["aggregator_temperature"],
                        max_tokens=common["max_tokens"],
                        tools=common["tools"],
                        extra_body=common["extra_body"] or None,
                        timeout=common["slot_timeout"],
                        stream=True,
                        stream_options={"include_usage": True},
                        **runtime,
                    )
                    for chunk in stream:
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
) -> None:
    """Persist the full proxied MoA turn via the canonical trace writer.

    Gated by ``moa.save_traces`` inside ``save_moa_turn`` — off by default,
    best-effort always. Proxy sessions are keyed by the client-supplied
    ``x-hermes-session-id`` header when present, else a stable hash of the
    first user message so one client conversation lands in one trace file.
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
        save_moa_turn(
            session_id=session_id,
            preset_name=common["preset_name"],
            reference_outputs=reference_outputs,
            aggregator_label=_slot_label(common["aggregator"]),
            aggregator_model=(common["aggregator"] or {}).get("model"),
            aggregator_provider=(common["aggregator"] or {}).get("provider"),
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
