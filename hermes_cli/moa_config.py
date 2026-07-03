"""Mixture-of-Agents configuration and slash-command helpers."""

from __future__ import annotations

import base64
import json
from copy import deepcopy
from typing import Any

MOA_MARKER_PREFIX = "__HERMES_MOA_TURN_V1__"
DEFAULT_MOA_PRESET_NAME = "default"

DEFAULT_MOA_REFERENCE_MODELS: list[dict[str, str]] = [
    {"provider": "openai-codex", "model": "gpt-5.5"},
    {"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
]

DEFAULT_MOA_AGGREGATOR: dict[str, str] = {
    "provider": "openrouter",
    "model": "anthropic/claude-opus-4.8",
}


def _coerce_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def _coerce_int_or_none(value: Any) -> int | None:
    """Coerce to a positive int, or None when unset/blank/invalid/non-positive.

    Used for optional caps (e.g. reference_max_tokens) where None means
    'no cap' — the safe default that preserves prior uncapped behavior.
    """
    if value is None or value == "":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        try:
            n = int(float(value))
        except (TypeError, ValueError):
            return None
    return n if n > 0 else None


def _clean_slot(slot: Any) -> dict[str, str] | None:
    if not isinstance(slot, dict):
        return None
    provider = str(slot.get("provider") or "").strip()
    model = str(slot.get("model") or "").strip()
    if not provider or not model:
        return None
    # MoA is a virtual provider whose presets are themselves MoA runs. Allowing
    # one as a reference or aggregator slot would create a recursive MoA tree
    # (the runtime guards in moa_loop.py skip references / raise on aggregators,
    # but that surfaces only mid-turn). Reject it here so it can never be saved:
    # an invalid slot is dropped, falling back to the preset's defaults.
    if provider.lower() == "moa":
        return None
    return {"provider": provider, "model": model}


def _default_preset() -> dict[str, Any]:
    return {
        "reference_models": deepcopy(DEFAULT_MOA_REFERENCE_MODELS),
        "aggregator": deepcopy(DEFAULT_MOA_AGGREGATOR),
        "reference_temperature": 0.6,
        "aggregator_temperature": 0.4,
        "max_tokens": 4096,
        "reference_max_tokens": None,
        "enabled": True,
    }


def _normalize_preset(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}

    raw_refs = raw.get("reference_models")
    if not isinstance(raw_refs, list):
        # A hand-edited scalar / single mapping (or a bad type) must degrade to
        # defaults instead of crashing the iteration, mirroring the tolerance
        # for the scalar fields below (reference_temperature / max_tokens).
        raw_refs = [raw_refs] if isinstance(raw_refs, dict) else []
    refs = [_clean_slot(item) for item in raw_refs]
    refs = [item for item in refs if item is not None]
    if not refs:
        refs = deepcopy(DEFAULT_MOA_REFERENCE_MODELS)

    aggregator = _clean_slot(raw.get("aggregator")) or deepcopy(DEFAULT_MOA_AGGREGATOR)

    # Optional router metadata: a one-line description of what this preset is
    # good at, consumed by the `moa:auto` classifier (see moa_router.py).
    # Presets without a route block are simply not routing candidates.
    route_raw = raw.get("route")
    route = None
    if isinstance(route_raw, dict):
        description = str(route_raw.get("description") or "").strip()
        if description:
            route = {"description": description}

    # Turn shape: "fanout" (default — references advise before the aggregator
    # acts) or "draft_review" (inverted: aggregator drafts solo, references
    # review the draft, aggregator revises). draft_review targets precise
    # code editing, where up-front advisory context measurably hurt pass@1.
    mode = str(raw.get("mode") or "fanout").strip().lower()
    if mode not in {"fanout", "draft_review"}:
        mode = "fanout"

    return {
        "enabled": bool(raw.get("enabled", True)),
        "reference_models": refs,
        "aggregator": aggregator,
        "route": route,
        "mode": mode,
        "reference_temperature": _coerce_float(raw.get("reference_temperature"), 0.6),
        "aggregator_temperature": _coerce_float(raw.get("aggregator_temperature"), 0.4),
        "max_tokens": _coerce_int(raw.get("max_tokens"), 4096),
        # Optional cap on how much each reference ADVISOR may generate per turn.
        # None (default) = uncapped: advisors write full-length advice, matching
        # prior behavior so existing presets are unchanged. Set a value (e.g.
        # 600) to make advisors give concise advice — the dominant MoA latency
        # is advisor generation (turn latency correlates ~0.88 with output
        # tokens), and the aggregator only needs the gist of each advisor's
        # judgement, so capping roughly halves per-turn wall time. Does NOT cap
        # the acting aggregator (its output is the user-visible answer).
        "reference_max_tokens": _coerce_int_or_none(raw.get("reference_max_tokens")),
        # Optional straggler dropping for the reference fan-out: once all but
        # one reference are done at elapsed T, the last gets T*grace extra
        # seconds, then is dropped with a labelled note. Turn latency equals
        # the slowest reference, and the measured pathology is one reference
        # taking 3-10x the others. None (default) = wait for all.
        "reference_quorum_grace": (
            float(raw["reference_quorum_grace"])
            if str(raw.get("reference_quorum_grace") or "").replace(".", "", 1).isdigit()
            else None
        ),
    }


def normalize_moa_router(raw: Any, presets: dict[str, Any]) -> dict[str, Any]:
    """Validate the ``moa.router`` block against the normalized presets.

    Returns a dict with ``enabled`` False unless the block is coherent: a
    classifier slot must resolve and at least one preset must carry a
    ``route.description`` (otherwise there is nothing to classify onto).
    ``default`` falls back to the first routable preset when unset/unknown.
    ``self_answer`` (default true) enables the SELF class: trivial requests
    answered directly by ``self_answer_model`` (default: the classifier slot)
    with no reference fan-out — the main latency/cost win of routing.

    A DISABLED preset with a ``route.description`` is still routable: disabled
    means "aggregator acts alone, hidden from /v1/models" — exactly how solo
    lanes are modeled — and routing coding traffic to a strong solo is a
    first-class configuration (measured: the fan-out can hurt precise code
    editing). Only presets without a route description are excluded.
    """
    if not isinstance(raw, dict):
        raw = {}
    classifier = _clean_slot(raw.get("classifier"))
    routable = [
        name
        for name, preset in (presets or {}).items()
        if (preset.get("route") or {}).get("description")
    ]
    default = str(raw.get("default") or "").strip()
    if default not in (presets or {}):
        default = routable[0] if routable else ""
    # Optional failure-gated escalation: when a sticky conversation shows a
    # failure signal (tests failed, traceback, ...) in its latest tool/user
    # message, re-route it to a stronger preset instead of retrying the same
    # lane. Measured motivation: strong-solo-first + frontier-on-failure
    # matched full frontier quality at ~25% of the frontier calls on both
    # aider polyglot and SWE-bench Lite (docs/plans/moa-public-bench-results).
    esc_raw = raw.get("escalation")
    escalation = None
    if isinstance(esc_raw, dict):
        # Tiers: either a single `preset` or an ordered `tiers` list — each
        # NEW failure signal advances the conversation one tier (mid-tier
        # models absorb most escalations at a fraction of frontier price).
        tiers_raw = esc_raw.get("tiers")
        if not isinstance(tiers_raw, list):
            tiers_raw = [esc_raw.get("preset")]
        tiers = [
            str(t).strip()
            for t in tiers_raw
            if str(t or "").strip() and str(t).strip() in (presets or {})
        ]
        if tiers:
            patterns = esc_raw.get("on_patterns")
            if not isinstance(patterns, list) or not patterns:
                patterns = [
                    "FAILED", "FAIL:", "AssertionError", "Traceback (most recent call last)",
                    "tests failed", "test failed", "SyntaxError", "does not pass",
                ]
            escalation = {
                "preset": tiers[0],  # back-compat single-tier view
                "tiers": tiers,
                "on_patterns": [str(p) for p in patterns if str(p).strip()],
            }

    return {
        "enabled": bool(raw.get("enabled", False)) and classifier is not None and bool(routable),
        "classifier": classifier,
        "default": default,
        "self_answer": bool(raw.get("self_answer", True)),
        "self_answer_model": _clean_slot(raw.get("self_answer_model")) or classifier,
        "timeout_s": _coerce_float(raw.get("timeout_s"), 8.0),
        "routable_presets": routable,
        "escalation": escalation,
    }


def normalize_moa_config(raw: Any) -> dict[str, Any]:
    """Return validated MoA config with named presets.

    Backward compatible with the first PR shape where ``moa`` itself contained
    ``reference_models`` and ``aggregator`` directly.
    """
    if not isinstance(raw, dict):
        raw = {}

    presets_raw = raw.get("presets")
    presets: dict[str, dict[str, Any]] = {}
    if isinstance(presets_raw, dict):
        for name, preset in presets_raw.items():
            clean_name = str(name or "").strip()
            if clean_name:
                presets[clean_name] = _normalize_preset(preset)

    # Legacy flat config becomes the default preset.
    if not presets:
        presets[DEFAULT_MOA_PRESET_NAME] = _normalize_preset(raw)

    default_name = str(raw.get("default_preset") or "").strip()
    if not default_name or default_name not in presets:
        default_name = next(iter(presets), DEFAULT_MOA_PRESET_NAME)
    if default_name not in presets:
        presets[default_name] = _default_preset()

    active_name = str(raw.get("active_preset") or "").strip()
    if active_name not in presets:
        active_name = ""

    active = presets[default_name]
    return {
        "default_preset": default_name,
        "active_preset": active_name,
        "presets": presets,
        "router": normalize_moa_router(raw.get("router"), presets),
        # Hard per-upstream-call timeout for proxied MoA turns (seconds). A
        # wedged provider connection must not hang a client request forever;
        # a timed-out reference degrades to a labelled failure note and a
        # timed-out aggregator returns a clean 502. 0/negative disables.
        "slot_timeout_s": _coerce_float(raw.get("slot_timeout_s"), 300.0),
        # Compatibility/flattened view for existing dashboard/desktop callers.
        "reference_models": deepcopy(active["reference_models"]),
        "aggregator": deepcopy(active["aggregator"]),
        "reference_temperature": active["reference_temperature"],
        "aggregator_temperature": active["aggregator_temperature"],
        "max_tokens": active["max_tokens"],
        "reference_max_tokens": active.get("reference_max_tokens"),
        "enabled": active["enabled"],
    }


def list_moa_presets(config: Any) -> list[str]:
    cfg = normalize_moa_config(config)
    return list(cfg["presets"].keys())


def resolve_moa_preset(config: Any, name: str | None = None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    preset_name = str(name or cfg.get("default_preset") or DEFAULT_MOA_PRESET_NAME).strip()
    preset = cfg["presets"].get(preset_name)
    if preset is None:
        raise KeyError(preset_name)
    return deepcopy(preset)


def exact_moa_preset_name(config: Any, text: str) -> str | None:
    """Return the preset name iff ``text`` exactly matches an *enabled* preset.

    Used by the no-explicit-provider switch path (PATH B in
    ``hermes_cli/model_switch.py``) to recognize a bare ``/model <preset>``
    that the user typed without the ``moa:`` prefix. This is an *implicit*
    match, so it must honor the per-preset ``enabled`` opt-out: a user who set
    ``enabled: false`` to disable a preset must not have a plain model switch
    whose name happens to collide with that preset key silently pivot the
    session onto the MoA virtual provider (issue #55187). Explicit selection
    via ``--provider moa`` / the model picker does not go through here, so a
    disabled preset is still reachable when the user explicitly asks for it.
    """
    wanted = str(text or "").strip()
    if not wanted:
        return None
    cfg = normalize_moa_config(config)
    preset = cfg["presets"].get(wanted)
    if preset is None or not preset.get("enabled", True):
        return None
    return wanted


def set_active_moa_preset(config: Any, name: str | None) -> dict[str, Any]:
    cfg = normalize_moa_config(config)
    clean = str(name or "").strip()
    if clean and clean not in cfg["presets"]:
        raise KeyError(clean)
    cfg["active_preset"] = clean
    return cfg


def encode_moa_turn(prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Encode a /moa one-shot turn for frontends that can only send text."""
    payload = {
        "prompt": str(prompt or ""),
        "config": resolve_moa_preset(config or {}, preset),
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).decode("ascii")
    return f"{MOA_MARKER_PREFIX}{encoded}"


def decode_moa_turn(message: Any) -> tuple[str, dict[str, Any] | None]:
    """Decode a hidden /moa one-shot marker."""
    if not isinstance(message, str) or not message.startswith(MOA_MARKER_PREFIX):
        return message, None
    encoded = message[len(MOA_MARKER_PREFIX):].strip()
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except Exception:
        return message, None
    prompt = str(payload.get("prompt") or "")
    return prompt, _normalize_preset(payload.get("config") or {})


def build_moa_turn_prompt(user_prompt: str, config: Any = None, preset: str | None = None) -> str:
    """Build the hidden one-shot payload used by TUI/gateway routing."""
    return encode_moa_turn(user_prompt, config, preset=preset)


def moa_usage() -> str:
    return "Usage: /moa <prompt>  (runs one prompt through the default MoA preset, then restores your model; pick a preset from the model picker to switch for the session)"
