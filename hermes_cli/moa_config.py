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


def _coerce_float_or_none(value: Any) -> float | None:
    """Coerce to a float, or None when unset/blank/invalid.

    Used for optional sampling params (reference_temperature /
    aggregator_temperature) where None means 'don't send the parameter —
    provider default applies', matching how a single-model Hermes agent
    never sends temperature unless explicitly configured.
    """
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any, default: float) -> float:
    """Coerce to a float with a hard default — for REQUIRED tunables
    (router timeout_s, cascade slot_timeout_s) where 'unset' must still
    yield a working value, unlike the optional-sampling-param semantics of
    ``_coerce_float_or_none``."""
    coerced = _coerce_float_or_none(value)
    return default if coerced is None else coerced


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


def _coerce_fanout(value: Any) -> str:
    """Normalize the fan-out cadence; unknown values fall back to default."""
    mode = str(value or "").strip().lower()
    return mode if mode in {"per_iteration", "user_turn"} else "per_iteration"


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
    out = {"provider": provider, "model": model}
    # Optional per-slot generation cap (overrides the preset-level
    # reference_max_tokens for THIS slot only) — thinking-heavy voters need
    # caps >= their reasoning budget to reliably emit final answers.
    cap = slot.get("max_tokens")
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        cap = None
    if cap and cap > 0:
        out["max_tokens"] = cap
    # Addendum v1.3 (RLM voter slots): opt this slot into the reason -> python
    # -> observe loop (agent/moa_loop.py, cascade direct-voter path only; any
    # other value is ignored). `rlm_rounds` only means anything alongside
    # agent="rlm", so it is only preserved when the agent flag resolved.
    agent = str(slot.get("agent") or "").strip().lower()
    if agent == "rlm":
        out["agent"] = "rlm"
        out["rlm_rounds"] = max(2, min(_coerce_int(slot.get("rlm_rounds"), 6), 12))
    return out


def _default_preset() -> dict[str, Any]:
    return {
        "reference_models": deepcopy(DEFAULT_MOA_REFERENCE_MODELS),
        "aggregator": deepcopy(DEFAULT_MOA_AGGREGATOR),
        # None = temperature omitted from API calls (provider default),
        # matching single-model agent behavior.
        "reference_temperature": None,
        "aggregator_temperature": None,
        "max_tokens": 4096,
        "reference_max_tokens": None,
        "fanout": "per_iteration",
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
    # acts), "draft_review" (inverted: aggregator drafts solo, references
    # review the draft, aggregator revises), or "cascade" (lazy MoA: voters
    # answer directly first, an aggregator only runs on disagreement — see
    # docs/plans/moa-cascade-spec.md). draft_review targets precise code
    # editing, where up-front advisory context measurably hurt pass@1.
    mode = str(raw.get("mode") or "fanout").strip().lower()
    if mode not in {"fanout", "draft_review", "cascade"}:
        mode = "fanout"
    # Cascade needs at least two voters to have a consensus to check; a preset
    # with fewer reference slots silently downgrades to fanout rather than
    # erroring, matching the tolerant-degrade style of the rest of this
    # function (e.g. bad reference_models types above).
    if mode == "cascade" and len(refs) < 2:
        mode = "fanout"

    cascade = None
    if mode == "cascade":
        cascade_raw = raw.get("cascade")
        if not isinstance(cascade_raw, dict):
            cascade_raw = {}
        escalate_to = str(cascade_raw.get("escalate_to") or "").strip() or None
        # Default min_consensus is unanimity (len(refs)); clamp to [2, len(refs)]
        # so a hand-edited value can't require 0/1 votes (trivial) or more votes
        # than there are voters (impossible to ever reach consensus).
        min_consensus = _coerce_int(cascade_raw.get("min_consensus"), len(refs))
        min_consensus = max(2, min(min_consensus, len(refs)))
        # Addendum v1.1 (judge gate): "exact" is the v1 behavior (only literal
        # candidate agreement fires tier 0). "judge" additionally tries one
        # cheap LLM consistency check on freeform (no-candidate-consensus)
        # voter output before falling through to the aggregator. The judge
        # slot itself may be explicit here, or default to the router
        # classifier — but that default can only be resolved once the router
        # block is normalized, so it is finished in `normalize_moa_config`
        # below; here we just record the requested gate and any explicit slot.
        gate = str(cascade_raw.get("gate") or "exact").strip().lower()
        if gate not in {"exact", "judge"}:
            gate = "exact"
        judge = _clean_slot(cascade_raw.get("judge"))
        # Addendum v1.2 (verified cascade): "python" opts a preset into the
        # sandboxed-verifier check on exact-candidate answers (see
        # hermes_cli/proxy/moa_cascade.py:run_verification and
        # moa_server._run_cascade_turn). Any other/absent value is "off" — the
        # verifier slot fallback (explicit -> judge slot -> router classifier)
        # can only be resolved once the router block is normalized, so — like
        # the judge-gate fallback above — it is finished in
        # `normalize_moa_config` below; here we just record the request and
        # any explicit slot.
        verify = str(cascade_raw.get("verify") or "").strip().lower() or None
        if verify != "python":
            verify = None
        verify_when = str(cascade_raw.get("verify_when") or "weak").strip().lower()
        if verify_when not in {"weak", "always"}:
            verify_when = "weak"
        verifier = _clean_slot(cascade_raw.get("verifier"))
        # Voter slots are wafer/local-class models with ~128k contexts; a
        # request bigger than this estimate bypasses them entirely and runs
        # the acting slot solo ("context-solo"), instead of erroring through
        # the whole voter pool. 0 disables the guard.
        max_context_tokens = _coerce_int(cascade_raw.get("max_context_tokens"), 100_000)
        if max_context_tokens < 0:
            max_context_tokens = 100_000
        # Addendum v1.5 (session-aware tool turns): "detect" (default)
        # re-checks EVERY turn whether the client's tools are still in play
        # — a fresh user turn re-engages the voter pool (see
        # moa_server._cascade_bypass_mode / _run_cascade_tool_turn_gate)
        # instead of pinning the whole session to acting-solo. "solo"
        # reproduces the addendum v1.4 behavior unconditionally (every
        # tool-carrying request bypasses voters). Any other/absent value
        # degrades to "detect", matching the tolerant-degrade style used
        # throughout this function.
        tool_turns = str(cascade_raw.get("tool_turns") or "detect").strip().lower()
        if tool_turns not in {"detect", "solo"}:
            tool_turns = "detect"
        # Addendum v1.6 §B (advisor lane): an optional slot that reviews the
        # session ASYNC, after a turn returns, over the same voter-shaped
        # projected+windowed view (see moa_server._run_cascade_advisor). No
        # slot configured (the default) means the advisor never runs — the
        # anchoring law (measured: always-on advisory context hurts precise
        # tool/code work) means this is opt-in, not a cascade default.
        # "notes" (default) only ever injects a note as a tail system
        # message on the NEXT turn; "escalate" additionally lets a BLOCKER
        # note force that next turn past tier-0 straight to tier-1 (see
        # moa_server._cascade_tier0_gate). Unknown/absent degrades to
        # "notes", matching the tolerant-degrade style used throughout this
        # function.
        advisor = _clean_slot(cascade_raw.get("advisor"))
        advisor_mode = str(cascade_raw.get("advisor_mode") or "notes").strip().lower()
        if advisor_mode not in {"notes", "escalate"}:
            advisor_mode = "notes"
        cascade = {
            "escalate_to": escalate_to,
            "max_context_tokens": max_context_tokens,
            "min_consensus": min_consensus,
            "gate": gate,
            "judge": judge,
            "verify": verify,
            "verifier": verifier,
            "verify_when": verify_when,
            "tool_turns": tool_turns,
            "advisor": advisor,
            "advisor_mode": advisor_mode,
            # Serving robustness: when true, min_consensus degrades to the
            # LIVE voter count (floor 2) during partial fan-out outages
            # (quota 429s, provider blips) instead of counting dead voters
            # in the denominator and falling to acting-solo. Default false:
            # benchmark presets keep strict semantics.
            "degraded_consensus": bool(cascade_raw.get("degraded_consensus", False)),
            # Recency window (~tokens) each cascade VOTER sees — the acting
            # lanes always keep the full transcript. Bounds the fan-out
            # token bill on real sessions (measured: full-context fan-out
            # tripped the Cerebras TPM quota after one 10k-context turn;
            # provider prefix caches don't help because quotas count cached
            # tokens). Default 8000; set 0/negative for the old unbounded
            # behavior. Short requests (benchmarks) are unaffected — a
            # history inside the budget passes through untouched.
            "voter_context_tokens": (
                _coerce_int(cascade_raw.get("voter_context_tokens"), 8000)
                if _coerce_int(cascade_raw.get("voter_context_tokens"), 8000) > 0
                else None
            ),
            # When true, the disagreement arbiter (tier-1 aggregator and the
            # tier-2 escalate slot) is called CLEAN — client messages only,
            # no voter context. Measured motivation (2026-07-04): voter
            # context anchors even a frontier arbiter — GPT-5.5 judging
            # disagreements scored 7/9 where GPT-5.5 solo runs ~98%, the
            # same contamination measured on code editing.
            "clean_arbiter": bool(cascade_raw.get("clean_arbiter", False)),
        }

    return {
        "enabled": bool(raw.get("enabled", True)),
        "reference_models": refs,
        "aggregator": aggregator,
        "route": route,
        "mode": mode,
        "cascade": cascade,
        "reference_temperature": _coerce_float_or_none(raw.get("reference_temperature")),
        "aggregator_temperature": _coerce_float_or_none(raw.get("aggregator_temperature")),
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
        # When the reference fan-out runs. "per_iteration" (default) re-runs
        # the advisors whenever the advisory view changes — i.e. every tool
        # iteration, so advice tracks live task state. "user_turn" runs the
        # advisors ONCE per user turn (the original MoA shape): the
        # aggregator gets their upfront plan-level advice, then acts alone
        # for the rest of the tool loop.
        "fanout": _coerce_fanout(raw.get("fanout")),
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
                # Escalate only on the Nth request carrying a failure signal.
                # Debugging workloads print tracebacks as part of NORMAL work
                # (reproducing the bug), so first-failure escalation over-fires
                # (measured: 20/25 SWE conversations escalated); repeated
                # failures indicate the lane is actually stuck.
                "min_failures": max(1, _coerce_int(esc_raw.get("min_failures"), 1)),
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

    # Validate cascade.escalate_to against the resolved presets map, mirroring
    # the router's escalation-tier validation below: a name that doesn't
    # resolve to any preset (typo, removed preset) must not silently point a
    # tier-2 escalation at a KeyError — it degrades to "no escalation" (tier 1
    # is the final answer) instead.
    for preset in presets.values():
        cascade = preset.get("cascade")
        if cascade and cascade.get("escalate_to") not in presets:
            cascade["escalate_to"] = None

    router = normalize_moa_router(raw.get("router"), presets)

    # Addendum v1.1: resolve the judge-gate fallback now that the router is
    # normalized. A preset asking for gate="judge" without its own judge slot
    # borrows the router's classifier (when routing is enabled) — a single
    # small model already paid for and warmed up for classification duty is a
    # natural fit for a second cheap yes/no call. With neither an explicit
    # judge slot nor a usable router, "judge" has nothing to call, so it
    # degrades to "exact" here — the server never has to guess at request
    # time whether a judge slot exists.
    for preset in presets.values():
        cascade = preset.get("cascade")
        if not cascade or cascade.get("gate") != "judge":
            continue
        if cascade.get("judge") is None:
            if router.get("enabled") and router.get("classifier"):
                cascade["judge"] = deepcopy(router["classifier"])
            else:
                cascade["gate"] = "exact"

    # Addendum v1.2: resolve the verifier fallback chain now that both the
    # router AND the judge-gate default above are settled. An explicit
    # `cascade.verifier` slot wins; otherwise a resolved `cascade.judge` slot
    # is reused (already-warm classifier-shaped model, same rationale as the
    # judge-gate default); otherwise the router classifier; otherwise there is
    # nothing to call it with, so `verify` degrades to disabled (None) here —
    # the server never has to guess at request time whether a verifier slot
    # exists.
    for preset in presets.values():
        cascade = preset.get("cascade")
        if not cascade or cascade.get("verify") != "python":
            continue
        if cascade.get("verifier") is None:
            if cascade.get("judge") is not None:
                cascade["verifier"] = deepcopy(cascade["judge"])
            elif router.get("enabled") and router.get("classifier"):
                cascade["verifier"] = deepcopy(router["classifier"])
            else:
                cascade["verify"] = None

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
        "router": router,
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
        "fanout": active.get("fanout", "per_iteration"),
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
