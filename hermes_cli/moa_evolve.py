"""`hermes moa evolve` — distill MoA trace files into the aggregation skill.

The improvement loop for Mixture of Agents (docs/moa-openai-endpoint.md):
turns recorded by ``moa.save_traces`` are graded offline by an LLM — did the
aggregator follow, correct, or ignore each reference; which references were
right — and the durable lessons are distilled into
``<hermes_home>/skills/moa-aggregation/SKILL.md``. That skill body is injected
into every subsequent aggregator guidance block (``aggregation_skill_block``
in ``agent/moa_loop.py``), so aggregation quality compounds with use.

The distillation is a REWRITE, not an append: the model receives the current
skill body plus a digest of recent turns and returns the full updated
heuristics document, merging duplicates and dropping stale rules. That keeps
the skill bounded (it rides in every MoA prompt) instead of growing without
limit.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How much of each recorded field enters the grading digest. The grader needs
# enough to judge agreement/quality, not full transcripts — full traces are
# there on disk if a human wants them.
_PROMPT_PREVIEW = 500
_REFERENCE_PREVIEW = 1200
_AGGREGATOR_PREVIEW = 1500

# Ceiling for the distilled skill body. It is injected into EVERY MoA
# aggregator prompt, so it must stay cheap; the rewrite prompt asks the model
# to stay under this and we hard-truncate as a backstop.
_SKILL_BODY_MAX_CHARS = 4000

_DISTILL_SYSTEM_PROMPT = (
    "You are the offline grader/distiller for a Mixture of Agents (MoA) "
    "system. You receive (a) the CURRENT distilled aggregation-heuristics "
    "document and (b) a digest of recent MoA turns: what each reference "
    "advisor said and what the acting aggregator did.\n\n"
    "Grade the turns: where references agreed or disagreed, which advice the "
    "aggregator followed or ignored, which references look reliable or "
    "unreliable for which kinds of task, and any recurring aggregation "
    "mistakes (e.g. following a confident-but-wrong reference, discarding a "
    "correct minority view, verbatim-copying instead of synthesizing). Some "
    "turns may carry a 'Graded outcome' line from an external grader — treat "
    "that as ground truth about whether the aggregator's final answer was "
    "right, and mine the incorrect turns hardest for what the aggregator "
    "should have done differently. Distill transferable heuristics about HOW "
    "to aggregate (verification habits, when to trust majority vs minority, "
    "per-model reliability), never task-specific answers or facts.\n\n"
    "Then return the FULL UPDATED heuristics document as plain markdown:\n"
    "- keep still-valid existing heuristics, merge duplicates, drop rules the "
    "new evidence contradicts\n"
    "- each heuristic must be concrete and actionable for an aggregator "
    "deciding how to weigh reference advice — no platitudes\n"
    "- include a '## Reference model notes' section with per-model "
    "reliability notes ONLY where the evidence supports them\n"
    f"- stay under {_SKILL_BODY_MAX_CHARS} characters total\n\n"
    "Return ONLY the markdown body — no frontmatter, no preamble, no code "
    "fences, no commentary about what you changed."
)


def _trace_dir() -> Path:
    from agent.moa_trace import _traces_enabled_and_dir
    from hermes_constants import get_hermes_home

    enabled_dir = _traces_enabled_and_dir()
    if enabled_dir is not None:
        return enabled_dir
    # Tracing may have been enabled earlier and switched off since; evolve
    # should still read whatever was recorded.
    return get_hermes_home() / "moa-traces"


def _load_recent_turns(trace_dir: Path, max_turns: int) -> list[dict[str, Any]]:
    """Newest ``max_turns`` records across every session trace file."""
    records: list[dict[str, Any]] = []
    for path in trace_dir.glob("*.jsonl"):
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        records.append(rec)
        except OSError as exc:
            logger.debug("moa evolve: cannot read %s: %s", path, exc)
    records.sort(key=lambda r: r.get("ts") or 0)
    return records[-max_turns:]


def _clip(text: Any, budget: int) -> str:
    s = str(text or "").strip()
    if len(s) <= budget:
        return s
    return s[:budget] + f" [...{len(s) - budget} chars omitted]"


def _turn_digest(rec: dict[str, Any], idx: int) -> str:
    """Render one trace record as a compact block for the grading prompt."""
    lines = [f"### Turn {idx} (preset: {rec.get('preset') or '?'})"]
    agg = rec.get("aggregator") or {}
    prompt = ""
    for msg in reversed(agg.get("input_messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
            prompt = msg["content"]
            break
    if prompt:
        lines.append(f"User/task (tail): {_clip(prompt, _PROMPT_PREVIEW)}")
    for ref in rec.get("references") or []:
        if not isinstance(ref, dict):
            continue
        lines.append(
            f"- Reference {ref.get('label') or '?'}: "
            f"{_clip(ref.get('output'), _REFERENCE_PREVIEW)}"
        )
    lines.append(
        f"- Aggregator {agg.get('label') or '?'} acted: "
        f"{_clip(agg.get('output') or '(output streamed; not captured)', _AGGREGATOR_PREVIEW)}"
    )
    outcome = rec.get("outcome")
    if isinstance(outcome, dict):
        verdict = "CORRECT" if outcome.get("correct") else "INCORRECT"
        detail = ""
        if outcome.get("correct") is False and outcome.get("expected") is not None:
            detail = (
                f" (expected {_clip(outcome.get('expected'), 80)!r}, "
                f"got {_clip(outcome.get('extracted'), 80)!r})"
            )
        lines.append(f"- Graded outcome: {verdict}{detail}")
    return "\n".join(lines)


def _skill_path() -> Path:
    from agent.moa_loop import AGGREGATION_SKILL_RELPATH
    from hermes_constants import get_hermes_home

    return get_hermes_home().joinpath(*AGGREGATION_SKILL_RELPATH)


def _resolve_distiller_slot(model_arg: str | None) -> dict[str, str]:
    """``--model provider:model`` wins; default is the default preset's
    aggregator (already a model the user trusts to synthesize)."""
    if model_arg:
        provider, sep, model = str(model_arg).partition(":")
        if not sep or not provider.strip() or not model.strip():
            raise SystemExit(
                f"--model must be provider:model (got {model_arg!r})"
            )
        return {"provider": provider.strip(), "model": model.strip()}
    from hermes_cli.config import load_config
    from hermes_cli.moa_config import resolve_moa_preset

    preset = resolve_moa_preset((load_config() or {}).get("moa") or {}, None)
    return dict(preset.get("aggregator") or {})


def _render_skill_file(body: str, turns_analyzed: int) -> str:
    updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return (
        "---\n"
        "name: moa-aggregation\n"
        "description: Distilled heuristics for the MoA aggregator "
        "(auto-updated by `hermes moa evolve`)\n"
        "metadata:\n"
        "  hermes:\n"
        "    auto_generated: moa-evolve\n"
        f"    updated: {updated}\n"
        f"    turns_analyzed: {turns_analyzed}\n"
        "---\n\n"
        f"{body.strip()}\n"
    )


def distill_skill(
    turns: list[dict[str, Any]],
    existing_body: str,
    slot: dict[str, str],
) -> str:
    """One LLM call: grade the turn digest and rewrite the skill body."""
    # Called through the moa_loop module binding (not a direct
    # auxiliary_client import) so the whole MoA stack shares one call_llm
    # seam — tests and instrumentation patch agent.moa_loop.call_llm once.
    from agent import moa_loop

    digest = "\n\n".join(_turn_digest(rec, idx) for idx, rec in enumerate(turns, start=1))
    user_prompt = (
        "## Current heuristics document\n"
        f"{existing_body.strip() or '(empty — this is the first distillation)'}\n\n"
        "## Recent MoA turns\n"
        f"{digest}"
    )
    response = moa_loop.call_llm(
        task="moa_aggregator",
        messages=[
            {"role": "system", "content": _DISTILL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2500,
        **moa_loop._slot_runtime(slot),
    )
    body = moa_loop._extract_text(response)
    if not body:
        raise RuntimeError("distiller returned an empty document")
    # Models occasionally fence the whole document despite instructions.
    stripped = body.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").lstrip("markdown").strip()
    return stripped[:_SKILL_BODY_MAX_CHARS]


# Minimum graded turns a preset needs before it earns its own skill file;
# below this the evidence is too thin and the global skill serves it.
_PER_PRESET_MIN_TURNS = 5


def cmd_moa_evolve(args: Any) -> int:
    """Grade recent MoA traces and update the aggregation skill(s).

    Default: one global skill from all recent turns. ``--per-preset``: group
    turns by the preset that produced them and distill a separate
    ``<preset>.SKILL.md`` per preset with enough evidence — per-route
    heuristics keep one lane's habits (e.g. heavy verification) from taxing
    another lane's turns. Injection precedence: per-preset file, else global.
    """
    from agent.moa_loop import _slot_label, load_aggregation_skill

    trace_dir = Path(getattr(args, "trace_dir", None) or _trace_dir())
    max_turns = int(getattr(args, "max_turns", None) or 30)
    dry_run = bool(getattr(args, "dry_run", False))
    per_preset = bool(getattr(args, "per_preset", False))

    turns = _load_recent_turns(trace_dir, max_turns)
    if not turns:
        print(
            f"No MoA traces found in {trace_dir}.\n"
            "Enable them with config `moa.save_traces: true`, run some MoA "
            "turns, then re-run `hermes moa evolve`."
        )
        return 1

    slot = _resolve_distiller_slot(getattr(args, "model", None))

    groups: list[tuple[str | None, list[dict[str, Any]]]] = [(None, turns)]
    if per_preset:
        by_preset: dict[str, list[dict[str, Any]]] = {}
        for rec in turns:
            by_preset.setdefault(str(rec.get("preset") or "unknown"), []).append(rec)
        groups = [
            (name, recs)
            for name, recs in sorted(by_preset.items())
            if len(recs) >= _PER_PRESET_MIN_TURNS
        ]
        skipped = sorted(
            name for name, recs in by_preset.items() if len(recs) < _PER_PRESET_MIN_TURNS
        )
        if skipped:
            print(
                f"Skipping presets with <{_PER_PRESET_MIN_TURNS} turns "
                f"(global skill serves them): {', '.join(skipped)}"
            )
        if not groups:
            print("No preset has enough turns for per-preset distillation.")
            return 1

    failures = 0
    for preset_name, recs in groups:
        existing_body = load_aggregation_skill(preset_name)
        label = f"preset '{preset_name}'" if preset_name else "the global skill"
        print(
            f"Distilling {len(recs)} MoA turn(s) from {trace_dir} "
            f"with {_slot_label(slot)} for {label}..."
        )
        try:
            body = distill_skill(recs, existing_body, slot)
        except Exception as exc:
            print(f"moa evolve: distillation failed for {label}: {exc}")
            failures += 1
            continue

        rendered = _render_skill_file(body, len(recs))
        if dry_run:
            print(f"\n--- {label} (dry run, not written) ---\n")
            print(rendered)
            continue

        path = _skill_path()
        if preset_name:
            safe = "".join(
                c if (c.isalnum() or c in "-_") else "_" for c in preset_name
            )
            path = path.parent / f"{safe}.SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {path} ({len(body)} chars).")

    if not dry_run and not failures:
        print("Skills are injected into matching MoA aggregator guidance blocks.")
    return 1 if failures == len(groups) else 0


__all__ = ["cmd_moa_evolve", "distill_skill"]
