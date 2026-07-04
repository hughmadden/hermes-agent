"""Pure helpers for cascade-mode MoA (`mode: cascade`) — no I/O.

Lazy MoA with observation-based gating: cheap wafer-speed voters answer a
tool-free request directly and in parallel; when enough of them land on the
same normalized answer, that answer is returned with no aggregator call at
all. See docs/plans/moa-cascade-spec.md for the full tier-0/1/2 semantics;
this module only holds the candidate-extraction and agreement primitives the
server (``hermes_cli/proxy/moa_server.py``) uses to implement them.
"""

from __future__ import annotations

import re
from collections import Counter
from fractions import Fraction

_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+$")
_SLASH_FRAC_RE = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
_LATEX_FRAC_RE = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")

# Runtime boilerplate a failed/dropped/skipped reference leaves as its whole
# output (see agent/moa_loop.py). These must NEVER become answer candidates:
# identical failure notes across voters would otherwise manufacture a false
# tier-0 consensus and return an error string as the client-facing answer.
_BOILERPLATE_PREFIXES = ("[failed:", "[dropped:", "[skipped:", "(empty response")


def _boxed_contents(text: str) -> list[str]:
    """All ``\\boxed{...}`` payloads with BALANCED braces (nesting-aware).

    A naive ``[^}]*`` regex truncates ``\\boxed{\\frac{1}{2}}`` at the first
    inner ``}``, collapsing genuinely different fractions to the same
    fragment — which turns real voter disagreement into a false consensus.
    """
    out = []
    idx = 0
    marker = "\\boxed{"
    while True:
        start = text.find(marker, idx)
        if start == -1:
            break
        depth = 1
        pos = start + len(marker)
        begin = pos
        while pos < len(text) and depth:
            ch = text[pos]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            out.append(text[begin:pos - 1])
            idx = pos
        else:  # unbalanced tail — no more complete boxes
            break
    return out


def _strip_wrapping(s: str) -> str:
    """Strip backticks, markdown emphasis (*_), and a trailing '.'/'!' .

    Repeats until stable so nested/ordered wrapping (e.g. `` `42`. `` — a
    trailing '.' sitting outside a closing backtick) fully unwraps instead of
    stopping after a single outside-in pass.
    """
    s = s.strip()
    prev = None
    while prev != s:
        prev = s
        s = s.strip("`*_")
        while s and s[-1] in ".!":
            s = s[:-1]
        s = s.strip()
    return s


def extract_candidate(text: str) -> str | None:
    """Pull a short candidate answer out of a voter/aggregator response.

    Preference order: the last ``ANSWER: ...`` line (case-insensitive, first
    line of the captured text only) → the last ``\\boxed{...}`` → the last
    non-empty line IF it is short enough (<=80 chars) to plausibly be a bare
    answer rather than prose → None. The winning candidate has surrounding
    backticks/markdown emphasis and trailing punctuation stripped.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.lower().startswith(_BOILERPLATE_PREFIXES):
        # Whole output is a failure/drop/skip note from the fan-out runtime —
        # not an answer, and never allowed to vote.
        return None

    answer_matches = _ANSWER_RE.findall(text)
    if answer_matches:
        candidate = answer_matches[-1].strip().splitlines()[0].strip()
        cleaned = _strip_wrapping(candidate)
        return cleaned or None

    boxed_matches = _boxed_contents(text)
    if boxed_matches:
        cleaned = _strip_wrapping(boxed_matches[-1].strip())
        return cleaned or None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last) <= 80 and not last.lower().startswith(_BOILERPLATE_PREFIXES):
            cleaned = _strip_wrapping(last)
            return cleaned or None

    return None


def normalize_candidate(s: str) -> str:
    """Canonicalize a candidate answer so equivalent forms compare equal.

    Strips whitespace/quotes/``$``, lowercases, and collapses inner
    whitespace. Pure integers (``-?\\d+``) canonicalize to their ``int(...)``
    string form (drops leading zeros/plus signs). ``a/b`` and
    ``\\frac{a}{b}`` reduce to a lowest-terms ``a/b`` via ``fractions.Fraction``.
    Anything else is returned as the cleaned (but otherwise untouched) text.
    """
    s = str(s or "").strip()
    s = s.strip("'\"")
    s = s.replace("$", "")
    s = re.sub(r"\s+", " ", s.strip())
    s = s.lower()

    if _INT_RE.match(s):
        return str(int(s))

    m = _LATEX_FRAC_RE.match(s) or _SLASH_FRAC_RE.match(s)
    if m:
        try:
            frac = Fraction(int(m.group(1)), int(m.group(2)))
            return f"{frac.numerator}/{frac.denominator}"
        except (ValueError, ZeroDivisionError):
            return s

    return s


def consensus(candidates: list[str | None], min_consensus: int) -> str | None:
    """Normalized value of the largest candidate group, if it meets the bar.

    ``None`` candidates are ignored. Groups are keyed by
    ``normalize_candidate``; the largest group's normalized value is returned
    only if its size is >= ``min_consensus``, else ``None``.
    """
    counts: Counter[str] = Counter(
        normalize_candidate(c) for c in candidates if c is not None
    )
    if not counts:
        return None
    value, size = counts.most_common(1)[0]
    return value if size >= min_consensus else None


def agrees(a: str | None, b: str | None) -> bool:
    """True iff both candidates are present and normalize to the same value."""
    if a is None or b is None:
        return False
    return normalize_candidate(a) == normalize_candidate(b)


__all__ = ["extract_candidate", "normalize_candidate", "consensus", "agrees"]
