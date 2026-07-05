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
import subprocess
import sys
import tempfile
from collections import Counter
from fractions import Fraction

_ANSWER_RE = re.compile(r"(?:ANSWER|FINAL)\s*:\s*(.+)", re.IGNORECASE)
_INT_RE = re.compile(r"^-?\d+$")
_SLASH_FRAC_RE = re.compile(r"^(-?\d+)\s*/\s*(-?\d+)$")
_LATEX_FRAC_RE = re.compile(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$")
_PY_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_VERDICT_RE = re.compile(r"VERDICT\s*:\s*(CORRECT|WRONG|UNCHECKABLE)", re.IGNORECASE)
# Wall-clock cap on the sandboxed verifier subprocess (addendum v1.2). The
# verifier LLM is asked for "under 5 seconds of compute"; 12s leaves headroom
# for interpreter startup without letting a runaway script hang a turn.
_VERIFY_SUBPROCESS_TIMEOUT_S = 12

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

    Preference order: the last ``ANSWER: ...`` OR ``FINAL: ...`` line
    (case-insensitive, first line of the captured text only — ``FINAL:`` is
    the addendum v1.3 RLM voter loop's terminator, see
    scripts/moa_rlm_bench.py and agent/moa_loop.py) → the last
    ``\\boxed{...}`` → the last non-empty line IF it is short enough (<=80
    chars) to plausibly be a bare answer rather than prose → None. The
    winning candidate has surrounding backticks/markdown emphasis and
    trailing punctuation stripped.
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
        # RLM voters may stack terminators ("FINAL: ANSWER: 4") when the
        # client's own prompt also demands an ANSWER: line; the outer match
        # then captures the inner prefix as part of the candidate and a
        # mixed RLM/plain voter pool can never reach consensus. Peel any
        # leading ANSWER:/FINAL: prefixes off the captured candidate.
        candidate = re.sub(
            r"^(?:(?:ANSWER|FINAL)\s*:\s*)+", "", candidate, flags=re.IGNORECASE
        )
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


def is_boilerplate(text: str) -> bool:
    """True iff ``text`` is a runtime failure/drop/skip/empty-response note
    (see ``_BOILERPLATE_PREFIXES``) rather than real model output.

    Shared by ``extract_candidate`` (a boilerplate output never becomes an
    answer candidate) and the judge gate (addendum v1.1: only substantive,
    non-boilerplate voter outputs are eligible for the freeform consistency
    check) so both call sites agree on what counts as "real" output.
    """
    return str(text or "").strip().lower().startswith(_BOILERPLATE_PREFIXES)


def extract_python_block(text: str) -> str:
    """First ` ```python ` fenced code block in an LLM reply, or the whole
    reply verbatim when no such fence is present.

    Used by the addendum v1.2 verifier: the verifier LLM is asked to write a
    standalone check script, almost always inside a fenced block; falling
    back to the whole reply keeps a plain (unfenced) script executable too
    instead of discarding it.
    """
    if not text:
        return ""
    match = _PY_FENCE_RE.search(text)
    return match.group(1) if match else text


def run_verification(code: str) -> str:
    """Execute an LLM-generated verification script and return its verdict.

    Runs ``code`` as untrusted, model-generated Python in an isolated
    subprocess: ``-I`` (isolated mode — ignores ``PYTHON*`` env vars and does
    not add the script's directory or the user site-packages to
    ``sys.path``), an empty environment, a scratch temp directory as ``cwd``,
    and a hard wall-clock timeout. Parses the LAST ``VERDICT:\\s*(CORRECT|
    WRONG|UNCHECKABLE)`` line from stdout: ``WRONG`` -> ``"wrong"``,
    ``CORRECT`` -> ``"correct"``. Anything else — no verdict line at all,
    ``UNCHECKABLE``, a non-zero exit, an execution error, or a timeout —
    returns ``"inconclusive"``. Verification is a signal, never a blocker, so
    infrastructure failure must never raise out of this function.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=_VERIFY_SUBPROCESS_TIMEOUT_S,
                env={},
                cwd=tmpdir,
            )
    except Exception:
        return "inconclusive"

    if result.returncode != 0:
        # A check script that crashed after printing a verdict cannot be
        # trusted — the crash may be the very computation the verdict
        # depended on (adversarial-review finding, 2026-07-04).
        return "inconclusive"

    matches = _VERDICT_RE.findall(result.stdout or "")
    if not matches:
        return "inconclusive"
    verdict = matches[-1].upper()
    if verdict == "CORRECT":
        return "correct"
    if verdict == "WRONG":
        return "wrong"
    return "inconclusive"


# Wall-clock cap + output tail for the RLM voter loop's sandboxed python-fence
# execution (addendum v1.3). Same sandbox posture as ``run_verification``
# (isolated interpreter, empty env, scratch cwd) and the same 12s budget, but
# this is a raw-output sibling: the RLM loop feeds the output BACK to the
# model as an "OUTPUT:" turn rather than grading a VERDICT line, mirroring
# ``scripts/moa_rlm_bench.py:run_python``.
_RLM_EXEC_TAIL_CHARS = 1500


def run_rlm_exec(code: str) -> str:
    """Execute one RLM voter's python fence and return its raw output tail.

    Runs ``code`` with the same sandboxing as ``run_verification`` (``-I``,
    empty environment, scratch temp-dir cwd, a hard wall-clock timeout) but —
    unlike ``run_verification`` — does NOT parse a ``VERDICT:`` line. It
    returns the last ``_RLM_EXEC_TAIL_CHARS`` characters of stdout+stderr
    combined, which the addendum v1.3 RLM reference loop (agent/moa_loop.py)
    feeds back to the model verbatim as an ``OUTPUT:`` user turn. Execution
    failures (timeout, exception) are folded into the returned text as a
    bracketed note rather than raised — a broken sandbox must degrade the
    loop's next turn, never crash the reference fan-out.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=_VERIFY_SUBPROCESS_TIMEOUT_S,
                env={},
                cwd=tmpdir,
            )
        out = (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        out = f"[execution timed out after {_VERIFY_SUBPROCESS_TIMEOUT_S}s]"
    except Exception as exc:  # pragma: no cover - defensive
        out = f"[execution error: {exc}]"
    return out[-_RLM_EXEC_TAIL_CHARS:]


__all__ = [
    "extract_candidate",
    "normalize_candidate",
    "consensus",
    "agrees",
    "is_boilerplate",
    "extract_python_block",
    "run_verification",
    "run_rlm_exec",
]
