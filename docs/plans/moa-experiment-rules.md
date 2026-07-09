# MoA Experiment — Rules & Iteration Protocol

**This document is the standing contract for the Hermes MoA proxy campaign.
Read it at the START of every iteration and check against it at the END of
every iteration.** (Directive from Hugh, 2026-07-09.)

The public report lives at
`/opt/turq-playground/www/share/moa-next-phase/index.html` →
https://services.turquoisebay.ai/share/moa-next-phase/index.html
Code lives on branch `feature/moa-openai-proxy` of `~/opt/hermes-agent-moa`,
pushed only to the **hughmadden fork** (never NousResearch `origin`).

---

## Purpose

Evolve the `hermes moa serve` OpenAI-compatible proxy and answer, with
measurements, one question: **when is the proxy actually worth using over a
solo frontier API** for real long-running agentic work — smart enough,
accurate enough, fast enough, cost-effective enough?

---

## Start-of-iteration checklist

1. **Read this rules doc** and the report's Executive Summary + the most
   recent section, plus `~/opt/moa-next-phase-STATUS.md`.
2. **Repo integrity gate**: confirm `feature/moa-openai-proxy` is checked out,
   working tree clean, and in sync (`git status -sb`). If HEAD drifted to
   `main` or another branch (it has happened via external `checkout`/`pull`),
   reconcile before committing.
3. **State the hypothesis/goal** for this iteration in one line, and what
   result would confirm or refute it.
4. Confirm the current **iteration number** (latest + 1) and use it
   consistently everywhere.

## End-of-iteration checklist

1. **Whole-report review for correctness** — read the entire report, not just
   the new section. Every iteration.
2. **Rectify corrected findings everywhere.** If this iteration corrected or
   refined an earlier finding, fix EVERY place that stated the old version
   (exec summary, section bodies, roadmap predictions, changelog) — the report
   must not contradict itself.
3. **Refresh the living sections**: Executive Summary (incl. the recipe book),
   the tooling-&-execution tally, and add one Changelog appendix row. Update
   any superseded numbers/counts.
4. **Report hygiene** (see below): exec summary leads, changelog is the
   appendix, tags balanced, nesting valid, renders correctly.
5. **Publish + verify**: write file, `curl` the public URL for HTTP 200, and
   headless-render the URL to confirm it opens on the Executive Summary (not
   the changelog) and the new content is present.
6. **PO Hugh** a one-line status with the report link **in the Pushover `url`
   field** (tappable), not buried in the message body.
7. **Commit + push to the fork** (`git push fork feature/moa-openai-proxy`).
   Never push to `origin` (NousResearch).
8. **Update** `STATUS.md` + memories; leave the next iteration's trigger state.
9. **Re-reference this rules doc** — note in STATUS that the protocol was run.

---

## Report correctness rules

- **Leads with the Executive Summary**; the Changelog is an appendix at the
  end. Never open the report with the changelog.
- **Recipe book stays current**: the "proven configurations" list reflects the
  latest measured winners with their real numbers.
- **No stale counts.** Iteration number, spend, test counts, resolve rates
  must match reality. The Changelog is an honest condensed highlights log — do
  not claim it enumerates every iteration if it doesn't.
- **Internal consistency**: no two sections may state contradictory versions
  of the same finding. A corrected finding is corrected globally.
- **Safe-edit protocol before publishing**: validate tag balance AND nesting
  with a real HTML parser (not just regex counts — `<span\nclass>` breaks
  naive regex), then headless-render to confirm.
- **Never fabricate.** Missing data (e.g. undocumented iterations, un-run
  arms, un-evaluated instances) stays explicitly missing/marked. No invented
  changelog rows, no invented numbers.

## Scientific integrity rules

- **Never predict — observe.** Route/decide on observed signals (consensus,
  test results, verdicts), never predicted difficulty. Predicted-difficulty
  routing failed every time it was tried.
- **Clean vs directional**: label results honestly. Small/biased/unequal-N or
  infra-degraded runs are "directional," not headline numbers. A clean re-run
  supersedes a directional one and the report must say so.
- **Compare on a common evaluable set.** SWE-bench eval harness dep-fetches
  fail deterministically on ~5-6 Lite instances here; report resolved/25
  (empty patch / eval-error = unresolved) rather than a fluctuating
  "completed" subset.
- **Report failures and negatives** as first-class results (a refuted
  prediction is a finding).

## Operational & security rules (standing)

- Secrets via **gopass only**; never in code, config, chat, or logs.
- Push to the **hughmadden fork only**; no upstream PR to NousResearch.
- Public reports **sanitized**: no internal paths, IPs, credentials, or
  family details.
- **Stop experiment containers** when done; keep serving proxies
  (`moa-serve-worker`, `moa-stable`) up. Benchmarks run in containers, never
  host Python.
- **Cerebras is the windowed gate-voter tier only** — never an acting/
  aggregator/classifier lane for per-turn agentic load (RPM/TPM/context caps;
  no higher tier exists to buy). Acting lanes belong on local silicon, kimi,
  or plan-GPT-5.5.
- **q80 local (bench 262k)** is throughput-limited for agentic loops (long
  contexts time out the acting call and hang). Use a fast cloud actor for
  advisor/weak-actor experiments needing reliable completion.

---

## Core measured laws (the running verdict — keep current)

- **moa:auto** (cheap actor + escalate-to-frontier on repeated failure) matches
  solo GPT-5.5 resolve at ~1/4–1/5 the cost — the reason to run the proxy for
  agentic coding. The proxy is *not smarter* than its acting model.
- **Advisor = safety net, not quality lift.** It rescues actors that fail on
  procedure, is dead weight for actors limited by capability, and inline-always
  injection can corrupt a competent actor's tool-call format. Gate and defer by
  default; escalate (swap the model) when you need capability.
- **cascade-rlm** family = the hard-reasoning champion (97% AIME 2026 at wafer
  speed). RLM loop lifts non-thinking models only.
- **Diversity value = independence × competence**; anchoring guard (clean
  arbiter) and observed-signal routing are load-bearing.
