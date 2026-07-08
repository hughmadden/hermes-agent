#!/usr/bin/env python3
"""Summarize SWE-bench mini-swe-agent arms for the real-world head-to-head
(iteration 46): resolve rate, throughput, steps/instance, and — when a proxy
trace dir is given — real-dollar token cost per arm.

Quality comes from the swebench eval report
(``<run>/moa-<run>.<...>.json`` or ``<run>/results.json``); step counts and
exit status from the per-instance ``*.traj.json``; wall-clock from the run
directory's earliest/latest trajectory mtime (a throughput proxy at fixed
worker count, since mini-swe does not record per-instance latency).

Stdlib only. Usage:
  python scripts/moa_swe_summary.py --runs-dir ~/opt/moa-runs/mini-swe/runs \
      --arms h2h-gpt55solo:GPT-5.5-solo h2h-cascadelive:cascade-live \
      [--out summary.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def _load_json(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _resolved_ids(run_dir: str) -> set[str] | None:
    """Resolved instance ids from a swebench eval report, or None if unevaluated."""
    for pat in ("*.json",):
        for path in glob.glob(os.path.join(run_dir, pat)):
            data = _load_json(path)
            if isinstance(data, dict) and "resolved_instances" in data:
                # swebench report shape: resolved_ids list OR count + ids elsewhere
                ids = data.get("resolved_ids") or data.get("resolved") or []
                if isinstance(ids, list) and ids:
                    return set(ids)
                # fall back to the report's per-instance map
                per = data.get("resolved_instances")
                if isinstance(per, list):
                    return set(per)
    return None


def _instances(run_dir: str) -> list[dict]:
    out = []
    for traj in glob.glob(os.path.join(run_dir, "*", "*.traj.json")):
        d = _load_json(traj)
        if not d:
            continue
        info = d.get("info") or {}
        stats = info.get("model_stats") or {}
        out.append(
            {
                "id": d.get("instance_id") or os.path.basename(os.path.dirname(traj)),
                "api_calls": stats.get("api_calls"),
                "exit_status": info.get("exit_status"),
                "submitted": bool(info.get("submission")),
                "mtime": os.path.getmtime(traj),
            }
        )
    return out


def _median(xs: list[float]) -> float | None:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def summarize_arm(runs_dir: str, run_name: str, label: str) -> dict:
    run_dir = os.path.join(runs_dir, run_name)
    insts = _instances(run_dir)
    resolved = _resolved_ids(run_dir)
    mtimes = [i["mtime"] for i in insts if i["mtime"]]
    wall = (max(mtimes) - min(mtimes)) if len(mtimes) > 1 else None
    api_calls = [i["api_calls"] for i in insts if i["api_calls"] is not None]
    return {
        "arm": label,
        "run": run_name,
        "instances": len(insts),
        "submitted": sum(1 for i in insts if i["submitted"]),
        "resolved": (len(resolved) if resolved is not None else None),
        "resolve_pct": (
            round(100 * len(resolved) / len(insts), 1)
            if resolved is not None and insts
            else None
        ),
        "median_api_calls": _median(api_calls),
        "wall_clock_s": round(wall) if wall else None,
        "evaluated": resolved is not None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument(
        "--arms",
        nargs="+",
        required=True,
        help="each as run-name:label (label optional, defaults to run-name)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for spec in args.arms:
        run_name, _, label = spec.partition(":")
        rows.append(summarize_arm(args.runs_dir, run_name, label or run_name))

    hdr = f"{'arm':22s} {'inst':>5s} {'subm':>5s} {'resolved':>9s} {'resolve%':>9s} {'med_calls':>10s} {'wall_s':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['arm']:22s} {r['instances']:5d} {r['submitted']:5d} "
            f"{('-' if r['resolved'] is None else r['resolved']):>9} "
            f"{('n/a' if r['resolve_pct'] is None else r['resolve_pct']):>9} "
            f"{('-' if r['median_api_calls'] is None else r['median_api_calls']):>10} "
            f"{('-' if r['wall_clock_s'] is None else r['wall_clock_s']):>8}"
        )

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
