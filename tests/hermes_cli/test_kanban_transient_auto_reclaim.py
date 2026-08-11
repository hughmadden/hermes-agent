"""Provider-transient worker crash auto-reclaim regression coverage."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    conn = kb.connect()
    try:
        yield conn
    finally:
        conn.close()


def _crash(conn, tid: str, pid: int, monkeypatch, log: str | None, **kwargs):
    host = kb._claimer_id().split(":", 1)[0]
    assert kb.claim_task(conn, tid, claimer=f"{host}:test") is not None
    kb._set_worker_pid(conn, tid, pid)
    if log is not None:
        path = kb.worker_log_path(tid)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(log, encoding="utf-8")
    kb._record_worker_exit(pid, 256)
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    return kb.detect_crashed_workers(conn, **kwargs)


def test_transient_5xx_requeues_without_count_and_writes_audit(board, monkeypatch):
    tid = kb.create_task(board, title="provider 500", assignee="default")

    assert _crash(
        board, tid, 91001, monkeypatch,
        "retrying\nAPI call failed after 3 retries: HTTP 500\n",
        failure_limit=1,
    ) == [tid]

    task = kb.get_task(board, tid)
    assert task.status == "ready"
    assert task.consecutive_failures == 0
    assert kb.latest_run(board, tid).outcome == "auto_reclaimed"
    events = kb.list_events(board, tid)
    auto = [event for event in events if event.kind == "auto_reclaim"]
    assert len(auto) == 1
    assert auto[0].payload["cause"] == "http_5xx"
    assert auto[0].payload["failure_counted"] is False
    comments = kb.list_comments(board, tid)
    assert len(comments) == 1
    assert "HTTP 500" in comments[0].body
    assert tid in kb.detect_crashed_workers._last_auto_reclaimed


def test_any_failure_of_auto_reclaimed_retry_blocks(board, monkeypatch):
    tid = kb.create_task(board, title="retry fails", assignee="default")
    _crash(
        board, tid, 91002, monkeypatch,
        "API call failed after 3 retries: HTTP 500\n",
        failure_limit=4,
    )

    _crash(
        board, tid, 91003, monkeypatch,
        "ValueError: genuine task failure\n",
        failure_limit=4,
    )

    task = kb.get_task(board, tid)
    assert task.status == "blocked"
    assert task.consecutive_failures == 1
    gave_up = [event for event in kb.list_events(board, tid) if event.kind == "gave_up"]
    assert gave_up[-1].payload["auto_reclaim_retry_failed"] is True


@pytest.mark.parametrize(
    "log",
    [
        "ValueError: invalid task input\n",
        None,
        "API call failed: HTTP 503\nAuthenticationError: invalid API key\n",
    ],
)
def test_nontransient_missing_or_ambiguous_log_fails_closed(
    board, monkeypatch, log,
):
    tid = kb.create_task(board, title="not transient", assignee="default")

    _crash(board, tid, 92000 + len(kb.list_runs(board, tid)), monkeypatch, log,
           failure_limit=1)

    task = kb.get_task(board, tid)
    assert task.status == "blocked"
    assert task.consecutive_failures == 1
    assert not [event for event in kb.list_events(board, tid)
                if event.kind == "auto_reclaim"]


def test_cooldown_budget_exhaustion_uses_normal_accounting(board, monkeypatch):
    tid = kb.create_task(board, title="cooldown", assignee="default")
    with kb.write_txn(board):
        kb._append_event(
            board, tid, "auto_reclaim",
            {"classification": "provider_transient", "cause": "http_5xx"},
        )

    _crash(
        board, tid, 93001, monkeypatch,
        "API call failed after 3 retries: HTTP 500\n",
        failure_limit=1,
        transient_auto_reclaim_cooldown_seconds=300,
        transient_auto_reclaim_max_per_24h=2,
    )

    assert kb.get_task(board, tid).status == "blocked"
    assert len([event for event in kb.list_events(board, tid)
                if event.kind == "auto_reclaim"]) == 1


def test_daily_budget_exhaustion_uses_normal_accounting(board, monkeypatch):
    tid = kb.create_task(board, title="daily budget", assignee="default")
    now = int(time.time())
    with kb.write_txn(board):
        for created_at in (now - 7200, now - 3600):
            kb._append_event(
                board, tid, "auto_reclaim",
                {"classification": "provider_transient", "cause": "timeout"},
            )
            board.execute(
                "UPDATE task_events SET created_at = ? WHERE id = last_insert_rowid()",
                (created_at,),
            )

    _crash(
        board, tid, 93002, monkeypatch,
        "Request timed out while calling provider\n",
        failure_limit=1,
        transient_auto_reclaim_cooldown_seconds=300,
        transient_auto_reclaim_max_per_24h=2,
    )

    assert kb.get_task(board, tid).status == "blocked"
    assert len([event for event in kb.list_events(board, tid)
                if event.kind == "auto_reclaim"]) == 2
