"""Assignee-level dispatcher quarantine and unknown-assignee diagnostics."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    # A kanban worker inherits the live board pin. Tests must never let that
    # override escape the temporary HERMES_HOME and write synthetic cards to
    # the board that dispatched the test process.
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _all_events(conn, kind: str):
    return conn.execute(
        "SELECT task_id, kind, payload FROM task_events WHERE kind = ? ORDER BY id",
        (kind,),
    ).fetchall()


def _create_task(conn, *, title: str, assignee: str, status: str = "ready") -> str:
    task_id = kb.create_task(conn, title=title, assignee=assignee)
    if status != "ready":
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (status, task_id))
        conn.commit()
    return task_id


def _drive_clean_boot_exit(
    conn,
    task_id: str,
    pid: int,
    monkeypatch,
    *,
    dispatcher_tip: bool = False,
) -> None:
    """Claim a card and reap an immediate rc=0 worker with no kanban calls."""
    host = kb._claimer_id().split(":", 1)[0]
    claimed = kb.claim_task(conn, task_id, claimer=f"{host}:acb-test")
    assert claimed is not None
    if dispatcher_tip:
        with kb.write_txn(conn):
            kb._append_event(conn, task_id, "tip_scratch_workspace", {"message": "tip"})
    kb._set_worker_pid(conn, task_id, pid)
    kb._record_worker_exit(pid, 0)
    monkeypatch.setattr(kb, "_pid_alive", lambda candidate: candidate != pid)
    assert task_id in kb.detect_crashed_workers(conn)


def _drive_clean_exit_after_heartbeat(
    conn, task_id: str, pid: int, monkeypatch
) -> None:
    host = kb._claimer_id().split(":", 1)[0]
    claimed = kb.claim_task(conn, task_id, claimer=f"{host}:acb-test")
    assert claimed is not None
    assert kb.heartbeat_worker(conn, task_id, note="worker reached tool loop")
    kb._set_worker_pid(conn, task_id, pid)
    kb._record_worker_exit(pid, 0)
    monkeypatch.setattr(kb, "_pid_alive", lambda candidate: candidate != pid)
    assert task_id in kb.detect_crashed_workers(conn)


def _seed_closed_run(
    conn,
    task_id: str,
    profile: str,
    outcome: str,
    *,
    duration: int = 1,
) -> int:
    ended = int(time.time())
    cur = conn.execute(
        """
        INSERT INTO task_runs (
            task_id, profile, status, started_at, ended_at, outcome
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (task_id, profile, outcome, ended - duration, ended, outcome),
    )
    conn.commit()
    return int(cur.lastrowid)


def test_assignee_quarantined_after_three_boot_failures_across_two_cards(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "broken"
    )
    conn = kb.connect()
    try:
        first = _create_task(conn, title="first", assignee="broken")
        second = _create_task(conn, title="second", assignee="broken")
        fourth = _create_task(conn, title="fourth", assignee="broken")
        kb.add_notify_sub(
            conn, task_id=fourth, platform="telegram", chat_id="operator"
        )

        # Dispatcher-owned bookkeeping after claim is not worker progress and
        # must not mask the first boot failure on a fresh install.
        _drive_clean_boot_exit(
            conn, first, 981001, monkeypatch, dispatcher_tip=True
        )
        _drive_clean_boot_exit(conn, second, 981002, monkeypatch)
        _drive_clean_boot_exit(conn, first, 981003, monkeypatch)

        spawned = []
        result = kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 777
        )

        assert fourth not in spawned
        assert kb.get_task(conn, fourth).status == "ready"
        assert (fourth, "broken") in result.skipped_assignee_quarantined
        quarantined = _all_events(conn, "assignee_quarantined")
        assert len(quarantined) == 1
        # The newest failed card has no origin subscription. Route the one
        # audit event through a subscribed ready card so the existing gateway
        # notifier can actually produce the required operator alert.
        assert quarantined[0]["task_id"] == fourth
        payload = json.loads(quarantined[0]["payload"])
        assert payload["assignee"] == "broken"
        assert len(payload["run_ids"]) == 3
        assert set(payload["task_ids"]) == {first, second}

        # A second tick remains quiet while the same quarantine is active.
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 778)
        assert len(_all_events(conn, "assignee_quarantined")) == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE body LIKE ?",
            ("%assignee 'broken' quarantined%",),
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_healthy_run_clears_assignee_quarantine(kanban_home, monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "broken"
    )
    conn = kb.connect()
    try:
        first = _create_task(conn, title="first", assignee="broken")
        second = _create_task(conn, title="second", assignee="broken")
        waiting = _create_task(conn, title="waiting", assignee="broken")
        _drive_clean_boot_exit(conn, first, 982001, monkeypatch)
        _drive_clean_boot_exit(conn, second, 982002, monkeypatch)
        _drive_clean_boot_exit(conn, first, 982003, monkeypatch)
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 888)
        assert len(_all_events(conn, "assignee_quarantined")) == 1

        smoke = _create_task(
            conn, title="manual smoke", assignee="broken", status="done"
        )
        healthy_run_id = _seed_closed_run(conn, smoke, "broken", "completed")
        spawned = []
        result = kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 889
        )

        unquarantined = _all_events(conn, "assignee_unquarantined")
        assert len(unquarantined) == 1
        payload = json.loads(unquarantined[0]["payload"])
        assert payload == {
            "assignee": "broken",
            "run_id": healthy_run_id,
            "task_id": smoke,
        }
        assert waiting in spawned
        assert not result.skipped_assignee_quarantined
    finally:
        conn.close()


def test_first_newer_run_with_heartbeat_clears_even_if_later_run_boot_fails(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "broken"
    )
    conn = kb.connect()
    try:
        first = _create_task(conn, title="first", assignee="broken")
        second = _create_task(conn, title="second", assignee="broken")
        waiting = _create_task(conn, title="waiting", assignee="broken")
        _drive_clean_boot_exit(conn, first, 982101, monkeypatch)
        _drive_clean_boot_exit(conn, second, 982102, monkeypatch)
        _drive_clean_boot_exit(conn, first, 982103, monkeypatch)
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 888)

        smoke = _create_task(conn, title="smoke with progress", assignee="broken")
        _drive_clean_exit_after_heartbeat(conn, smoke, 982104, monkeypatch)
        later = _create_task(conn, title="later boot failure", assignee="broken")
        _drive_clean_boot_exit(conn, later, 982105, monkeypatch)

        spawned = []
        kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 889
        )

        events = _all_events(conn, "assignee_unquarantined")
        assert len(events) == 1
        payload = json.loads(events[0]["payload"])
        assert payload["task_id"] == smoke
        assert waiting in spawned
    finally:
        conn.close()


def test_three_failures_after_reset_reopen_quarantine_in_same_tick(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "broken"
    )
    conn = kb.connect()
    try:
        first = _create_task(conn, title="first", assignee="broken")
        second = _create_task(conn, title="second", assignee="broken")
        waiting = _create_task(conn, title="waiting", assignee="broken")
        _drive_clean_boot_exit(conn, first, 982201, monkeypatch)
        _drive_clean_boot_exit(conn, second, 982202, monkeypatch)
        _drive_clean_boot_exit(conn, first, 982203, monkeypatch)
        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 888)

        smoke = _create_task(
            conn, title="healthy smoke", assignee="broken", status="done"
        )
        healthy_run_id = _seed_closed_run(conn, smoke, "broken", "completed")
        third = _create_task(conn, title="third", assignee="broken")
        fourth = _create_task(conn, title="fourth", assignee="broken")
        _drive_clean_boot_exit(conn, third, 982204, monkeypatch)
        _drive_clean_boot_exit(conn, fourth, 982205, monkeypatch)
        _drive_clean_boot_exit(conn, third, 982206, monkeypatch)

        spawned = []
        result = kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 889
        )

        assert waiting not in spawned
        assert (waiting, "broken") in result.skipped_assignee_quarantined
        assert len(_all_events(conn, "assignee_unquarantined")) == 1
        assert len(_all_events(conn, "assignee_quarantined")) == 2
        reset_payload = json.loads(
            _all_events(conn, "assignee_unquarantined")[0]["payload"]
        )
        assert reset_payload["run_id"] == healthy_run_id
    finally:
        conn.close()


def test_long_or_blocked_runs_do_not_quarantine_known_good_assignee(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "good"
    )
    conn = kb.connect()
    try:
        history = _create_task(conn, title="history", assignee="good", status="done")
        _seed_closed_run(conn, history, "good", "protocol_violation", duration=120)
        _seed_closed_run(conn, history, "good", "blocked")
        ready = _create_task(conn, title="ready", assignee="good")
        spawned = []

        result = kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 990
        )

        assert ready in spawned
        assert result.skipped_assignee_quarantined == []
        assert _all_events(conn, "assignee_quarantined") == []
    finally:
        conn.close()


def test_kanban_event_disqualifies_protocol_violation_as_boot_failure(
    kanban_home, monkeypatch
):
    monkeypatch.setattr(
        "hermes_cli.profiles.profile_exists", lambda name: name == "good"
    )
    conn = kb.connect()
    try:
        first = _create_task(conn, title="first", assignee="good")
        second = _create_task(conn, title="second", assignee="good")
        waiting = _create_task(conn, title="waiting", assignee="good")
        _drive_clean_boot_exit(conn, first, 983001, monkeypatch)
        _drive_clean_boot_exit(conn, second, 983002, monkeypatch)
        _drive_clean_exit_after_heartbeat(conn, first, 983003, monkeypatch)
        spawned = []

        kb.dispatch_once(
            conn, spawn_fn=lambda task, workspace: spawned.append(task.id) or 991
        )

        assert waiting in spawned
        assert _all_events(conn, "assignee_quarantined") == []
    finally:
        conn.close()


def test_unknown_assignee_skip_emits_loud_audit_event_once(
    kanban_home, monkeypatch, caplog
):
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    conn = kb.connect()
    try:
        task_id = _create_task(
            conn, title="stranded", assignee="does-not-exist"
        )

        result = kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 123)
        assert task_id in result.skipped_unknown_assignee
        events = _all_events(conn, "unknown_assignee_skipped")
        assert len(events) == 1
        assert json.loads(events[0]["payload"]) == {
            "assignee": "does-not-exist"
        }
        assert "unknown assignee" in caplog.text.lower()

        kb.dispatch_once(conn, spawn_fn=lambda task, workspace: 124)
        assert len(_all_events(conn, "unknown_assignee_skipped")) == 1
        assert kb.get_task(conn, task_id).status == "ready"
    finally:
        conn.close()
