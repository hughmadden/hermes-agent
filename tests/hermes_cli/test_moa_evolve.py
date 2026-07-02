"""Tests for the MoA skills/evolution loop.

Covers the aggregation-skill loader + guidance injection (agent/moa_loop.py)
and the `hermes moa evolve` trace distiller (hermes_cli/moa_evolve.py). All
LLM calls are faked; the live loop is exercised in
tests/integration/test_moa_proxy_live.py.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

import agent.moa_loop as moa_loop
from hermes_cli.moa_evolve import _load_recent_turns, cmd_moa_evolve


def _write_skill(home, body: str, with_frontmatter: bool = True) -> None:
    path = home / "skills" / "moa-aggregation" / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: moa-aggregation\n---\n\n{body}\n" if with_frontmatter else body
    path.write_text(content, encoding="utf-8")


@pytest.fixture()
def moa_home(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: ref-model-a
      aggregator:
        provider: openrouter
        model: agg-model
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    moa_loop._skill_cache.clear()
    return home


# ---------------------------------------------------------------------------
# Skill loading + injection
# ---------------------------------------------------------------------------


def test_load_aggregation_skill_absent_is_empty(moa_home):
    assert moa_loop.load_aggregation_skill() == ""
    assert moa_loop.aggregation_skill_block() == ""


def test_load_aggregation_skill_strips_frontmatter(moa_home):
    _write_skill(moa_home, "- Trust reference A on SQL.")
    body = moa_loop.load_aggregation_skill()
    assert body == "- Trust reference A on SQL."
    assert "name: moa-aggregation" not in body
    block = moa_loop.aggregation_skill_block()
    assert "[Aggregation heuristics" in block
    assert "- Trust reference A on SQL." in block


def test_load_aggregation_skill_mtime_cache_refreshes(moa_home):
    _write_skill(moa_home, "- old rule")
    assert "old rule" in moa_loop.load_aggregation_skill()
    path = moa_home / "skills" / "moa-aggregation" / "SKILL.md"
    _write_skill(moa_home, "- new rule")
    # Ensure the mtime actually changes even on coarse filesystem clocks.
    os.utime(path, (time.time() + 2, time.time() + 2))
    assert "new rule" in moa_loop.load_aggregation_skill()


def test_skill_injected_into_facade_guidance(moa_home, monkeypatch):
    _write_skill(moa_home, "- Weigh minority dissent seriously.")
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr(moa_loop, "call_llm", fake_call_llm)
    facade = moa_loop.MoAChatCompletions("review")
    facade.create(messages=[{"role": "user", "content": "q"}])
    agg_call = next(c for c in calls if c["task"] == "moa_aggregator")
    guidance = agg_call["messages"][-1]["content"]
    assert "[Aggregation heuristics" in guidance
    assert "Weigh minority dissent seriously." in guidance


@pytest.mark.asyncio
async def test_skill_injected_into_proxy_guidance(moa_home, monkeypatch):
    from aiohttp.test_utils import TestClient, TestServer

    import hermes_cli.proxy.moa_server as moa_server

    _write_skill(moa_home, "- Prefer concrete file paths over vague plans.")
    moa_server._ref_cache.clear()
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(content="ok", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr(moa_server, "call_llm", fake_call_llm)
    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)

    client = TestClient(TestServer(moa_server.create_moa_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/v1/chat/completions",
            json={"model": "moa:review", "messages": [{"role": "user", "content": "q"}]},
        )
        assert resp.status == 200
        agg_call = next(c for c in calls if c["task"] == "moa_aggregator")
        guidance = agg_call["messages"][-1]["content"]
        assert "[Aggregation heuristics" in guidance
        assert "Prefer concrete file paths over vague plans." in guidance
    finally:
        await client.close()


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def _trace_record(ts: float, preset: str = "review", agg_output: str = "did it") -> dict:
    return {
        "ts": ts,
        "preset": preset,
        "references": [
            {"label": "openrouter:ref-model-a", "output": "advice text"}
        ],
        "aggregator": {
            "label": "openrouter:agg-model",
            "input_messages": [{"role": "user", "content": "the task"}],
            "output": agg_output,
        },
    }


def _write_traces(home, records, name="session1"):
    trace_dir = home / "moa-traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    with (trace_dir / f"{name}.jsonl").open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return trace_dir


def test_load_recent_turns_orders_and_limits(moa_home):
    trace_dir = _write_traces(
        moa_home, [_trace_record(3), _trace_record(1)], name="a"
    )
    _write_traces(moa_home, [_trace_record(2)], name="b")
    turns = _load_recent_turns(trace_dir, max_turns=2)
    assert [t["ts"] for t in turns] == [2, 3]
    turns = _load_recent_turns(trace_dir, max_turns=10)
    assert [t["ts"] for t in turns] == [1, 2, 3]


# ---------------------------------------------------------------------------
# cmd_moa_evolve
# ---------------------------------------------------------------------------


def _args(**kw):
    defaults = {"max_turns": 30, "model": None, "trace_dir": None, "dry_run": False}
    defaults.update(kw)
    return SimpleNamespace(**defaults)


def test_evolve_no_traces_returns_1(moa_home, capsys):
    assert cmd_moa_evolve(_args()) == 1
    assert "No MoA traces" in capsys.readouterr().out


def test_evolve_writes_skill_file(moa_home, monkeypatch, capsys):
    _write_traces(moa_home, [_trace_record(1), _trace_record(2)])
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(
            content="## Heuristics\n- Distrust vague advice.", tool_calls=[]
        )
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    assert cmd_moa_evolve(_args()) == 0

    path = moa_home / "skills" / "moa-aggregation" / "SKILL.md"
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\nname: moa-aggregation")
    assert "auto_generated: moa-evolve" in content
    assert "turns_analyzed: 2" in content
    assert "- Distrust vague advice." in content

    # The grading prompt carried the trace evidence and used the default
    # preset's aggregator as the distiller.
    prompt = seen["messages"][-1]["content"]
    assert "advice text" in prompt
    assert "the task" in prompt
    assert seen["model"] == "agg-model"

    # And the freshly written skill is immediately live for injection.
    moa_loop._skill_cache.clear()
    assert "Distrust vague advice." in moa_loop.load_aggregation_skill()


def test_evolve_rewrite_includes_existing_body(moa_home, monkeypatch):
    _write_skill(moa_home, "- Existing rule to keep or merge.")
    _write_traces(moa_home, [_trace_record(1)])
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content="- merged rules", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    assert cmd_moa_evolve(_args()) == 0
    assert "Existing rule to keep or merge." in seen["messages"][-1]["content"]


def test_evolve_dry_run_does_not_write(moa_home, monkeypatch, capsys):
    _write_traces(moa_home, [_trace_record(1)])

    def fake_call_llm(**kwargs):
        message = SimpleNamespace(content="- would-be rule", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    assert cmd_moa_evolve(_args(dry_run=True)) == 0
    assert not (moa_home / "skills" / "moa-aggregation" / "SKILL.md").exists()
    assert "would-be rule" in capsys.readouterr().out


def test_evolve_explicit_model_arg(moa_home, monkeypatch):
    _write_traces(moa_home, [_trace_record(1)])
    seen = {}

    def fake_call_llm(**kwargs):
        seen.update(kwargs)
        message = SimpleNamespace(content="- rule", tool_calls=[])
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=None, model="fake")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    assert cmd_moa_evolve(_args(model="openrouter:some/cheap-model")) == 0
    assert seen["model"] == "some/cheap-model"
    assert seen["provider"] == "openrouter"


def test_evolve_bad_model_arg_exits(moa_home):
    _write_traces(moa_home, [_trace_record(1)])
    with pytest.raises(SystemExit):
        cmd_moa_evolve(_args(model="not-a-slot"))
