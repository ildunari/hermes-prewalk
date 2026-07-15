from __future__ import annotations

from prewalk import core
from conftest import FakeAgent


def test_edit_is_blocked_until_successful_todo(fake_agent):
    core.command_handler("test")
    blocked = core.on_pre_tool_call(
        tool_name="patch", session_id=fake_agent.session_id, args={"path": "x.py"}
    )
    assert blocked["action"] == "block"
    assert "todo" in blocked["message"].lower()

    core.on_post_tool_call(
        tool_name="todo", status="error", session_id=fake_agent.session_id, result='{"error":"bad"}'
    )
    assert core.on_pre_tool_call(
        tool_name="patch", session_id=fake_agent.session_id, args={}
    )["action"] == "block"

    core.on_post_tool_call(
        tool_name="todo", status="ok", session_id=fake_agent.session_id, result='{"ok":true}'
    )
    assert core.on_pre_tool_call(
        tool_name="patch", session_id=fake_agent.session_id, args={}
    ) is None


def test_arm_fails_cleanly_when_todo_tool_is_unavailable(monkeypatch, presets):
    agent = FakeAgent(session_id="session-no-todo", with_todo=False)
    monkeypatch.setattr(core, "load_presets", lambda: presets)
    monkeypatch.setattr(core, "_get_agent", lambda: agent)
    monkeypatch.setattr(core, "_get_cli", lambda: None)

    message = core.command_handler("test")
    assert "todo tool is unavailable" in message.lower()
    assert core.get_state("session-no-todo") is None


def test_repeated_premature_edits_disarm_instead_of_looping(fake_agent):
    core.command_handler("test")
    first = core.on_pre_tool_call(tool_name="patch", session_id=fake_agent.session_id, args={})
    second = core.on_pre_tool_call(tool_name="patch", session_id=fake_agent.session_id, args={})
    assert first["action"] == "block"
    assert second["action"] == "block"
    assert "disarmed" in second["message"].lower()
    assert core.get_state(fake_agent.session_id) is None
