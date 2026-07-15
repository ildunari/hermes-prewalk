from __future__ import annotations

from prewalk import core
from conftest import FakeAgent


VALID_TODO = {
    "todos": [{
        "id": "widget",
        "content": "Change the widget; verify with the focused widget test",
        "status": "pending",
    }]
}


def test_edit_is_blocked_until_successful_todo(fake_agent):
    core.command_handler("test")
    blocked = core.on_pre_tool_call(
        tool_name="patch", session_id=fake_agent.session_id, args={"path": "x.py"}
    )
    assert blocked["action"] == "block"
    assert "todo" in blocked["message"].lower()

    core.on_post_tool_call(
        tool_name="todo", tool_args=VALID_TODO, status="error",
        session_id=fake_agent.session_id, result='{"error":"bad"}'
    )
    assert core.on_pre_tool_call(
        tool_name="patch", session_id=fake_agent.session_id, args={}
    )["action"] == "block"

    core.on_post_tool_call(
        tool_name="todo", tool_args=VALID_TODO, status="ok",
        session_id=fake_agent.session_id, result='{"ok":true}'
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


def test_empty_or_non_verifiable_todo_never_unlocks_edits(fake_agent):
    core.command_handler("test")
    for payload in (
        {"todos": []},
        {},
        {"todos": [{"id": "x", "content": "Change it", "status": "pending"}]},
    ):
        blocked = core.on_pre_tool_call(
            tool_name="todo", args=payload, session_id=fake_agent.session_id
        )
        assert blocked["action"] == "block"
        core.on_post_tool_call(
            tool_name="todo", tool_args=payload, status="ok",
            session_id=fake_agent.session_id, result="{}"
        )
        assert core.get_state(fake_agent.session_id).todo_ready is False


def test_mutating_shell_and_execute_code_are_gated_but_read_only_shell_is_allowed(fake_agent):
    core.command_handler("test")
    assert core.on_pre_tool_call(
        tool_name="terminal", args={"command": "git status --short"},
        session_id=fake_agent.session_id,
    ) is None
    assert core.on_pre_tool_call(
        tool_name="search_files", args={"pattern": "widget"},
        session_id=fake_agent.session_id,
    ) is None
    for tool_name, args in (
        ("terminal", {"command": "python3 -c 'open(\"x.py\", \"w\").write(\"x\")'"}),
        ("terminal", {"command": "sed -i '' 's/a/b/' x.py"}),
        ("terminal", {"command": "cat a > b"}),
        ("terminal", {"command": "cat a | python3 -c 'open(\"x\", \"w\").write(\"x\")'"}),
        ("terminal", {"command": "git branch -D old-work"}),
        ("terminal", {"command": "git show HEAD --output=/tmp/leak.patch"}),
        ("execute_code", {"code": "from pathlib import Path; Path('x').write_text('x')"}),
        ("mcp__open_design__write_file", {"projectId": "p", "path": "x.ts"}),
        ("tool_call", {"name": "mcp__open_design__delete_file", "arguments": {}}),
        ("mcp__open_design__start_run", {"projectId": "p"}),
        ("tool_call", {"name": "mcp__open_design__cancel_run", "arguments": {}}),
    ):
        core.command_handler("test")
        blocked = core.on_pre_tool_call(
            tool_name=tool_name, args=args, session_id=fake_agent.session_id
        )
        assert blocked["action"] == "block", (tool_name, args)
