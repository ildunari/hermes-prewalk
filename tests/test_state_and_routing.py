from __future__ import annotations

import copy

from prewalk import core


def test_failed_executor_switch_does_not_advance_phase(fake_agent, monkeypatch):
    core.command_handler("test")
    state = core.get_state(fake_agent.session_id)
    core.on_post_tool_call(tool_name="todo", status="ok", session_id=fake_agent.session_id, result="{}")
    monkeypatch.setattr(core, "_switch_live_agent", lambda agent, slot: (False, "executor unavailable"))

    core.on_post_tool_call(tool_name="patch", status="ok", session_id=fake_agent.session_id, result="{}")

    assert state.phase == "planning"
    assert state.transitioning is False
    assert "executor unavailable" in state.last_error


def test_post_llm_restores_full_posture_and_removes_state(fake_agent):
    original_reasoning = copy.deepcopy(fake_agent.reasoning_config)
    core.command_handler("test")
    core.on_post_llm_call(session_id=fake_agent.session_id, assistant_response="done")

    assert core.get_state(fake_agent.session_id) is None
    assert fake_agent.model == "original-model"
    assert fake_agent.provider == "openai-codex"
    assert fake_agent.reasoning_config == original_reasoning


def test_gateway_or_headless_command_cannot_arm(monkeypatch, presets):
    monkeypatch.setattr(core, "load_presets", lambda: presets)
    monkeypatch.setattr(core, "_get_agent", lambda: None)
    message = core.command_handler("test")
    assert "cli-only" in message.lower()
    assert core.state_count() == 0
