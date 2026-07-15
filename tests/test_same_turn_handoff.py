from __future__ import annotations

import json

from prewalk import core


def _wire(request: dict) -> str:
    return json.dumps(request, sort_keys=True)


def _request(model: str = "original-model") -> dict:
    return {
        "model": model,
        "messages": [{"role": "user", "content": "Fix the widget."}],
        "reasoning_effort": "medium",
    }


def test_middleware_owns_planning_injection_and_same_turn_handoff(fake_agent):
    armed = core.command_handler("test")
    assert "prewalk armed" in armed

    planning_1 = core.on_llm_request(
        request=_request(), session_id=fake_agent.session_id, turn_id="turn-1", api_call_count=1
    )["request"]
    planning_2 = core.on_llm_request(
        request=_request(), session_id=fake_agent.session_id, turn_id="turn-1", api_call_count=2
    )["request"]

    assert planning_1["model"] == "planner-model"
    assert core.PLANNING_OPEN in _wire(planning_1)
    assert core.HANDOFF_OPEN not in _wire(planning_1)
    assert _wire(planning_1).count(core.PLANNING_OPEN) == 1
    assert _wire(planning_2).count(core.PLANNING_OPEN) == 1

    core.on_post_tool_call(
        tool_name="todo", status="ok", session_id=fake_agent.session_id, result='{"ok": true}'
    )
    core.on_post_tool_call(
        tool_name="patch", status="ok", session_id=fake_agent.session_id, result='{"ok": true}'
    )

    executor_1 = core.on_llm_request(
        request=_request("planner-model"), session_id=fake_agent.session_id,
        turn_id="turn-1", api_call_count=3
    )["request"]
    executor_2 = core.on_llm_request(
        request=_request("executor-model"), session_id=fake_agent.session_id,
        turn_id="turn-1", api_call_count=4
    )["request"]

    assert executor_1["model"] == "executor-model"
    assert core.PLANNING_OPEN not in _wire(executor_1)
    assert _wire(executor_1).count(core.HANDOFF_OPEN) == 1
    assert core.PLANNING_OPEN not in _wire(executor_2)
    assert core.HANDOFF_OPEN not in _wire(executor_2)


def test_responses_input_shape_is_rewritten(fake_agent):
    core.command_handler("test")
    request = {
        "model": "original-model",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Fix it"}]}],
    }
    rewritten = core.on_llm_request(
        request=request, session_id=fake_agent.session_id, turn_id="turn-r", api_call_count=1
    )["request"]
    assert rewritten["model"] == "planner-model"
    assert core.PLANNING_OPEN in _wire(rewritten)


def test_middleware_is_session_isolated(fake_agent):
    core.command_handler("test")
    untouched = core.on_llm_request(
        request=_request(), session_id="session-b", turn_id="turn-b", api_call_count=1
    )
    assert untouched is None
