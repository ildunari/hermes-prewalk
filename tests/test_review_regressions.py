from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

from conftest import FakeAgent
from prewalk import core


VALID_TODO = {
    "todos": [{
        "id": "edit",
        "content": "Edit the target; verify with the focused test",
        "status": "pending",
    }]
}


def _request():
    return {"model": "original", "messages": [{"role": "user", "content": "Fix it"}]}


def test_budget_exhaustion_restores_and_disarms_before_an_unbounded_call(fake_agent, monkeypatch, presets):
    limited = copy.deepcopy(presets)
    limited["settings"]["max_planner_api_calls"] = 1
    monkeypatch.setattr(core, "load_presets", lambda: copy.deepcopy(limited))
    core.command_handler("test")

    first = core.on_llm_request(
        request=_request(), session_id=fake_agent.session_id, turn_id="t", api_call_count=1
    )["request"]
    assert first["model"] == "planner-model"

    second = core.on_llm_request(
        request=_request(), session_id=fake_agent.session_id, turn_id="t", api_call_count=2
    )["request"]
    assert second["model"] == "original-model"
    assert "budget" in str(second).lower()
    assert core.get_state(fake_agent.session_id) is None
    assert fake_agent.model == "original-model"


def test_resolved_fallback_routes_are_used_by_middleware(monkeypatch, presets):
    agent = FakeAgent()
    monkeypatch.setattr(core, "load_presets", lambda: copy.deepcopy(presets))
    monkeypatch.setattr(core, "_get_agent", lambda: agent)
    monkeypatch.setattr(core, "_get_cli", lambda: None)
    switches = iter([
        ("planner-fallback", "fallback-planner-provider"),
        ("executor-fallback", "fallback-executor-provider"),
    ])

    def fallback_switch(live_agent, slot):
        model, provider = next(switches)
        live_agent.model = model
        live_agent.provider = provider
        return True, f"{model} switched"

    monkeypatch.setattr(core, "_switch_live_agent", fallback_switch)
    core.command_handler("test")
    planner = core.on_llm_request(
        request=_request(), session_id=agent.session_id, turn_id="t", api_call_count=1
    )["request"]
    assert planner["model"] == "planner-fallback"

    core.on_post_tool_call(
        tool_name="todo", tool_args=VALID_TODO, status="ok", session_id=agent.session_id
    )
    core.on_post_tool_call(tool_name="patch", status="ok", session_id=agent.session_id)
    executor = core.on_llm_request(
        request=_request(), session_id=agent.session_id, turn_id="t", api_call_count=2
    )["request"]
    assert executor["model"] == "executor-fallback"


def test_two_interleaved_sessions_keep_routes_counters_and_cleanup_independent(monkeypatch, presets):
    data = copy.deepcopy(presets)
    data["presets"]["other"] = copy.deepcopy(data["presets"]["test"])
    data["presets"]["other"]["planner"]["model"] = "planner-other"
    data["presets"]["other"]["executor"]["model"] = "executor-other"
    monkeypatch.setattr(core, "load_presets", lambda: copy.deepcopy(data))

    agent_a = FakeAgent("session-a")
    agent_b = FakeAgent("session-b")
    active = {"agent": agent_a}
    monkeypatch.setattr(core, "_get_agent", lambda: active["agent"])
    monkeypatch.setattr(core, "_get_cli", lambda: None)

    def switch(live_agent, slot):
        live_agent.model = slot["model"]
        live_agent.provider = slot["provider"]
        return True, "switched"

    monkeypatch.setattr(core, "_switch_live_agent", switch)
    core.command_handler("test")
    active["agent"] = agent_b
    core.command_handler("other")

    with ThreadPoolExecutor(max_workers=2) as pool:
        future_a = pool.submit(
            core.on_llm_request, request=_request(), session_id="session-a",
            turn_id="turn-a", api_call_count=1
        )
        future_b = pool.submit(
            core.on_llm_request, request=_request(), session_id="session-b",
            turn_id="turn-b", api_call_count=1
        )
        request_a = future_a.result()["request"]
        request_b = future_b.result()["request"]

    assert request_a["model"] == "planner-model"
    assert request_b["model"] == "planner-other"
    assert core.get_state("session-a").planner_api_calls == 1
    assert core.get_state("session-b").planner_api_calls == 1

    core.on_session_reset(session_id="session-a")
    assert core.get_state("session-a") is None
    assert core.get_state("session-b") is not None
    assert agent_a.model == "original-model"
    assert agent_b.model == "planner-other"


def test_pre_verify_uses_host_directive_contract_once(fake_agent, monkeypatch, presets):
    core.command_handler("test")
    core.on_post_tool_call(
        tool_name="todo", tool_args=VALID_TODO, status="ok", session_id=fake_agent.session_id
    )
    core.on_post_tool_call(tool_name="patch", status="ok", session_id=fake_agent.session_id)
    core.on_llm_request(
        request=_request(), session_id=fake_agent.session_id,
        turn_id="turn-verify", api_call_count=4,
    )
    enabled = copy.deepcopy(presets)
    enabled["verify"] = {"enabled": True}
    monkeypatch.setattr(core, "load_presets", lambda: enabled)

    directive = core.on_pre_verify(session_id=fake_agent.session_id, attempt=0)
    assert directive is not None
    assert directive["action"] == "continue"
    assert "Prewalk verification" in directive["message"]
    assert core.on_pre_verify(session_id=fake_agent.session_id, attempt=0) is None
