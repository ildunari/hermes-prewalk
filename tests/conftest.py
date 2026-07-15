from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(os.environ.get("HERMES_AGENT_ROOT", Path.home() / ".hermes" / "hermes-agent"))
for path in (ROOT, HERMES_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prewalk import core  # noqa: E402


class FakeAgent:
    def __init__(self, session_id: str = "session-a", *, with_todo: bool = True):
        self.session_id = session_id
        self.platform = "cli"
        self.model = "original-model"
        self.provider = "openai-codex"
        self.base_url = "https://original.invalid/v1"
        self.api_mode = "responses"
        self.api_key = "original-secret"
        self.reasoning_config = {"effort": "medium", "summary": "auto"}
        names = ["read_file", "patch", "write_file"]
        if with_todo:
            names.append("todo")
        self.tools = [
            {"type": "function", "function": {"name": name, "parameters": {}}}
            for name in names
        ]


@pytest.fixture(autouse=True)
def clean_registry(monkeypatch):
    core.reset_all_state_for_tests()
    monkeypatch.setattr(core, "CTX", SimpleNamespace(profile_name="gpt"), raising=False)
    yield
    core.reset_all_state_for_tests()


@pytest.fixture
def presets():
    return {
        "default_preset": "test",
        "settings": {
            "edit_tools": ["write_file", "patch"],
            "todo_cap": 4,
            "max_planner_api_calls": 8,
            "max_planning_seconds": 300,
        },
        "verify": {"enabled": False},
        "presets": {
            "test": {
                "description": "test preset",
                "planner": {"provider": "openai-codex", "model": "planner-model", "effort": "high"},
                "executor": {"provider": "openai-codex", "model": "executor-model", "effort": "low"},
            }
        },
    }


@pytest.fixture
def fake_agent(monkeypatch, presets):
    agent = FakeAgent()
    monkeypatch.setattr(core, "load_presets", lambda: copy.deepcopy(presets))
    monkeypatch.setattr(core, "_get_agent", lambda: agent)
    monkeypatch.setattr(core, "_get_cli", lambda: None)
    def fake_switch(live_agent, slot):
        live_agent.model = slot["model"]
        live_agent.provider = slot["provider"]
        live_agent.base_url = "https://planner.invalid/v1"
        live_agent.api_mode = "chat_completions"
        live_agent.api_key = "planner-secret"
        live_agent.reasoning_config = {"effort": slot.get("effort")}
        return True, f"{slot['model']} switched"

    monkeypatch.setattr(core, "_switch_live_agent", fake_switch)
    return agent
