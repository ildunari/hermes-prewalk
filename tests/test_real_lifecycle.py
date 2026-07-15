"""Trajectory-level regression: inspect provider-bound requests in one Hermes turn."""

from __future__ import annotations

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from prewalk import core


class _Provider(BaseHTTPRequestHandler):
    captured: list[dict] = []
    response_queue: list[dict] = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        # Fresh Hermes homes issue model-context probes through the same
        # chat-completions endpoint. Those probes do not carry the agent's
        # tool schema and are not part of the task trajectory under test.
        if not payload.get("tools"):
            response = _text_response("context probe")
        else:
            type(self).captured.append(payload)
            response = type(self).response_queue.pop(0)
        message = response["choices"][0]["message"]
        if payload.get("stream") is True:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"id": "r", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            ]
            if message.get("content"):
                chunks.append({"id": "r", "choices": [{"index": 0, "delta": {"content": message["content"]}, "finish_reason": None}]})
            for index, tool_call in enumerate(message.get("tool_calls") or []):
                chunks.append({
                    "id": "r",
                    "choices": [{
                        "index": 0,
                        "delta": {"tool_calls": [{
                            "index": index,
                            "id": tool_call["id"],
                            "type": "function",
                            "function": tool_call["function"],
                        }]},
                        "finish_reason": None,
                    }],
                })
            finish = "tool_calls" if message.get("tool_calls") else "stop"
            chunks.append({"id": "r", "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def _tool_response(name: str, arguments: dict, call_id: str) -> dict:
    return {
        "id": f"resp-{call_id}",
        "model": "fake",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    }


def _text_response(text: str) -> dict:
    return {
        "id": "resp-final",
        "model": "fake",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


def _tool_defs(*names: str) -> list[dict]:
    return [
        {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": {}}}}
        for name in names
    ]


@pytest.mark.integration
def test_real_one_turn_request_sequence(monkeypatch, presets):
    from unittest.mock import patch

    from hermes_cli import plugins as plugins_mod
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    import prewalk
    from run_agent import AIAgent

    manager = PluginManager()
    monkeypatch.setattr(plugins_mod, "_plugin_manager", manager)
    prewalk.register(PluginContext(PluginManifest(name="prewalk", version="1.2.1"), manager))

    _Provider.captured = []
    workdir = Path(__file__).resolve().parents[1] / ".pytest-work"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir()
    widget = workdir / "widget.py"
    widget.write_text("OLD\n", encoding="utf-8")
    monkeypatch.chdir(workdir)

    _Provider.response_queue = [
        _tool_response("read_file", {"path": str(widget)}, "call-read"),
        _tool_response(
            "todo",
            {"todos": [{"id": "edit", "content": "Edit widget and run focused test", "status": "pending"}]},
            "call-todo",
        ),
        _tool_response(
            "patch",
            {"mode": "replace", "path": str(widget), "old_string": "OLD", "new_string": "NEW"},
            "call-patch",
        ),
        _text_response("Implemented and validated."),
    ]
    server = HTTPServer(("127.0.0.1", 0), _Provider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    definitions = _tool_defs("read_file", "todo", "patch")
    try:
        with (
            patch("run_agent.get_tool_definitions", return_value=definitions),
            patch("run_agent.check_toolset_requirements", return_value={}),
        ):
            agent = AIAgent(
                api_key="test-key",
                base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                provider="openai-compat",
                model="original-model",
                max_iterations=10,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                save_trajectories=False,
                platform="cli",
            )
        agent.session_id = "session-real-lifecycle"
        agent.tools = definitions
        agent.valid_tool_names = {"read_file", "todo", "patch"}
        agent._cached_system_prompt = "SYSTEM"
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False

        monkeypatch.setattr(core, "load_presets", lambda: presets)
        monkeypatch.setattr(core, "_get_agent", lambda: agent)
        monkeypatch.setattr(core, "_get_cli", lambda: None)
        monkeypatch.setattr(
            "agent.verification_stop.verify_on_stop_enabled", lambda: False
        )

        def switch_without_replacing_client(live_agent, slot):
            live_agent.model = slot["model"]
            live_agent.provider = slot["provider"]
            return True, f"{slot['model']} switched"

        monkeypatch.setattr(core, "_switch_live_agent", switch_without_replacing_client)

        assert "prewalk armed" in core.command_handler("test")

        result = agent.run_conversation(
            "Update the widget.", conversation_history=[], task_id="task-prewalk"
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result["final_response"] == "Implemented and validated."
    assert widget.read_text(encoding="utf-8") == "NEW\n"
    assert len(_Provider.captured) == 4
    assert [request["model"] for request in _Provider.captured] == [
        "planner-model", "planner-model", "planner-model", "executor-model"
    ]

    planner_wires = [json.dumps(request["messages"]) for request in _Provider.captured[:3]]
    assert all(wire.count(core.PLANNING_OPEN) == 1 for wire in planner_wires)
    assert all(core.HANDOFF_OPEN not in wire for wire in planner_wires)

    executor_messages = _Provider.captured[3]["messages"]
    executor_wire = json.dumps(executor_messages)
    assert core.PLANNING_OPEN not in executor_wire
    assert executor_wire.count(core.HANDOFF_OPEN) == 1
    assert any(message.get("role") == "assistant" and message.get("tool_calls") for message in executor_messages)
    assert any(message.get("role") == "tool" and message.get("tool_call_id") == "call-patch" for message in executor_messages)
    assert core.get_state(agent.session_id) is None
    assert agent.model == "original-model"
    shutil.rmtree(workdir, ignore_errors=True)
