"""Prewalk v1.2 core: correct same-turn planner-to-executor handoff.

The request middleware owns planning instruction injection, pruning, and the
one-time handoff because Hermes invokes it before every provider request. Slash
command arming is intentionally CLI-only in v1.2: gateway plugin command
handlers receive only raw argument text and no session identity.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import shlex
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).parent
_PRESETS_FILE = _PLUGIN_DIR / "presets.yaml"
_VALID_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}

PLANNING_OPEN = "<prewalk-planning-v1>"
PLANNING_CLOSE = "</prewalk-planning-v1>"
HANDOFF_OPEN = "<prewalk-handoff-v1>"
HANDOFF_CLOSE = "</prewalk-handoff-v1>"
_MARKED_BLOCK_RE = re.compile(
    rf"(?:{re.escape(PLANNING_OPEN)}.*?{re.escape(PLANNING_CLOSE)}|"
    rf"{re.escape(HANDOFF_OPEN)}.*?{re.escape(HANDOFF_CLOSE)})",
    re.DOTALL,
)

CTX = None


@dataclass
class RuntimePosture:
    model: str
    provider: str
    base_url: str
    api_mode: str
    api_key: Any
    reasoning_config: Any
    cli_fields: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrewalkState:
    session_id: str
    preset_name: str
    preset: dict[str, Any]
    original: RuntimePosture
    agent: Any
    active_planner: dict[str, Any]
    active_executor: dict[str, Any]
    phase: str = "planning"
    planner_api_calls: int = 0
    seen_api_calls: set[tuple[str, int]] = field(default_factory=set)
    planning_started_at: float = field(default_factory=time.monotonic)
    todo_ready: bool = False
    first_edit_landed: bool = False
    handoff_injected: bool = False
    transitioning: bool = False
    blocked_edits: int = 0
    verify_nudges: int = 0
    budget_exhausted: bool = False
    last_error: str = ""
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


_STATES: dict[str, PrewalkState] = {}
_STATES_LOCK = threading.RLock()


def get_state(session_id: str) -> PrewalkState | None:
    if not session_id:
        return None
    with _STATES_LOCK:
        return _STATES.get(session_id)


def state_count() -> int:
    with _STATES_LOCK:
        return len(_STATES)


def reset_all_state_for_tests() -> None:
    with _STATES_LOCK:
        _STATES.clear()


def load_presets() -> dict:
    import yaml

    with open(_PRESETS_FILE, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.setdefault("presets", {})
    data.setdefault("settings", {})
    data.setdefault("verify", {})
    return data


def _preset_setting(preset: dict | None, data: dict, key: str, default: Any) -> Any:
    if preset and key in preset:
        return preset[key]
    return data.get("settings", {}).get(key, default)


def _slot(preset: dict | None, name: str) -> dict | None:
    value = (preset or {}).get(name)
    if not isinstance(value, dict) or not value.get("model"):
        return None
    return value


def _get_cli():
    try:
        from hermes_cli.plugins import get_plugin_manager

        return getattr(get_plugin_manager(), "_cli_ref", None)
    except Exception:
        return None


def _get_agent():
    cli = _get_cli()
    return getattr(cli, "agent", None) if cli is not None else None


def _tool_names(agent: Any) -> set[str]:
    names: set[str] = set()
    for item in getattr(agent, "tools", None) or []:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.add(str(function["name"]))
        elif item.get("name"):
            names.add(str(item["name"]))
    return names


def _snapshot_posture(agent: Any) -> RuntimePosture:
    cli = _get_cli()
    cli_fields: dict[str, Any] = {}
    if cli is not None:
        for name in (
            "model",
            "provider",
            "requested_provider",
            "api_key",
            "base_url",
            "api_mode",
            "_explicit_api_key",
            "_explicit_base_url",
        ):
            if hasattr(cli, name):
                try:
                    cli_fields[name] = copy.deepcopy(getattr(cli, name))
                except Exception:
                    cli_fields[name] = getattr(cli, name)
    try:
        reasoning = copy.deepcopy(getattr(agent, "reasoning_config", None))
    except Exception:
        reasoning = getattr(agent, "reasoning_config", None)
    return RuntimePosture(
        model=str(getattr(agent, "model", "") or ""),
        provider=str(getattr(agent, "provider", "") or ""),
        base_url=str(getattr(agent, "base_url", "") or ""),
        api_mode=str(getattr(agent, "api_mode", "") or ""),
        api_key=getattr(agent, "api_key", ""),
        reasoning_config=reasoning,
        cli_fields=cli_fields,
    )


def _apply_effort(agent: Any, effort: str) -> None:
    normalized = str(effort or "").lower()
    if normalized not in _VALID_EFFORTS:
        return
    try:
        from hermes_constants import parse_reasoning_effort

        config = parse_reasoning_effort(normalized)
        if config is not None:
            agent.reasoning_config = config
    except Exception as exc:
        logger.debug("prewalk: reasoning effort update failed: %s", exc)


def _switch_live_agent(agent: Any, slot: dict) -> tuple[bool, str]:
    """Resolve and atomically switch a live CLI agent to a preset slot."""
    try:
        from hermes_cli.model_switch import switch_model as resolve_switch
    except Exception as exc:
        return False, f"Hermes model switch unavailable: {exc}"

    candidates = [(str(slot.get("provider", "") or ""), str(slot["model"]))]
    for fallback in slot.get("fallbacks", []) or []:
        if isinstance(fallback, (list, tuple)) and len(fallback) >= 2:
            candidates.append((str(fallback[0]), str(fallback[1])))

    errors: list[str] = []
    cli = _get_cli()
    for provider, model in candidates:
        try:
            result = resolve_switch(
                raw_input=model,
                current_provider=getattr(agent, "provider", "") or "",
                current_model=getattr(agent, "model", "") or "",
                current_base_url=getattr(agent, "base_url", "") or "",
                current_api_key=getattr(agent, "api_key", "") or "",
                is_global=False,
                explicit_provider=provider,
            )
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            continue
        if not getattr(result, "success", False):
            errors.append(f"{model}: {getattr(result, 'error_message', 'resolution failed')}")
            continue
        try:
            agent.switch_model(
                new_model=result.new_model,
                new_provider=result.target_provider,
                api_key=result.api_key,
                base_url=result.base_url,
                api_mode=result.api_mode,
            )
        except Exception as exc:
            errors.append(f"{model}: live switch failed ({exc})")
            continue

        if cli is not None:
            try:
                cli.model = result.new_model
                cli.provider = result.target_provider
                cli.requested_provider = result.target_provider
                cli._explicit_api_key = result.api_key
                cli._explicit_base_url = result.base_url
                if result.api_key:
                    cli.api_key = result.api_key
                if result.base_url:
                    cli.base_url = result.base_url
                if result.api_mode:
                    cli.api_mode = result.api_mode
            except Exception as exc:
                logger.debug("prewalk: CLI field mirror failed: %s", exc)
        _apply_effort(agent, str(slot.get("effort", "") or ""))
        label = getattr(result, "provider_label", "") or result.target_provider
        return True, f"{result.new_model} via {label}"
    return False, "; ".join(errors) or "no model candidates resolved"


def _restore_runtime(state: PrewalkState) -> tuple[bool, str]:
    posture = state.original
    agent = state.agent
    try:
        if hasattr(agent, "switch_model"):
            agent.switch_model(
                new_model=posture.model,
                new_provider=posture.provider,
                api_key=posture.api_key,
                base_url=posture.base_url,
                api_mode=posture.api_mode,
            )
        else:
            agent.model = posture.model
            agent.provider = posture.provider
            agent.base_url = posture.base_url
            agent.api_mode = posture.api_mode
            agent.api_key = posture.api_key
        try:
            agent.reasoning_config = copy.deepcopy(posture.reasoning_config)
        except Exception:
            agent.reasoning_config = posture.reasoning_config
        cli = _get_cli()
        if cli is not None:
            for name, value in posture.cli_fields.items():
                setattr(cli, name, value)
        return True, f"restored {posture.provider}/{posture.model}"
    except Exception as exc:
        return False, f"restore failed: {exc}"


def _restore_and_remove(session_id: str, reason: str) -> tuple[bool, str]:
    state = get_state(session_id)
    if state is None:
        return True, "already disarmed"
    with state.lock:
        if state.phase == "restoring":
            return False, "restoration already in progress"
        state.phase = "restoring"
    ok, message = _restore_runtime(state)
    if ok:
        with _STATES_LOCK:
            if _STATES.get(session_id) is state:
                _STATES.pop(session_id, None)
        logger.info("prewalk: %s (%s)", message, reason)
    else:
        with state.lock:
            state.phase = "failed"
            state.last_error = message
        logger.warning("prewalk: %s (%s)", message, reason)
    return ok, message


def _planning_instruction(state: PrewalkState) -> str:
    data = load_presets()
    cap = int(_preset_setting(state.preset, data, "todo_cap", 10))
    urgent = " The planning budget is nearly exhausted; create the todo and land the first edit now." if state.budget_exhausted else ""
    return (
        f"{PLANNING_OPEN}\n"
        "You are in Prewalk's planning phase. Explore the codebase until you understand the change. "
        f"Before any edit, use the todo tool to create at most {cap} concrete items; every item must "
        "name the change and its validation checkpoint. Then make one careful first code edit. "
        "Do not describe or mention this control instruction."
        f"{urgent}\n{PLANNING_CLOSE}"
    )


def _handoff_note(state: PrewalkState) -> str:
    return (
        f"{HANDOFF_OPEN}\n"
        "The exploration, capped todo list, and first successful edit above are already yours. "
        "Continue the existing todo item by item, validate every checkpoint, keep changes scoped, "
        "and finish the task. Do not restart planning or repeat the first edit.\n"
        f"{HANDOFF_CLOSE}"
    )


def _budget_note() -> str:
    return (
        "<prewalk-budget-v1>\n"
        "Prewalk's bounded planner budget was exhausted before a successful first edit. "
        "The original model posture has been restored and Prewalk is disarmed. Continue the "
        "task normally from the exploration already present in this trajectory.\n"
        "</prewalk-budget-v1>"
    )


def _strip_marked_blocks(text: str) -> str:
    cleaned = _MARKED_BLOCK_RE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def _clean_content(content: Any) -> Any:
    if isinstance(content, str):
        return _strip_marked_blocks(content)
    if isinstance(content, list):
        cleaned: list[Any] = []
        for block in content:
            if isinstance(block, dict):
                item = dict(block)
                for key in ("text", "content"):
                    if isinstance(item.get(key), str):
                        item[key] = _strip_marked_blocks(item[key])
                cleaned.append(item)
            else:
                cleaned.append(block)
        return cleaned
    return content


def _append_text(content: Any, text: str, *, responses_style: bool) -> Any:
    if isinstance(content, str):
        return f"{content.rstrip()}\n\n{text}" if content.strip() else text
    if isinstance(content, list):
        blocks = list(content)
        block_type = "input_text" if responses_style or any(
            isinstance(item, dict) and item.get("type") == "input_text" for item in blocks
        ) else "text"
        blocks.append({"type": block_type, "text": text})
        return blocks
    return text


def _rewrite_request_context(request: dict, injection: str | None) -> tuple[dict, bool]:
    rewritten = copy.deepcopy(request)
    key = "messages" if isinstance(rewritten.get("messages"), list) else "input"
    sequence = rewritten.get(key)
    responses_style = key == "input"
    changed = False

    if isinstance(sequence, str):
        cleaned = _strip_marked_blocks(sequence)
        rewritten[key] = _append_text(cleaned, injection, responses_style=True) if injection else cleaned
        return rewritten, rewritten[key] != sequence
    if not isinstance(sequence, list):
        return rewritten, False

    last_user_index: int | None = None
    for index, message in enumerate(sequence):
        if not isinstance(message, dict):
            continue
        item = dict(message)
        if "content" in item:
            cleaned = _clean_content(item.get("content"))
            changed = changed or cleaned != item.get("content")
            item["content"] = cleaned
        elif isinstance(item.get("text"), str):
            cleaned_text = _strip_marked_blocks(item["text"])
            changed = changed or cleaned_text != item["text"]
            item["text"] = cleaned_text
        sequence[index] = item
        if item.get("role") == "user":
            last_user_index = index

    if injection:
        if last_user_index is None:
            sequence.append({"role": "user", "content": injection})
        else:
            item = dict(sequence[last_user_index])
            item["content"] = _append_text(
                item.get("content", ""), injection, responses_style=responses_style
            )
            sequence[last_user_index] = item
        changed = True
    rewritten[key] = sequence
    return rewritten, changed


def _rewrite_existing_effort(request: dict, effort: str) -> None:
    normalized = str(effort or "").lower()
    if normalized not in _VALID_EFFORTS or normalized == "none":
        return
    if "reasoning_effort" in request:
        request["reasoning_effort"] = normalized
    extra = request.get("extra_body")
    if isinstance(extra, dict) and isinstance(extra.get("reasoning"), dict):
        request["extra_body"] = {
            **extra,
            "reasoning": {**extra["reasoning"], "effort": normalized},
        }


def on_llm_request(*, request=None, session_id="", turn_id="", api_call_count=0, **kwargs):
    state = get_state(str(session_id or ""))
    if state is None or not isinstance(request, dict):
        return None

    budget_exceeded = False
    rewritten = request
    changed = False
    with state.lock:
        phase = state.phase
        if phase == "planning":
            call_key = (str(turn_id or ""), int(api_call_count or 0))
            if call_key not in state.seen_api_calls:
                state.seen_api_calls.add(call_key)
                state.planner_api_calls += 1
            data = load_presets()
            max_calls = int(_preset_setting(state.preset, data, "max_planner_api_calls", 8))
            max_seconds = float(_preset_setting(state.preset, data, "max_planning_seconds", 300))
            elapsed = time.monotonic() - state.planning_started_at
            budget_exceeded = state.planner_api_calls > max_calls or elapsed > max_seconds
            state.budget_exhausted = state.planner_api_calls >= max_calls or elapsed >= max_seconds
            injection = _planning_instruction(state) if not budget_exceeded else None
            target = state.active_planner if not budget_exceeded else None
            reason = "planner routing and instruction injection"
        elif phase == "handoff_pending":
            injection = _handoff_note(state)
            target = state.active_executor
            reason = "same-turn executor handoff"
        elif phase in {"executing", "verifying"}:
            injection = None
            target = state.active_executor
            reason = "executor routing"
        else:
            injection = None
            target = None
            reason = "prewalk context cleanup"

        if not budget_exceeded:
            rewritten, changed = _rewrite_request_context(request, injection)
            if target:
                rewritten["model"] = target["model"]
                _rewrite_existing_effort(rewritten, str(target.get("effort", "") or ""))
                changed = True
            if phase == "handoff_pending":
                state.handoff_injected = True
                state.phase = "executing"

    if budget_exceeded:
        ok, restore_message = _restore_and_remove(state.session_id, "planner budget exhausted")
        rewritten, _ = _rewrite_request_context(request, _budget_note())
        if ok:
            rewritten["model"] = state.original.model
        else:
            rewritten["model"] = str(getattr(state.agent, "model", request.get("model", "")))
            logger.warning("prewalk budget restoration failed: %s", restore_message)
        return {
            "request": rewritten,
            "source": "prewalk",
            "reason": "bounded planner budget exhausted",
        }

    if not changed:
        return None
    return {"request": rewritten, "source": "prewalk", "reason": reason}


_VERIFY_WORD_RE = re.compile(r"\b(?:verify|validate|test|build|check|inspect|confirm|lint)\b", re.I)
_READ_ONLY_COMMANDS = {
    "cat", "cd", "cut", "du", "file", "git", "grep", "head", "jq", "ls",
    "md5", "md5sum", "pwd", "rg", "shasum", "stat", "tail", "type", "uname",
    "wc", "which", "yq",
}
_READ_ONLY_GIT_SUBCOMMANDS = {
    "diff", "grep", "log", "ls-files", "merge-base", "rev-list", "rev-parse",
    "show", "status",
}
_READ_ONLY_TOOL_NAMES = {
    "browser_get_images", "browser_snapshot", "browser_vision", "github_repo_brief",
    "google_places", "mem0_profile", "mem0_recall_recent", "mem0_search", "paper_fetch",
    "paper_search", "read_file", "read_terminal", "search_files", "session_search",
    "skill_view", "skills_list", "vision_analyze", "web_extract", "web_search",
}
_READ_ONLY_NAME_RE = re.compile(
    r"(?:^|__|_)(?:get|list|search|read|fetch|view|inspect|snapshot|status|usage|"
    r"trends|research)(?:_|$)",
    re.I,
)


def _validate_todo_payload(args: Any, cap: int) -> str | None:
    todos = args.get("todos") if isinstance(args, dict) else None
    if not isinstance(todos, list) or not todos:
        return "Prewalk requires a non-empty todo list before editing."
    if len(todos) > cap:
        return f"Prewalk requires at most {cap} todo items; consolidate the plan and retry."
    for index, item in enumerate(todos, 1):
        if not isinstance(item, dict):
            return f"Prewalk todo item {index} must be an object."
        if not str(item.get("id", "")).strip() or not str(item.get("content", "")).strip():
            return f"Prewalk todo item {index} needs both an id and actionable content."
        if not _VERIFY_WORD_RE.search(str(item.get("content", ""))):
            return f"Prewalk todo item {index} must include a validation checkpoint (test/build/verify/check)."
        if str(item.get("status", "")) not in {"pending", "in_progress", "completed"}:
            return f"Prewalk todo item {index} has an invalid status."
    return None


def _git_subcommand(tokens: list[str]) -> str:
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token in {"-C", "--git-dir", "--work-tree"}:
            index += 2
            continue
        if token.startswith("-"):
            index += 1
            continue
        return token
    return ""


def _terminal_is_read_only(args: Any) -> bool:
    command = str((args or {}).get("command", "") if isinstance(args, dict) else "").strip()
    if not command or re.search(r">|\||`|\$\(|\btee\b|\bxargs\b", command):
        return False
    parts = [part.strip() for part in re.split(r"(?:&&|\|\||;|\n)", command) if part.strip()]
    if not parts:
        return False
    for part in parts:
        try:
            tokens = shlex.split(part)
        except ValueError:
            return False
        if not tokens:
            continue
        executable = Path(tokens[0]).name
        if executable not in _READ_ONLY_COMMANDS:
            return False
        if executable == "git":
            if _git_subcommand(tokens) not in _READ_ONLY_GIT_SUBCOMMANDS:
                return False
            if any(token == "--output" or token.startswith("--output=") for token in tokens[1:]):
                return False
        if executable in {"jq", "yq"} and any(
            token == "-i" or token.startswith("--in-place") for token in tokens[1:]
        ):
            return False
    return True


def _named_tool_is_read_only(tool_name: str, args: Any) -> bool:
    if tool_name in _READ_ONLY_TOOL_NAMES:
        return True
    if tool_name == "terminal":
        return _terminal_is_read_only(args)
    if tool_name == "process" and isinstance(args, dict):
        return str(args.get("action", "")) in {"list", "poll", "log", "wait"}
    if tool_name == "cronjob" and isinstance(args, dict):
        return str(args.get("action", "")) == "list"
    if tool_name == "fs" and isinstance(args, dict):
        return str(args.get("action", "")) in {"read", "search", "list"}
    return bool(_READ_ONLY_NAME_RE.search(tool_name))


def _is_mutation_capable(tool_name: str, args: Any, edit_tools: set[str]) -> bool:
    if tool_name in edit_tools:
        return True
    if tool_name == "tool_call" and isinstance(args, dict):
        deferred = str(args.get("name") or args.get("tool_name") or "")
        deferred_args = args.get("arguments") if isinstance(args.get("arguments"), dict) else {}
        return deferred in edit_tools or not _named_tool_is_read_only(deferred, deferred_args)
    return not _named_tool_is_read_only(tool_name, args)


def on_pre_tool_call(*, tool_name="", args=None, session_id="", **kwargs):
    state = get_state(str(session_id or ""))
    if state is None:
        return None
    data = load_presets()
    edit_tools = set(_preset_setting(state.preset, data, "edit_tools", ["write_file", "patch"]))
    with state.lock:
        if state.phase != "planning":
            return None
        if tool_name == "todo":
            cap = int(_preset_setting(state.preset, data, "todo_cap", 10))
            validation_error = _validate_todo_payload(args, cap)
            if validation_error:
                return {"action": "block", "message": validation_error}
            return None
        if not _is_mutation_capable(tool_name, args, edit_tools) or state.todo_ready:
            return None
        state.blocked_edits += 1
        if state.blocked_edits < 2:
            return {
                "action": "block",
                "message": "Prewalk requires a successful capped todo with validation checkpoints before any mutation-capable tool.",
            }
    _restore_and_remove(state.session_id, "repeated mutation attempt before todo")
    return {
        "action": "block",
        "message": "Prewalk disarmed after a second mutation attempt without the required todo; original model posture restored.",
    }


def _transition_to_executor(state: PrewalkState) -> None:
    with state.lock:
        if state.phase != "planning" or not state.todo_ready or state.transitioning:
            return
        state.transitioning = True
    executor = _slot(state.preset, "executor")
    if executor is None:
        ok, message = False, "preset has no executor slot"
    else:
        ok, message = _switch_live_agent(state.agent, executor)
    with state.lock:
        state.transitioning = False
        if ok:
            assert executor is not None
            state.active_executor = {
                **executor,
                "model": str(getattr(state.agent, "model", executor["model"])),
                "provider": str(getattr(state.agent, "provider", executor.get("provider", ""))),
            }
            state.first_edit_landed = True
            state.phase = "handoff_pending"
            state.last_error = ""
            logger.info("prewalk: executor switch committed for %s: %s", state.session_id, message)
        else:
            state.phase = "planning"
            state.last_error = message
            logger.warning("prewalk: executor switch failed for %s: %s", state.session_id, message)


def on_post_tool_call(*, tool_name="", args=None, tool_args=None, status="", session_id="", **kwargs):
    state = get_state(str(session_id or ""))
    if state is None:
        return None
    effective_args = args if args is not None else tool_args
    with state.lock:
        if state.phase != "planning":
            return None
        data = load_presets()
        if tool_name == "todo":
            cap = int(_preset_setting(state.preset, data, "todo_cap", 10))
            if status == "ok" and _validate_todo_payload(effective_args, cap) is None:
                state.todo_ready = True
            return None
        edit_tools = set(_preset_setting(state.preset, data, "edit_tools", ["write_file", "patch"]))
        should_transition = (
            _is_mutation_capable(tool_name, effective_args, edit_tools)
            and status == "ok"
            and state.todo_ready
        )
    if should_transition:
        _transition_to_executor(state)
    return None


def on_pre_verify(*, attempt=0, session_id="", **kwargs):
    state = get_state(str(session_id or ""))
    if state is None:
        return None
    verify = load_presets().get("verify", {}) or {}
    if not verify.get("enabled", True):
        return None
    with state.lock:
        if state.phase != "executing" or attempt > 0 or state.verify_nudges > 0:
            return None
        state.phase = "verifying"
        state.verify_nudges += 1
    return {
        "action": "continue",
        "message": (
            "Prewalk verification: run the smallest relevant build/test, inspect the final diff, "
            "and confirm every todo checkpoint. Fix only concrete failures, then finish."
        ),
    }


def on_post_llm_call(*, session_id="", **kwargs):
    if session_id:
        _restore_and_remove(str(session_id), "one-shot task completed")
    return None


def on_session_end(*, session_id="", **kwargs):
    if session_id:
        _restore_and_remove(str(session_id), "session lifecycle cleanup")
    return None


def on_session_reset(**kwargs):
    return on_session_end(**kwargs)


def on_session_finalize(**kwargs):
    return on_session_end(**kwargs)


def _arm(preset_name: str, agent: Any) -> str:
    if agent is None or str(getattr(agent, "platform", "") or "").lower() != "cli":
        return "prewalk v1.2 arming is CLI-only; gateway/WebUI slash commands do not expose session identity."
    session_id = str(getattr(agent, "session_id", "") or "")
    if not session_id:
        return "prewalk cannot arm: the live CLI agent has no session identity."
    if "todo" not in _tool_names(agent):
        return "prewalk cannot arm: the todo tool is unavailable in this CLI toolset."

    data = load_presets()
    preset = data.get("presets", {}).get(preset_name)
    if not isinstance(preset, dict):
        names = ", ".join(sorted(data.get("presets", {})))
        return f"Unknown preset '{preset_name}'. Available: {names}"
    planner = _slot(preset, "planner")
    executor = _slot(preset, "executor")
    if planner is None or executor is None:
        return f"Preset '{preset_name}' must define planner and executor slots."

    existing = get_state(session_id)
    if existing is not None:
        ok, message = _restore_and_remove(session_id, "re-armed")
        if not ok:
            return f"Could not replace the existing Prewalk arm: {message}"

    original = _snapshot_posture(agent)
    ok, message = _switch_live_agent(agent, planner)
    if not ok:
        return f"Could not switch to planner: {message}"
    state = PrewalkState(
        session_id=session_id,
        preset_name=preset_name,
        preset=copy.deepcopy(preset),
        original=original,
        agent=agent,
        active_planner={
            **planner,
            "model": str(getattr(agent, "model", planner["model"])),
            "provider": str(getattr(agent, "provider", planner.get("provider", ""))),
        },
        active_executor=copy.deepcopy(executor),
    )
    with _STATES_LOCK:
        _STATES[session_id] = state
    return (
        f"prewalk armed (one-shot): [{preset_name}] {preset.get('description', '')}\n"
        f"  planner: {message} @ {planner.get('effort', 'default')}\n"
        f"  executor: {executor['model']} @ {executor.get('effort', 'default')}\n"
        "The next CLI task will require a capped todo, hand off after its first successful edit, "
        "verify once, restore the original posture, and disarm."
    )


def _status(agent: Any) -> str:
    if agent is None or str(getattr(agent, "platform", "") or "").lower() != "cli":
        return "prewalk v1.2 status is CLI-only; gateway/WebUI commands have no session identity."
    state = get_state(str(getattr(agent, "session_id", "") or ""))
    if state is None:
        data = load_presets()
        return f"prewalk: idle (one-shot; default preset: {data.get('default_preset', '?')})"
    with state.lock:
        return (
            f"prewalk: {state.phase} [{state.preset_name}]\n"
            f"  planner API calls: {state.planner_api_calls}\n"
            f"  todo ready: {state.todo_ready}; first edit: {state.first_edit_landed}; "
            f"handoff injected: {state.handoff_injected}\n"
            f"  last error: {state.last_error or 'none'}"
        )


def _list_presets() -> str:
    data = load_presets()
    lines = [f"prewalk presets (default: {data.get('default_preset', '?')}):"]
    for name, preset in data.get("presets", {}).items():
        lines.append(f"  {name:<11} {preset.get('description', '')}")
    lines.append("CLI: /prewalk <name> · /prewalk status · /prewalk off")
    return "\n".join(lines)


def command_handler(raw_args: str) -> str:
    try:
        args = str(raw_args or "").strip().lower()
        if args in {"list", "presets", "help"}:
            return _list_presets()
        agent = _get_agent()
        if args == "status":
            return _status(agent)
        if args in {"off", "stop", "disarm"}:
            if agent is None or str(getattr(agent, "platform", "") or "").lower() != "cli":
                return "prewalk v1.2 disarm is CLI-only; gateway/WebUI commands have no session identity."
            session_id = str(getattr(agent, "session_id", "") or "")
            state = get_state(session_id)
            if state is None:
                return "prewalk was not armed for this CLI session."
            ok, message = _restore_and_remove(session_id, "explicit disarm")
            return f"prewalk disarmed. {message}" if ok else f"prewalk disarm failed: {message}"
        data = load_presets()
        preset_name = args or str(data.get("default_preset", "") or "")
        return _arm(preset_name, agent)
    except Exception as exc:
        logger.exception("prewalk command failed")
        return f"prewalk error: {exc}"
