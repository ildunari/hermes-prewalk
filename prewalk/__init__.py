"""Prewalk — correct same-turn planner/executor handoff for Hermes Agent."""

from . import core


def register(ctx):
    """Register the CLI-only one-shot command and session-aware lifecycle hooks."""
    core.CTX = ctx

    ctx.register_hook("pre_tool_call", core.on_pre_tool_call)
    ctx.register_hook("post_tool_call", core.on_post_tool_call)
    ctx.register_hook("pre_verify", core.on_pre_verify)
    ctx.register_hook("post_llm_call", core.on_post_llm_call)
    ctx.register_hook("on_session_end", core.on_session_end)
    ctx.register_hook("on_session_reset", core.on_session_reset)
    ctx.register_hook("on_session_finalize", core.on_session_finalize)
    ctx.register_middleware("llm_request", core.on_llm_request)

    ctx.register_command(
        "prewalk",
        core.command_handler,
        description=(
            "CLI-only one-shot Prewalk: frontier planner explores, writes a capped todo, "
            "lands the first edit, then hands the same trajectory to a faster executor."
        ),
        args_hint="<preset>|list|status|off",
    )
