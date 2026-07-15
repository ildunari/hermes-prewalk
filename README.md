# Hermes Prewalk

A one-shot, same-trajectory planner-to-executor handoff for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

Prewalk gives one coding task to a high-capability planner long enough to inspect the codebase, create a bounded implementation checklist, and land the first successful edit. On Hermes's **very next provider call in the same user turn**, the plugin removes its planning instruction, preserves the complete tool trajectory, inserts one handoff note, and routes the request to a faster executor. When the task finishes, it restores the original model posture and disarms.

The design is inspired by Stencil's [“Prewalk: A Simple Trick for Better Coding Agents”](https://stencil.so/blog/prewalk).

## Why this implementation is different

The handoff is implemented in Hermes's `llm_request` middleware, which runs before every provider request. The middleware owns both injection and removal, so it never depends on Hermes persisting or reattaching plugin user context. The release test captures actual provider-bound requests from a real multi-call `AIAgent.run_conversation()` turn and asserts:

- planning guidance appears exactly once on every planner request;
- the first successful edit is preceded by a successful `todo` call;
- the very next same-turn request uses the executor;
- the executor receives the complete read/todo/edit trajectory;
- planning guidance is absent from that executor request;
- one handoff note is present once;
- the original model posture is restored when the task completes.

## Requirements

- Hermes Agent **v0.18.2 or newer**
- An interactive Hermes **CLI** session
- The `todo` tool enabled
- Credentials for the planner and executor models selected by your preset

> [!IMPORTANT]
> Prewalk v1.2 deliberately supports arming, status, and disarming only in the interactive CLI. Hermes gateway plugin slash-command handlers currently receive only the raw argument string—not a session identity—so pretending gateway arming is session-safe would create cross-session races. Middleware and hooks are session-isolated once armed, but gateway arming remains out of scope until Hermes exposes command context.

## Install

### Hermes plugin installer

The plugin lives in the repository's `prewalk/` subdirectory:

```bash
hermes plugins install ildunari/hermes-prewalk/prewalk --enable
```

Restart or open a new interactive Hermes CLI session after installation so Hermes loads the plugin.

To update later:

```bash
hermes plugins update prewalk
```

### pip / Git install

The package also exposes a `hermes_agent.plugins` entry point:

```bash
python -m pip install 'git+https://github.com/ildunari/hermes-prewalk.git'
```

### Manual install

Copy or symlink `prewalk/` into the active profile's plugin directory and enable `prewalk` in `plugins.enabled`:

```text
$HERMES_HOME/plugins/prewalk/
├── __init__.py
├── core.py
├── plugin.yaml
└── presets.yaml
```

## Use

Start an interactive CLI session, then arm one task:

```text
/prewalk
```

The default preset is `code-value`. You can choose another preset explicitly:
```text
/prewalk code-value
/prewalk code-max
/prewalk speed
/prewalk budget
```

Other commands:

```text
/prewalk list
/prewalk status
/prewalk off
```

After arming, send the coding task normally. Prewalk is one-shot by default: completion restores the exact original provider/model/API posture and reasoning configuration, then removes the session state.

## Presets

The bundled presets are editable YAML examples. Model aliases such as
`gpt-5.6-sol`, `gpt-5.6-luna`, and `claude-fable-5` are available in the
environment this plugin was developed for but may not exist in a stock Hermes
installation. Run `/prewalk list`, then edit `presets.yaml` to use planner and
executor models your Hermes profile can resolve. Prewalk fails before arming if
no planner candidate can be resolved.

Edit `prewalk/presets.yaml` to add or change routes. Each preset needs planner and executor slots:

```yaml
presets:
  my-route:
    description: "My planner/executor route"
    planner:
      provider: anthropic
      model: claude-opus-4-6
      effort: high
    executor:
      provider: openai-codex
      model: gpt-5.5-codex
      effort: low
```

Optional slot `fallbacks` are ordered `[provider, model]` pairs. Global correctness controls:

```yaml
settings:
  edit_tools: [write_file, patch]
  todo_cap: 10
  max_planner_api_calls: 8
  max_planning_seconds: 300
verify:
  enabled: true
```

The planner budget counts actual provider calls via Hermes's middleware `api_call_count`, not user messages. It also maintains a monotonic elapsed-time cap.

## Safety and failure behavior

- State is keyed by Hermes `session_id`, not a process-global active flag.
- Arming fails when no CLI session identity or `todo` tool is available.
- Edit tools are blocked until a successful capped todo exists.
- A second premature edit attempt restores the original posture and disarms instead of looping.
- The executor phase commits only after the live model switch succeeds.
- A failed switch leaves the planner phase intact and records the error.
- Task completion, explicit `/prewalk off`, reset, finalize, and session end all attempt exact restoration.
- Unsupported gateway/WebUI command use returns a truthful CLI-only message and creates no state.

## Development and verification

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The local release suite includes unit/state tests plus a trajectory-level integration test against a local mock OpenAI-compatible server. The integration test requires a Hermes Agent source checkout at `~/.hermes/hermes-agent` or `HERMES_AGENT_ROOT`.

Compatibility matrix used for v1.2:

| Hermes surface | Revision | Result |
| --- | --- | --- |
| NousResearch v0.18.2 upstream `main` | `da4a28ec6db3c1e391db9abde20a60c828fa322e` | Full same-turn trajectory passes |

CI pins the upstream revision instead of silently testing against a moving host
lifecycle.

## License

MIT. See [LICENSE](LICENSE).
