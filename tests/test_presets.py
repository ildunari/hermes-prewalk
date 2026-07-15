from pathlib import Path

import yaml


def test_bundled_primary_executor_routes():
    data = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "prewalk" / "presets.yaml").read_text(
            encoding="utf-8"
        )
    )
    presets = data["presets"]
    expected = {
        "apple": ("anthropic", "claude-opus-4-8", "medium"),
        "sci": ("anthropic", "claude-opus-4-8", "medium"),
        "code-max": ("anthropic", "claude-fable-5", "medium"),
        "frontend": ("anthropic", "claude-opus-4-8", "medium"),
    }

    actual = {
        name: tuple(presets[name]["executor"][key] for key in ("provider", "model", "effort"))
        for name in expected
    }
    assert actual == expected
