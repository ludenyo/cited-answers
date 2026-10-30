from decimal import Decimal
from pathlib import Path

import pytest

from cited_answers.settings import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    Settings,
    get_app_config,
    load_app_config,
)


def test_settings_defaults_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)
    settings = Settings.from_env()
    assert settings.model == DEFAULT_MODEL == "claude-haiku-5-5"
    assert settings.max_tokens == DEFAULT_MAX_TOKENS


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-other-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")
    settings = Settings.from_env()
    assert (settings.model, settings.max_tokens) == ("some-other-model", 4096)


def test_shipped_app_yaml_has_the_specified_tolerances() -> None:
    config = get_app_config().verification
    assert config.abs_tolerance == Decimal("0.05")
    assert config.rel_tolerance == Decimal("0.005")


def test_app_config_loads_from_a_given_path(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text("verification:\n  abs_tolerance: 0.1\n  rel_tolerance: 0.01\n")
    config = load_app_config(path).verification
    assert (config.abs_tolerance, config.rel_tolerance) == (Decimal("0.1"), Decimal("0.01"))
