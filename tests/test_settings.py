from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cited_answers.settings import (
    DEFAULT_MAX_TOKENS,
    PROJECT_ROOT,
    Settings,
    SettingsError,
    get_app_config,
    load_app_config,
)


def test_missing_model_fails_with_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    with pytest.raises(SettingsError, match="ANTHROPIC_MODEL is not set.*\\.env"):
        Settings.from_env()


def test_blank_model_counts_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")
    with pytest.raises(SettingsError):
        Settings.from_env()


def test_settings_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "4096")
    settings = Settings.from_env()
    assert (settings.model, settings.max_tokens) == ("some-model", 4096)


def test_max_tokens_defaults_to_2048(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
    monkeypatch.delenv("ANTHROPIC_MAX_TOKENS", raising=False)
    assert Settings.from_env().max_tokens == DEFAULT_MAX_TOKENS == 2048


def test_shipped_app_yaml_has_every_specified_value() -> None:
    config = get_app_config()
    assert config.retrieval.top_k == 10
    assert config.retrieval.max_distance == 0.6
    assert config.llm.max_tool_iterations == 5
    assert config.verification.abs_tolerance == Decimal("0.05")
    assert config.verification.rel_tolerance == Decimal("0.001")
    assert config.answer.prompt_file == "prompts/answer_system.md"
    assert config.answer.prompt_path == PROJECT_ROOT / "prompts" / "answer_system.md"
    assert config.answer.prompt_path.is_file()


def test_there_is_no_temperature_setting() -> None:
    # Haiku 5.5 rejects non-default temperatures, so the config must not offer one.
    assert "temperature" not in get_app_config().llm.model_dump()


def test_app_config_loads_from_a_given_path(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "retrieval: {top_k: 3, max_distance: 0.4}\n"
        "llm: {max_tool_iterations: 2}\n"
        "verification: {abs_tolerance: 0.1, rel_tolerance: 0.01}\n"
        "answer: {prompt_file: p.md}\n"
    )
    config = load_app_config(path)
    assert (config.retrieval.top_k, config.llm.max_tool_iterations) == (3, 2)
    assert config.verification.rel_tolerance == Decimal("0.01")


def test_invalid_values_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "retrieval: {top_k: 0, max_distance: 0.4}\n"
        "llm: {max_tool_iterations: 2}\n"
        "verification: {abs_tolerance: 0.1, rel_tolerance: 0.01}\n"
        "answer: {prompt_file: p.md}\n"
    )
    with pytest.raises(ValidationError):
        load_app_config(path)
