"""Runtime settings: environment variables (Settings) and config/app.yaml (AppConfig)."""

import os
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_CONFIG_FILE = PROJECT_ROOT / "config" / "app.yaml"
# Thinking tokens count against max_tokens on current models, so leave headroom.
DEFAULT_MAX_TOKENS = 2048


class SettingsError(RuntimeError):
    """A required setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    model: str
    max_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        # ANTHROPIC_API_KEY is deliberately not read here: the SDK picks it up
        # from the environment itself, so it never passes through our code.
        model = os.environ.get("ANTHROPIC_MODEL", "").strip()
        if not model:
            # No default on purpose: the model is a cost and behaviour decision
            # that should be explicit, not silently baked into the code.
            raise SettingsError(
                "ANTHROPIC_MODEL is not set. Add it to .env (see .env.example), "
                "for example ANTHROPIC_MODEL=claude-haiku-5-5."
            )
        return cls(
            model=model,
            max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )


class RetrievalConfig(BaseModel):
    top_k: int = Field(gt=0)
    max_distance: float = Field(gt=0)  # cosine distance; results worse than this are dropped


class LlmConfig(BaseModel):
    # No temperature setting: Haiku 5.5 rejects any non-default value, see CLAUDE.md.
    max_tool_iterations: int = Field(gt=0)


class VerificationConfig(BaseModel):
    # Decimal, not float, so a claim exactly on the boundary is not decided by float noise.
    abs_tolerance: Decimal
    rel_tolerance: Decimal


class AnswerConfig(BaseModel):
    prompt_file: str  # relative to the project root

    @property
    def prompt_path(self) -> Path:
        return PROJECT_ROOT / self.prompt_file


class AppConfig(BaseModel):
    retrieval: RetrievalConfig
    llm: LlmConfig
    verification: VerificationConfig
    answer: AnswerConfig


def load_app_config(path: Path = APP_CONFIG_FILE) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text()))


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """The shipped config, read once per process."""
    return load_app_config()
