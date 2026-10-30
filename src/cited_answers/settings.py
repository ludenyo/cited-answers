"""Runtime settings read from environment variables."""

import os
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

APP_CONFIG_FILE = Path(__file__).resolve().parents[2] / "config" / "app.yaml"

# A small, cheap model by default; override with ANTHROPIC_MODEL.
DEFAULT_MODEL = "claude-haiku-5-5"
# Thinking tokens count against max_tokens on current models, so leave headroom.
DEFAULT_MAX_TOKENS = 2048


@dataclass(frozen=True)
class Settings:
    model: str
    max_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        # ANTHROPIC_API_KEY is deliberately not read here: the SDK picks it up
        # from the environment itself, so it never passes through our code.
        return cls(
            model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
            max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )


class VerificationConfig(BaseModel):
    # Decimal, not float, so a claim exactly on the boundary is not decided by float noise.
    abs_tolerance: Decimal
    rel_tolerance: Decimal


class AppConfig(BaseModel):
    verification: VerificationConfig


def load_app_config(path: Path = APP_CONFIG_FILE) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text()))


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """The shipped config, read once per process."""
    return load_app_config()
