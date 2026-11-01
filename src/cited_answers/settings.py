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
    # Model for the input classifier. None means "use the answering model": the classifier
    # is a small, cheap call, and so is the model the app is already configured with.
    classifier_model: str | None = None

    @property
    def guard_model(self) -> str:
        return self.classifier_model or self.model

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
            classifier_model=os.environ.get("ANTHROPIC_CLASSIFIER_MODEL", "").strip() or None,
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


class ClassifierConfig(BaseModel):
    prompt_file: str  # relative to the project root
    max_tokens: int = Field(gt=0)

    @property
    def prompt_path(self) -> Path:
        return PROJECT_ROOT / self.prompt_file


class RateLimitConfig(BaseModel):
    requests: int = Field(gt=0)  # allowed per client IP ...
    window_seconds: float = Field(gt=0)  # ... in this many seconds


class GuardrailsConfig(BaseModel):
    retry_max_iterations: int = Field(gt=0)  # model calls allowed for the one retry
    request_timeout_seconds: float = Field(gt=0)  # whole-request deadline
    max_total_output_tokens: int = Field(gt=0)  # across every model call in one request
    classifier: ClassifierConfig
    rate_limit: RateLimitConfig


class AppConfig(BaseModel):
    retrieval: RetrievalConfig
    llm: LlmConfig
    verification: VerificationConfig
    answer: AnswerConfig
    guardrails: GuardrailsConfig


def load_app_config(path: Path = APP_CONFIG_FILE) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text()))


@lru_cache(maxsize=1)
def get_app_config() -> AppConfig:
    """The shipped config, read once per process."""
    return load_app_config()


# ---- demo protections --------------------------------------------------------------------
# Environment overrides for a public deployment. Unset means "use config/app.yaml" (rate
# limit) or "no cap" (daily cap), so local development is unchanged.


def _env_number(name: str, kind: type[int] | type[float], minimum: float) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = kind(raw)
    except ValueError:
        raise SettingsError(f"{name} must be a number, got {raw!r}.") from None
    if value < minimum:
        raise SettingsError(f"{name} must be at least {minimum:g}, got {raw}.")
    return value


def rate_limit_from_env(base: RateLimitConfig) -> RateLimitConfig:
    """The per-IP limit: RATE_LIMIT_REQUESTS per RATE_LIMIT_WINDOW_SECONDS, else the config."""
    requests = _env_number("RATE_LIMIT_REQUESTS", int, 1)
    window = _env_number("RATE_LIMIT_WINDOW_SECONDS", float, 1)
    return RateLimitConfig(
        requests=int(requests) if requests is not None else base.requests,
        window_seconds=window if window is not None else base.window_seconds,
    )


def daily_ask_cap_from_env() -> int | None:
    """DAILY_ASK_CAP: most /ask calls per UTC day across all clients. None means no cap."""
    cap = _env_number("DAILY_ASK_CAP", int, 0)
    return int(cap) if cap else None  # 0 or unset: no cap


def trusted_proxy_hops() -> int:
    """TRUST_PROXY_HOPS: how many proxies in front of the app append to X-Forwarded-For.

    0 (the default) ignores the header, which is right when clients connect directly.
    Cloud Run without a load balancer in front is 1.
    """
    return int(_env_number("TRUST_PROXY_HOPS", int, 0) or 0)
