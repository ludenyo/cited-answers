"""Guardrails around the answering pipeline.

Input:   classify_question labels a question in_scope, out_of_scope or suspicious.
Output:  failed claims are fed back for one retry; if they still fail they are dropped
         and the answer is marked partial.
Limits:  a whole-request Deadline, and a per-IP RateLimiter. The iteration and output
         token limits are enforced by the loop in answer.py using the messages here.

Every trigger is recorded as a Langfuse event by record_trigger, and also logged.
"""

import hashlib
import html
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from cited_answers import tracing
from cited_answers.models import VerifiedClaim
from cited_answers.settings import GuardrailsConfig, Settings
from cited_answers.trace_format import messages_for_trace, usage_for_trace

logger = logging.getLogger(__name__)

# Names that appear in AskResponse.guardrails, in Langfuse events and in the eval file.
INPUT_OUT_OF_SCOPE = "input_out_of_scope"
INPUT_SUSPICIOUS = "input_suspicious"
CLASSIFIER_ERROR = "classifier_error"  # the classifier failed, so the question went through
OUTPUT_RETRY = "output_retry"
OUTPUT_PARTIAL = "output_partial_answer"
LIMIT_ITERATIONS = "limit_iterations"
LIMIT_TIMEOUT = "limit_timeout"
LIMIT_OUTPUT_TOKENS = "limit_output_tokens"
RATE_LIMITED = "rate_limited"

EVENT_NAME = "guardrail-triggered"
MAX_LISTED = 5  # claims named in a retry message or a partial-answer explanation

# Why a loop stopped early -> the guardrail that records it, and what the user is told.
STOP_TRIGGERS = {
    "iterations": LIMIT_ITERATIONS,
    "timeout": LIMIT_TIMEOUT,
    "output_tokens": LIMIT_OUTPUT_TOKENS,
}
STOP_MESSAGES = {
    "iterations": "The question could not be answered within the allowed steps.",
    "timeout": "The request took too long and was stopped. Please try a simpler question.",
    "output_tokens": "The answer needed more output than is allowed, so it was stopped.",
}

Label = Literal["in_scope", "out_of_scope", "suspicious"]
RefusalReason = Literal["out_of_scope", "suspicious"]

REFUSALS: dict[RefusalReason, str] = {
    "out_of_scope": (
        "I can only answer questions about development indicators for Kenya, Uganda and "
        "Tanzania, such as GDP, population, health, education and access to electricity, "
        "using World Bank data. I can't help with that question, but I'm happy to answer "
        "one on those topics."
    ),
    # Deliberately says nothing about what was detected.
    "suspicious": (
        "I can't help with that request. I can answer questions about development "
        "indicators for Kenya, Uganda and Tanzania from World Bank data, and I cite the "
        "figures I use."
    ),
}


# ---- recording --------------------------------------------------------------------------


def record_trigger(name: str, *, input: Any = None, **details: Any) -> None:
    """Log a guardrail trigger and record it in Langfuse as a warning event.

    `input` is the user's text, which goes through Langfuse's masking. Everything in
    `details` becomes event metadata, which is NOT masked, so keep user text out of it.
    """
    logger.warning("Guardrail triggered: %s %s", name, details)
    kwargs: dict[str, Any] = {"level": "WARNING", "metadata": {"guardrail": name, **details}}
    if input is not None:
        kwargs["input"] = input
    tracing.event(EVENT_NAME, **kwargs)


# ---- limits -----------------------------------------------------------------------------


class Deadline:
    """A whole-request time limit, checked between steps and passed to each model call."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._end = clock() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - self._clock())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


class RateLimiter:
    """At most `max_requests` per key in any `window_seconds`, kept in memory.

    Simple on purpose: one process, sliding window, no persistence. With several server
    processes each keeps its own count, so the effective limit is higher.
    """

    MAX_TRACKED_KEYS = 10_000  # bounds memory if many distinct clients show up

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """None if the request is allowed (and counted), else seconds until it would be."""
        now = self._clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            self._expire(hits, now)
            if len(hits) >= self._max:
                return hits[0] + self._window - now
            hits.append(now)
            if len(self._hits) > self.MAX_TRACKED_KEYS:
                self._forget_idle(now)
            return None

    def _expire(self, hits: deque[float], now: float) -> None:
        while hits and hits[0] <= now - self._window:
            hits.popleft()

    def _forget_idle(self, now: float) -> None:
        for key in list(self._hits):
            self._expire(self._hits[key], now)
            if not self._hits[key]:
                del self._hits[key]


def client_fingerprint(client_id: str) -> str:
    """A short stable hash, so logs and events can tell clients apart without storing IPs."""
    return hashlib.sha256(client_id.encode()).hexdigest()[:8]


# ---- input: classify the question ---------------------------------------------------------


class LabelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Label
    reason: str


# Written out by hand (a flat strict schema); tests keep it in sync with LabelInput.
CLASSIFY_TOOL: dict[str, Any] = {
    "name": "label_question",
    "description": "Record the label for the question.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["in_scope", "out_of_scope", "suspicious"]},
            "reason": {"type": "string"},
        },
        "required": ["label", "reason"],
        "additionalProperties": False,
    },
}


@dataclass(frozen=True)
class Classification:
    label: Label
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0
    failed: bool = False  # no usable label came back; the question is let through


@lru_cache(maxsize=4)
def load_classifier_prompt(path: Path) -> str:
    return path.read_text()


def wrap_question(question: str) -> str:
    """The question inside tags, with its angle brackets escaped so it cannot close the tag."""
    return f"<question>\n{html.escape(question, quote=False)}\n</question>"


def _label_from(response: Any) -> LabelInput | None:
    for block in response.content:
        if block.type == "tool_use" and block.name == CLASSIFY_TOOL["name"]:
            try:
                return LabelInput.model_validate(block.input)
            except ValidationError:
                return None
    return None


def _failed(reason: str, input_tokens: int = 0, output_tokens: int = 0) -> Classification:
    # Fail open: a classifier outage must not take the whole service down. The answering
    # model's own prompt and claim verification are still in place.
    return Classification("in_scope", reason, input_tokens, output_tokens, failed=True)


def classify_question(
    question: str,
    client: anthropic.Anthropic,
    settings: Settings,
    config: GuardrailsConfig,
    deadline: Deadline,
) -> Classification:
    """Label a question with a small model. Never raises on API trouble: see _failed."""
    prompt = load_classifier_prompt(config.classifier.prompt_path)
    messages = [{"role": "user", "content": wrap_question(question)}]
    model = settings.guard_model
    with tracing.observation(
        "classify-question",
        "generation",
        model=model,
        model_parameters={"max_tokens": config.classifier.max_tokens, "effort": "low"},
        input=messages_for_trace(prompt, messages),
    ) as span:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=config.classifier.max_tokens,
                system=prompt,
                tools=[CLASSIFY_TOOL],
                messages=messages,
                output_config={"effort": "low"},
                timeout=deadline.remaining(),
            )
        except anthropic.APIError as exc:
            span.update(level="ERROR", status_message=type(exc).__name__)
            return _failed(f"classifier call failed: {type(exc).__name__}")
        usage = usage_for_trace(response)
        label = _label_from(response)
        span.update(
            output=label.model_dump() if label else {"label": None},
            usage_details=usage,
            metadata={"stop_reason": response.stop_reason},
        )
    if label is None:
        return _failed("classifier returned no valid label", usage["input"], usage["output"])
    return Classification(label.label, label.reason, usage["input"], usage["output"])


def refusal_message(reason: RefusalReason) -> str:
    return REFUSALS[reason]


# ---- output: retry, then drop and flag --------------------------------------------------------


def failed_claims(verified: list[VerifiedClaim]) -> list[VerifiedClaim]:
    return [claim for claim in verified if not claim.passed]


def _number(value: float) -> str:
    return f"{value:.12g}"  # 55339003, not 5.5339e+07


def _claim_detail(claim: VerifiedClaim) -> str:
    """What the data says, so the retry can be corrected, not guessed again."""
    if claim.reason == "value mismatch" and claim.actual_value is not None:
        if claim.kind == "derived":
            sources = ", ".join(claim.source_figure_ids)
            return (
                f"recomputing {claim.operation} from {sources} gives {_number(claim.actual_value)}"
            )
        return f"the stored value is {_number(claim.actual_value)}"
    return claim.reason or "could not be verified"


def _listed(claims: list[VerifiedClaim], line: Callable[[VerifiedClaim], str]) -> str:
    lines = [line(claim) for claim in claims[:MAX_LISTED]]
    if len(claims) > MAX_LISTED:
        lines.append(f"- and {len(claims) - MAX_LISTED} more")
    return "\n".join(lines)


def retry_feedback(failed: list[VerifiedClaim]) -> str:
    """The text sent back to the model as the rejection of its submitted answer."""
    listed = _listed(
        failed,
        lambda c: f'- "{c.statement}": you claimed {_number(c.value)}, but {_claim_detail(c)}.',
    )
    return (
        "Verification rejected this answer. These claims do not match the data:\n"
        f"{listed}\n"
        "Fix each one using the tool results (call get_figure if you need to check), keep "
        "every claim that was fine, and call submit_answer again. A claim's value must be in "
        "the figure's own units (for example people, not millions) and match the data within "
        "rounding."
    )


def partial_explanation(dropped: list[VerifiedClaim]) -> str:
    """What the user is told when claims were dropped after the retry."""
    listed = _listed(dropped, lambda c: f'- "{c.statement}" ({_number(c.value)})')
    return (
        "Some figures in this answer could not be verified against the World Bank data, even "
        "after a second attempt. They were removed from the verified claims and should not be "
        f"relied on:\n{listed}"
    )
