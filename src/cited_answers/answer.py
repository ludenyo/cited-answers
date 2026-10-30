"""The answering pipeline: a tool loop that ends when the model submits an Answer."""

import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
import psycopg
from pydantic import ValidationError

from cited_answers import guardrails, retrieval, tracing
from cited_answers.guardrails import Deadline
from cited_answers.models import Answer, AskResponse, FigureResult, VerifiedClaim
from cited_answers.settings import AppConfig, Settings
from cited_answers.trace_format import messages_for_trace, response_for_trace, usage_for_trace
from cited_answers.verify import Lookup, pass_rate, verify_claims

SUBMIT_TOOL = "submit_answer"
MAX_K = 20  # the most figures one search_figures call may return

logger = logging.getLogger(__name__)

# (tool name, parsed JSON input) -> JSON text for the tool result. May raise.
ToolRunner = Callable[[str, dict[str, Any]], str]


def _clean_schema(node: Any, defs: dict[str, Any]) -> Any:
    """Inline $ref links and drop titles from a Pydantic JSON schema.

    Strict tool schemas want a flat, fully inlined shape; Pydantic emits $defs,
    $ref and a title on every field. Property names are kept even if one of
    them happens to be called "title".
    """
    if isinstance(node, list):
        return [_clean_schema(item, defs) for item in node]
    if not isinstance(node, dict):
        return node
    if "$ref" in node:
        return _clean_schema(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "title"):
            continue
        if key == "properties":
            cleaned[key] = {name: _clean_schema(spec, defs) for name, spec in value.items()}
        else:
            cleaned[key] = _clean_schema(value, defs)
    return cleaned


def build_answer_schema() -> dict[str, Any]:
    """The submit_answer input schema, generated from the Answer model."""
    raw = Answer.model_json_schema()
    return _clean_schema(raw, raw.get("$defs", {}))  # type: ignore[no-any-return]


ANSWER_SCHEMA: dict[str, Any] = build_answer_schema()

# The answer comes back through a tool call (submit_answer), so the model has to
# fill the schema. tool_choice stays "auto" because the model must search first.
# strict=True makes the API guarantee tool input matches the schema, but it is
# left off for search_figures: its k input is optional, and optional properties
# are not confirmed to work under strict mode. The code clamps k instead.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_figures",
        "description": (
            "Semantic search over World Bank indicator figures for Kenya, Uganda and "
            "Tanzania. Use for open or vague questions, or when you don't know the exact "
            "indicator code. Returns figures with figure_id and source_url."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": MAX_K},
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_figure",
        "description": (
            "Exact lookup of one figure. Use when the country, indicator and year are all "
            "known. Prefer this over search_figures for precise questions. Call "
            "list_indicators first if unsure of the indicator code."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "country_code": {"type": "string", "enum": ["KEN", "UGA", "TZA"]},
                "indicator_code": {"type": "string"},
                "year": {"type": "integer"},
            },
            "required": ["country_code", "indicator_code", "year"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_indicators",
        "description": "List all available indicator codes, names and units.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": SUBMIT_TOOL,
        "description": (
            "Submit the final answer. You must call this exactly once to finish. Every "
            "number in text must appear in claims with its figure_id."
        ),
        "strict": True,
        "input_schema": ANSWER_SCHEMA,
    },
]


@lru_cache(maxsize=4)
def load_system_prompt(path: Path) -> str:
    return path.read_text()


def prompt_version(prompt: str) -> str:
    """Short fingerprint of the prompt text, so a response says which prompt made it."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:8]


def _figure_json(figure: FigureResult) -> dict[str, Any]:
    return figure.model_dump(exclude_none=True)


def _clamp_k(k: Any) -> int | None:
    return None if k is None else max(1, min(int(k), MAX_K))


def run_tool(
    conn: psycopg.Connection,
    name: str,
    tool_input: dict[str, Any],
    embed: retrieval.EmbedFn | None = None,
) -> str:
    """Execute one data tool and return its result as JSON text."""
    if name == "search_figures":
        kwargs: dict[str, Any] = {"k": _clamp_k(tool_input.get("k"))}
        if embed:
            kwargs["embed"] = embed
        found = retrieval.search_figures(conn, tool_input["question"], **kwargs)
        return json.dumps({"figures": [_figure_json(f) for f in found]})
    if name == "get_figure":
        figure = retrieval.get_figure(
            conn, tool_input["country_code"], tool_input["indicator_code"], tool_input["year"]
        )
        if figure is None:
            miss: dict[str, Any] = {"figure": None, "message": "No figure for that combination."}
            # Tell the model how far this series goes, so it need not guess another year.
            latest = retrieval.get_latest_year(
                conn, tool_input["country_code"], tool_input["indicator_code"]
            )
            if latest is not None:
                miss["latest_year"] = latest
            return json.dumps(miss)
        return json.dumps({"figure": _figure_json(figure)})
    if name == "list_indicators":
        indicators = retrieval.list_indicators(conn)
        return json.dumps({"indicators": [i.model_dump() for i in indicators]})
    raise ValueError(f"Unknown tool: {name}")


@dataclass
class Usage:
    """Token counts summed over every model call made for one question."""

    input_tokens: int = 0
    output_tokens: int = 0

    def add(self, response: Any) -> None:
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens


@dataclass
class Conversation:
    """One conversation with the answering model: its messages and the tokens it has used."""

    messages: list[dict[str, Any]]
    usage: Usage
    submit_call_id: str | None = None  # the submit_answer call a retry has to reject
    # Results already computed for the other calls in the turn that ended with the answer,
    # so rejecting that answer does not run those lookups a second time.
    turn_results: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class LoopResult:
    answer: Answer
    conversation: Conversation
    stop: str | None = None  # "iterations", "timeout" or "output_tokens" when a limit ended it

    @property
    def usage(self) -> Usage:
        return self.conversation.usage

    @property
    def hit_iteration_limit(self) -> bool:
        return self.stop == "iterations"


def _insufficient(message: str) -> Answer:
    return Answer(text=message, claims=[], insufficient_data=True)


def _result(tool_use_id: str, content: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}
    if is_error:
        block["is_error"] = True  # only sent when true; the API treats absent as false
    return block


def _execute(block: Any, run: ToolRunner) -> dict[str, Any]:
    """Run one tool call. Failures go back to the model, they never raise."""
    try:
        return _result(block.id, run(block.name, block.input))
    except Exception as exc:  # noqa: BLE001 - any tool failure must reach the model
        logger.warning("Tool %s failed: %s", block.name, exc)
        return _result(block.id, f"Error: {exc}", is_error=True)


def _parse_submission(block: Any) -> tuple[Answer | None, dict[str, Any] | None]:
    """Validate a submit_answer call: (answer, None) or (None, error tool result)."""
    try:
        return Answer.model_validate(block.input), None
    except ValidationError as exc:
        return None, _result(block.id, f"Invalid answer, fix and resubmit: {exc}", is_error=True)


def _call_model(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
    settings: Settings,
    system_prompt: str,
    deadline: Deadline,
) -> Any:
    """One model call, recorded as its own generation with model, tokens and messages."""
    parameters = {"max_tokens": settings.max_tokens, "effort": "low"}
    with tracing.observation(
        "call-model",
        "generation",
        model=settings.model,
        model_parameters=parameters,
        # Converted now: `messages` keeps growing after this call.
        input=messages_for_trace(system_prompt, messages),
    ) as span:
        response = client.messages.create(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": "low"},  # keeps a lookup task cheap and fast
            timeout=deadline.remaining(),  # no call may outlast the request's deadline
        )
        span.update(
            output=response_for_trace(response),
            usage_details=usage_for_trace(response),
            metadata={"stop_reason": response.stop_reason},
        )
    return response


def _step(
    client: anthropic.Anthropic,
    conversation: Conversation,
    settings: Settings,
    system_prompt: str,
    run: ToolRunner,
    deadline: Deadline,
) -> Answer | None:
    """One model turn. Returns the Answer once submitted, else None to continue."""
    messages = conversation.messages
    response = _call_model(client, messages, settings, system_prompt, deadline)
    conversation.usage.add(response)
    if response.stop_reason == "refusal":
        return _insufficient("The model declined to answer this question.")
    # Append the full content (not just text) so thinking blocks round-trip.
    messages.append({"role": "assistant", "content": response.content})
    calls = [b for b in response.content if b.type == "tool_use"]
    if not calls:
        messages.append({"role": "user", "content": f"Finish by calling {SUBMIT_TOOL}."})
        return None
    results = []
    for block in calls:
        if block.name == SUBMIT_TOOL:
            answer, error = _parse_submission(block)
            if answer:
                conversation.submit_call_id = block.id  # a retry must answer this call
                conversation.turn_results = {r["tool_use_id"]: r for r in results}
                return answer
            results.append(error)
        else:
            results.append(_execute(block, run))
    messages.append({"role": "user", "content": results})
    return None


def _stop_reason(conversation: Conversation, config: AppConfig, deadline: Deadline) -> str | None:
    """Why the loop must stop before another model call, if it must."""
    if deadline.expired():
        return "timeout"
    if conversation.usage.output_tokens >= config.guardrails.max_total_output_tokens:
        return "output_tokens"
    return None


def _stopped(conversation: Conversation, reason: str) -> LoopResult:
    return LoopResult(_insufficient(guardrails.STOP_MESSAGES[reason]), conversation, stop=reason)


def run_loop(
    conversation: Conversation,
    client: anthropic.Anthropic,
    run: ToolRunner,
    settings: Settings,
    config: AppConfig,
    deadline: Deadline,
    max_iterations: int,
) -> LoopResult:
    """Call the model until it submits an answer, or a limit stops it."""
    system_prompt = load_system_prompt(config.answer.prompt_path)
    for _ in range(max_iterations):
        reason = _stop_reason(conversation, config, deadline)
        if reason:
            return _stopped(conversation, reason)
        try:
            answer = _step(client, conversation, settings, system_prompt, run, deadline)
        except anthropic.APITimeoutError:
            return _stopped(conversation, "timeout")  # the call hit the request's deadline
        if answer:
            return LoopResult(answer, conversation)
    return _stopped(conversation, "iterations")


def answer_question(
    question: str,
    client: anthropic.Anthropic,
    run: ToolRunner,
    settings: Settings,
    config: AppConfig,
    deadline: Deadline | None = None,
) -> LoopResult:
    """Run the tool loop until the model calls submit_answer, or give up."""
    deadline = deadline or Deadline(config.guardrails.request_timeout_seconds)
    conversation = Conversation([{"role": "user", "content": question}], Usage())
    return run_loop(
        conversation, client, run, settings, config, deadline, config.llm.max_tool_iterations
    )


def _rejection_results(
    conversation: Conversation, run: ToolRunner, failed: list[VerifiedClaim]
) -> list[dict[str, Any]]:
    """A tool result for every call in the model's last turn: the submit call is rejected
    with the failed claims, and any other call that came with it is simply run."""
    results = []
    for block in conversation.messages[-1]["content"]:
        if block.type != "tool_use":
            continue
        if block.id == conversation.submit_call_id:
            feedback = guardrails.retry_feedback(failed)
            results.append(_result(block.id, feedback, is_error=True))
        elif block.id in conversation.turn_results:
            results.append(conversation.turn_results[block.id])  # already run this turn
        else:
            results.append(_execute(block, run))
    return results


def retry_answer(
    first: LoopResult,
    failed: list[VerifiedClaim],
    client: anthropic.Anthropic,
    run: ToolRunner,
    settings: Settings,
    config: AppConfig,
    deadline: Deadline,
) -> LoopResult:
    """Continue the same conversation: reject the submission, list what failed, let the model
    fix it. It gets its own small allowance of model calls, not the original five."""
    conversation = first.conversation
    results = _rejection_results(conversation, run, failed)
    conversation.messages.append({"role": "user", "content": results})
    return run_loop(
        conversation, client, run, settings, config, deadline,
        config.guardrails.retry_max_iterations,
    )  # fmt: skip


def _verify_traced(
    claims: list[Any], lookup: Lookup, config: AppConfig
) -> tuple[list[VerifiedClaim], float]:
    """Verification as its own span: input is the claims, output is what each check found."""
    with tracing.observation(
        "verify-claims", "evaluator", input=[c.model_dump() for c in claims]
    ) as span:
        verified = verify_claims(claims, lookup, config.verification)
        rate = pass_rate(verified)
        span.update(
            output=[v.model_dump() for v in verified],
            metadata={"citation_pass_rate": rate},
        )
    return verified, rate


def _note_stop(result: LoopResult) -> list[str]:
    """Record the guardrail for a loop that a limit ended early, if any."""
    if result.stop is None:
        return []
    name = guardrails.STOP_TRIGGERS[result.stop]
    guardrails.record_trigger(name, output_tokens=result.usage.output_tokens)
    return [name]


@dataclass
class Checked:
    """An answer after verification, retry and dropping, ready to become a response."""

    result: LoopResult
    claims: list[VerifiedClaim]  # only claims that verified
    pass_rate: float  # of the final attempt, counting claims that were dropped
    first_pass_rate: float  # of the first attempt, before any retry
    triggers: list[str]
    partial_explanation: str | None = None


def _answer_checked(
    question: str,
    client: anthropic.Anthropic,
    run: ToolRunner,
    lookup: Lookup,
    settings: Settings,
    config: AppConfig,
    deadline: Deadline,
) -> Checked:
    """Answer, verify, retry once on failed claims, then drop whatever still fails."""
    result = answer_question(question, client, run, settings, config, deadline)
    triggers = _note_stop(result)
    verified, rate = _verify_traced(result.answer.claims, lookup, config)
    first_rate, failed = rate, guardrails.failed_claims(verified)
    if failed:
        guardrails.record_trigger(guardrails.OUTPUT_RETRY, failed_claims=len(failed))
        triggers.append(guardrails.OUTPUT_RETRY)
        retried = retry_answer(result, failed, client, run, settings, config, deadline)
        if retried.stop is None:
            result = retried
            verified, rate = _verify_traced(result.answer.claims, lookup, config)
            failed = guardrails.failed_claims(verified)
        else:
            triggers += _note_stop(retried)  # keep the first answer, minus its failed claims
    explanation = None
    if failed:
        guardrails.record_trigger(guardrails.OUTPUT_PARTIAL, dropped_claims=len(failed))
        triggers.append(guardrails.OUTPUT_PARTIAL)
        explanation = guardrails.partial_explanation(failed)
        verified = [v for v in verified if v.passed]
    return Checked(result, verified, rate, first_rate, triggers, explanation)


def _screen(
    question: str,
    client: anthropic.Anthropic,
    settings: Settings,
    config: AppConfig,
    deadline: Deadline,
) -> guardrails.Classification:
    """Classify the question and record any guardrail that fires. Does not decide anything."""
    classification = guardrails.classify_question(
        question, client, settings, config.guardrails, deadline
    )
    if classification.failed:
        guardrails.record_trigger(guardrails.CLASSIFIER_ERROR, reason=classification.reason)
    elif classification.label == "out_of_scope":
        guardrails.record_trigger(
            guardrails.INPUT_OUT_OF_SCOPE, input=question, reason=classification.reason
        )
    elif classification.label == "suspicious":
        guardrails.record_trigger(
            guardrails.INPUT_SUSPICIOUS, input=question, reason=classification.reason
        )
    return classification


def _refusal(
    classification: guardrails.Classification, version: str, settings: Settings, latency_ms: int
) -> AskResponse:
    reason = classification.label
    assert reason != "in_scope"  # only called for a refusal
    return AskResponse(
        answer=guardrails.refusal_message(reason),
        insufficient_data=True,
        claims=[],
        citation_pass_rate=1.0,
        model=settings.model,
        prompt_version=version,
        latency_ms=latency_ms,
        input_tokens=classification.input_tokens,
        output_tokens=classification.output_tokens,
        refused=True,
        refusal_reason=reason,
        guardrails=[
            guardrails.INPUT_OUT_OF_SCOPE
            if reason == "out_of_scope"
            else guardrails.INPUT_SUSPICIOUS
        ],
    )


def _record_outcome(root: Any, response: AskResponse, extra: dict[str, Any]) -> None:
    """Final trace output, summary metadata, and (for answered questions) the two scores."""
    passed = sum(c.passed for c in response.claims)
    root.update(
        output=response.answer,  # what a reviewer wants to see at a glance
        metadata={
            "model": response.model,
            "prompt_version": response.prompt_version,
            "insufficient_data": response.insufficient_data,
            "claims_total": len(response.claims),
            "claims_passed": passed,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "latency_ms": response.latency_ms,
            "refused": response.refused,
            "partial_answer": response.partial_answer,
            "guardrails": response.guardrails,
            **extra,
        },
    )
    if response.refused:
        return  # nothing was answered, so a pass rate or an insufficient_data score means nothing
    tracing.score_current_trace("citation_pass_rate", response.citation_pass_rate, "NUMERIC")
    # Langfuse stores boolean scores as 1 or 0.
    tracing.score_current_trace("insufficient_data", float(response.insufficient_data), "BOOLEAN")


def _answered(
    checked: Checked,
    classification: guardrails.Classification,
    version: str,
    settings: Settings,
    latency_ms: int,
) -> AskResponse:
    usage = checked.result.usage
    triggers = ([guardrails.CLASSIFIER_ERROR] if classification.failed else []) + checked.triggers
    return AskResponse(
        answer=checked.result.answer.text,
        insufficient_data=checked.result.answer.insufficient_data,
        claims=checked.claims,
        citation_pass_rate=checked.pass_rate,
        model=settings.model,
        prompt_version=version,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens + classification.input_tokens,
        output_tokens=usage.output_tokens + classification.output_tokens,
        partial_answer=checked.partial_explanation is not None,
        partial_answer_explanation=checked.partial_explanation,
        guardrails=triggers,
    )


def ask(
    question: str,
    client: anthropic.Anthropic,
    run: ToolRunner,
    lookup: Lookup,
    settings: Settings,
    config: AppConfig,
) -> AskResponse:
    """Screen a question, answer it, verify the claims, and apply the output guardrails.

    The whole call is one trace. It is tagged with the prompt version, known up front,
    because tags cannot be added after the first span exists. One deadline covers the
    classifier, the answering loop and the retry together.
    """
    started = time.perf_counter()
    deadline = Deadline(config.guardrails.request_timeout_seconds)
    version = prompt_version(load_system_prompt(config.answer.prompt_path))
    with (
        tracing.observation("answer-question", "agent", input=question) as root,
        tracing.trace_attributes(
            trace_name="answer-question",
            tags=[f"prompt:{version}"],
            metadata={"prompt_version": version, "model": settings.model},
        ),
    ):
        classification = _screen(question, client, settings, config, deadline)
        elapsed = lambda: round((time.perf_counter() - started) * 1000)  # noqa: E731
        if classification.label != "in_scope":
            response = _refusal(classification, version, settings, elapsed())
            _record_outcome(root, response, {"classification": classification.label})
        else:
            checked = _answer_checked(question, client, run, lookup, settings, config, deadline)
            response = _answered(checked, classification, version, settings, elapsed())
            extra = {
                "classification": classification.label,
                "first_attempt_pass_rate": checked.first_pass_rate,
                "hit_iteration_limit": checked.result.hit_iteration_limit,
            }
            _record_outcome(root, response, extra)
    logger.info(
        "ask: model=%s prompt=%s latency_ms=%d input_tokens=%d output_tokens=%d claims=%d "
        "pass_rate=%.2f guardrails=%s question=%r",
        settings.model,
        version,
        response.latency_ms,
        response.input_tokens,
        response.output_tokens,
        len(response.claims),
        response.citation_pass_rate,
        response.guardrails,
        question,
    )
    return response
