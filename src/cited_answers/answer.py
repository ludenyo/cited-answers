"""The answering pipeline: a tool loop that ends when the model submits an Answer."""

import json
import logging
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
import psycopg
from pydantic import ValidationError

from cited_answers import retrieval
from cited_answers.models import Answer, AskResponse, FigureResult
from cited_answers.settings import Settings, VerificationConfig
from cited_answers.verify import Lookup, pass_rate, verify_claims

PROMPT_FILE = Path(__file__).resolve().parents[2] / "prompts" / "answer_system.md"
MAX_ITERATIONS = 6
SUBMIT_TOOL = "submit_answer"

logger = logging.getLogger(__name__)

# (tool name, parsed JSON input) -> JSON text for the tool result. May raise.
ToolRunner = Callable[[str, dict[str, Any]], str]

_CLAIM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "statement": {"type": "string"},
        "value": {"type": "number"},
        "figure_id": {"type": "string"},
    },
    "required": ["statement", "value", "figure_id"],
    "additionalProperties": False,
}

# Written out by hand rather than generated from the Answer model: strict tool
# schemas want a flat, fully inlined shape, and Pydantic emits $defs/$ref.
# tests/test_answer.py checks this stays in sync with the Answer model.
ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "claims": {"type": "array", "items": _CLAIM_SCHEMA},
        "insufficient_data": {"type": "boolean"},
    },
    "required": ["text", "claims", "insufficient_data"],
    "additionalProperties": False,
}

# strict=True makes the API guarantee that tool input matches the schema. The
# answer is returned through a tool call (submit_answer) so the model must fill
# the schema; tool_choice stays "auto" because it has to search first.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "search_figures",
        "description": (
            "Semantic search over World Bank indicator figures for Kenya, Uganda and "
            "Tanzania. Use for open questions or when the indicator code is unknown. "
            "Returns figures with figure_id, indicator_code, year, value and source_url."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_figure",
        "description": (
            "Exact lookup of one figure when the country, indicator code and year are "
            "all known. Use indicator codes returned by search_figures."
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
        "name": SUBMIT_TOOL,
        "description": (
            "Submit the final answer. Call exactly once to finish. Every number in "
            "text must appear in claims with its figure_id."
        ),
        "strict": True,
        "input_schema": ANSWER_SCHEMA,
    },
]


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    return PROMPT_FILE.read_text()


def _figure_json(figure: FigureResult) -> dict[str, Any]:
    return figure.model_dump(exclude_none=True)


def run_tool(
    conn: psycopg.Connection,
    name: str,
    tool_input: dict[str, Any],
    embed: retrieval.EmbedFn | None = None,
) -> str:
    """Execute one data tool and return its result as JSON text."""
    if name == "search_figures":
        kwargs = {"embed": embed} if embed else {}
        found = retrieval.search_figures(conn, tool_input["question"], **kwargs)
        return json.dumps({"figures": [_figure_json(f) for f in found]})
    if name == "get_figure":
        figure = retrieval.get_figure(
            conn, tool_input["country_code"], tool_input["indicator_code"], tool_input["year"]
        )
        if figure is None:
            return json.dumps({"figure": None, "message": "No figure for that combination."})
        return json.dumps({"figure": _figure_json(figure)})
    raise ValueError(f"Unknown tool: {name}")


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


def _step(
    client: anthropic.Anthropic,
    messages: list[dict[str, Any]],
    settings: Settings,
    run: ToolRunner,
) -> Answer | None:
    """One model turn. Returns the Answer once submitted, else None to continue."""
    response = client.messages.create(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=load_system_prompt(),
        tools=TOOLS,
        messages=messages,
        output_config={"effort": "low"},  # keeps a lookup task cheap and fast
    )
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
                return answer
            results.append(error)
        else:
            results.append(_execute(block, run))
    messages.append({"role": "user", "content": results})
    return None


def answer_question(
    question: str, client: anthropic.Anthropic, run: ToolRunner, settings: Settings
) -> Answer:
    """Run the tool loop until the model calls submit_answer, or give up."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
    for _ in range(MAX_ITERATIONS):
        answer = _step(client, messages, settings, run)
        if answer:
            return answer
    return _insufficient("The question could not be answered within the allowed steps.")


def ask(
    question: str,
    client: anthropic.Anthropic,
    run: ToolRunner,
    lookup: Lookup,
    settings: Settings,
    verification: VerificationConfig,
) -> AskResponse:
    """Answer a question, then verify every claim against the database."""
    answer = answer_question(question, client, run, settings)
    verified = verify_claims(answer.claims, lookup, verification)
    rate = pass_rate(verified)
    logger.info("ask: claims=%d pass_rate=%.2f question=%r", len(verified), rate, question)
    return AskResponse(
        answer=answer.text,
        insufficient_data=answer.insufficient_data,
        claims=verified,
        citation_pass_rate=rate,
    )
