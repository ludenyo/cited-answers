import json
from decimal import Decimal
from types import SimpleNamespace
from typing import Any

import pytest

from cited_answers import answer as answer_module
from cited_answers.answer import (
    ANSWER_SCHEMA,
    MAX_ITERATIONS,
    TOOLS,
    answer_question,
    ask,
    load_system_prompt,
    run_tool,
)
from cited_answers.models import Answer, Claim, FigureResult
from cited_answers.settings import Settings, VerificationConfig

SETTINGS = Settings(model="test-model", max_tokens=512)
VERIFICATION = VerificationConfig(abs_tolerance=Decimal("0.05"), rel_tolerance=Decimal("0.005"))
FIGURE = FigureResult(
    figure_id="KEN:NY.GDP.MKTP.KD.ZG:2023",
    country_code="KEN",
    country_name="Kenya",
    indicator_code="NY.GDP.MKTP.KD.ZG",
    indicator_name="GDP growth (annual %)",
    year=2023,
    value=5.72,
    unit="annual %",
    source_url="https://example.test/ken-2023",
)
GOOD_ANSWER = {
    "text": "Kenya's GDP grew 5.72% in 2023.",
    "claims": [{"statement": "grew 5.72%", "value": 5.72, "figure_id": FIGURE.figure_id}],
    "insufficient_data": False,
}


def _tool_use(call_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def _response(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


class FakeClient:
    """Stands in for anthropic.Anthropic: replays scripted responses, never hits the network."""

    def __init__(self, *responses: SimpleNamespace) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        kwargs["messages"] = [dict(m) for m in kwargs["messages"]]  # snapshot before mutation
        self.requests.append(kwargs)
        return self._responses.pop(0)


def _runner(result: str = "{}") -> Any:
    calls: list[tuple[str, dict[str, Any]]] = []

    def run(name: str, tool_input: dict[str, Any]) -> str:
        calls.append((name, tool_input))
        return result

    run.calls = calls  # type: ignore[attr-defined]
    return run


def test_model_calls_get_figure_then_submit_answer() -> None:
    get_args = {"country_code": "KEN", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2023}
    client = FakeClient(
        _response(_tool_use("t1", "get_figure", get_args)),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER)),
    )
    run = _runner(json.dumps({"figure": FIGURE.model_dump()}))

    result = answer_question("Kenya GDP growth 2023?", client, run, SETTINGS)  # type: ignore[arg-type]

    assert result == Answer.model_validate(GOOD_ANSWER)
    assert run.calls == [("get_figure", get_args)]
    second_request_messages = client.requests[1]["messages"]
    tool_result = second_request_messages[-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert "is_error" not in tool_result
    assert FIGURE.source_url in tool_result["content"]  # source_url reaches the model


def test_request_carries_model_prompt_and_strict_tools() -> None:
    client = FakeClient(_response(_tool_use("t1", "submit_answer", GOOD_ANSWER)))

    answer_question("q", client, _runner(), SETTINGS)  # type: ignore[arg-type]

    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 512
    assert "Use ONLY numbers returned by your tools" in request["system"]
    assert [t["name"] for t in request["tools"]] == [
        "search_figures",
        "get_figure",
        "submit_answer",
    ]
    assert all(t["strict"] is True for t in request["tools"])
    assert "temperature" not in request  # rejected by current Haiku models


def test_no_submit_answer_gives_insufficient_data_after_max_iterations() -> None:
    chatter = _response(SimpleNamespace(type="text", text="thinking..."), stop="end_turn")
    client = FakeClient(*[chatter] * MAX_ITERATIONS)

    result = answer_question("q", client, _runner(), SETTINGS)  # type: ignore[arg-type]

    assert result.insufficient_data is True
    assert result.claims == []
    assert len(client.requests) == MAX_ITERATIONS
    assert "Finish by calling submit_answer" in client.requests[1]["messages"][-1]["content"]


def test_endless_tool_calls_also_stop_at_the_limit() -> None:
    search = _tool_use("t", "search_figures", {"question": "x"})
    client = FakeClient(*[_response(search)] * MAX_ITERATIONS)

    result = answer_question("q", client, _runner(), SETTINGS)  # type: ignore[arg-type]

    assert result.insufficient_data is True
    assert len(client.requests) == MAX_ITERATIONS


def test_tool_error_is_returned_to_model_not_raised() -> None:
    def failing(name: str, tool_input: dict[str, Any]) -> str:
        raise RuntimeError("database is down")

    client = FakeClient(
        _response(_tool_use("t1", "search_figures", {"question": "x"})),
        _response(_tool_use("t2", "submit_answer", {**GOOD_ANSWER, "insufficient_data": True})),
    )

    result = answer_question("q", client, failing, SETTINGS)  # type: ignore[arg-type]

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "database is down" in tool_result["content"]
    assert result.insufficient_data is True  # the loop carried on and finished


def test_invalid_submission_is_an_error_result_and_model_can_retry() -> None:
    client = FakeClient(
        _response(_tool_use("t1", "submit_answer", {"text": "no claims field"})),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER)),
    )

    result = answer_question("q", client, _runner(), SETTINGS)  # type: ignore[arg-type]

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert result.text == GOOD_ANSWER["text"]


def test_refusal_becomes_insufficient_data() -> None:
    client = FakeClient(_response(stop="refusal"))
    result = answer_question("q", client, _runner(), SETTINGS)  # type: ignore[arg-type]
    assert result.insufficient_data is True


def test_ask_verifies_claims_and_reports_pass_rate() -> None:
    wrong = {"statement": "x", "value": 9.9, "figure_id": FIGURE.figure_id}
    submitted = {**GOOD_ANSWER, "claims": [*GOOD_ANSWER["claims"], wrong]}
    client = FakeClient(_response(_tool_use("t1", "submit_answer", submitted)))

    response = ask("q", client, _runner(), lambda _id: FIGURE, SETTINGS, VERIFICATION)  # type: ignore[arg-type]

    assert [c.passed for c in response.claims] == [True, False]
    assert response.citation_pass_rate == 0.5
    assert response.claims[0].source_url == FIGURE.source_url


def test_answer_schema_matches_the_pydantic_model() -> None:
    assert set(ANSWER_SCHEMA["properties"]) == set(Answer.model_fields)
    assert set(ANSWER_SCHEMA["required"]) == set(Answer.model_fields)
    claim_schema = ANSWER_SCHEMA["properties"]["claims"]["items"]
    assert set(claim_schema["properties"]) == set(Claim.model_fields)
    assert set(claim_schema["required"]) == set(Claim.model_fields)
    assert TOOLS[-1]["input_schema"] is ANSWER_SCHEMA


def test_system_prompt_loads_from_its_own_file() -> None:
    assert load_system_prompt() == answer_module.PROMPT_FILE.read_text()
    assert answer_module.PROMPT_FILE.name == "answer_system.md"


def test_run_tool_get_figure_found_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = iter([FIGURE, None])
    monkeypatch.setattr(answer_module.retrieval, "get_figure", lambda *_a: next(lookups))
    args = {"country_code": "KEN", "indicator_code": "X", "year": 2023}

    found = json.loads(run_tool(None, "get_figure", args))  # type: ignore[arg-type]
    missing = json.loads(run_tool(None, "get_figure", args))  # type: ignore[arg-type]

    assert found["figure"]["figure_id"] == FIGURE.figure_id
    assert found["figure"]["source_url"] == FIGURE.source_url
    assert missing["figure"] is None


def test_run_tool_search_and_unknown_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(answer_module.retrieval, "search_figures", lambda *_a, **_k: [FIGURE])

    found = json.loads(run_tool(None, "search_figures", {"question": "gdp"}))  # type: ignore[arg-type]

    assert found["figures"][0]["figure_id"] == FIGURE.figure_id
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool(None, "drop_tables", {})  # type: ignore[arg-type]
