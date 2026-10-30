import hashlib
import json
import re
from types import SimpleNamespace
from typing import Any

import pytest

from cited_answers import answer as answer_module
from cited_answers.answer import (
    ANSWER_SCHEMA,
    MAX_K,
    TOOLS,
    answer_question,
    ask,
    build_answer_schema,
    load_system_prompt,
    prompt_version,
    run_tool,
)
from cited_answers.models import Answer, Claim, FigureResult, IndicatorInfo
from cited_answers.settings import Settings, get_app_config

SETTINGS = Settings(model="test-model", max_tokens=512)
CONFIG = get_app_config()
MAX_ITERATIONS = CONFIG.llm.max_tool_iterations
PROMPT = CONFIG.answer.prompt_path

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
    "claims": [
        {
            "statement": "grew 5.72%",
            "value": 5.72,
            "kind": "figure",
            "figure_id": FIGURE.figure_id,
            "source_figure_ids": [],
            "operation": None,
        }
    ],
    "insufficient_data": False,
}


def _tool_use(call_id: str, name: str, tool_input: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input)


def _response(
    *blocks: SimpleNamespace, stop: str = "tool_use", tokens: tuple[int, int] = (10, 5)
) -> SimpleNamespace:
    usage = SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1])
    return SimpleNamespace(stop_reason=stop, content=list(blocks), usage=usage)


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


def _answer(client: FakeClient, run: Any = None) -> Any:
    return answer_question("q", client, run or _runner(), SETTINGS, CONFIG)  # type: ignore[arg-type]


# ---- the loop ---------------------------------------------------------------


def test_model_calls_get_figure_then_submit_answer() -> None:
    get_args = {"country_code": "KEN", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2023}
    client = FakeClient(
        _response(_tool_use("t1", "get_figure", get_args)),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER)),
    )
    run = _runner(json.dumps({"figure": FIGURE.model_dump()}))

    result = _answer(client, run)

    assert result.answer == Answer.model_validate(GOOD_ANSWER)
    assert run.calls == [("get_figure", get_args)]
    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert "is_error" not in tool_result
    assert FIGURE.source_url in tool_result["content"]  # source_url reaches the model


def test_request_carries_model_prompt_and_tools() -> None:
    client = FakeClient(_response(_tool_use("t1", "submit_answer", GOOD_ANSWER)))

    _answer(client)

    request = client.requests[0]
    assert request["model"] == "test-model"
    assert request["max_tokens"] == 512
    assert request["system"] == PROMPT.read_text()
    assert [t["name"] for t in request["tools"]] == [
        "search_figures",
        "get_figure",
        "list_indicators",
        "submit_answer",
    ]
    assert "temperature" not in request  # rejected by current Haiku models


def test_no_submit_answer_gives_insufficient_data_after_max_iterations() -> None:
    chatter = _response(SimpleNamespace(type="text", text="thinking..."), stop="end_turn")
    client = FakeClient(*[chatter] * MAX_ITERATIONS)

    result = _answer(client)

    assert result.answer.insufficient_data is True
    assert result.answer.claims == []
    assert "allowed steps" in result.answer.text
    assert len(client.requests) == MAX_ITERATIONS == 5
    assert "Finish by calling submit_answer" in client.requests[1]["messages"][-1]["content"]


def test_endless_tool_calls_also_stop_at_the_limit() -> None:
    search = _tool_use("t", "search_figures", {"question": "x"})
    client = FakeClient(*[_response(search)] * MAX_ITERATIONS)

    result = _answer(client)

    assert result.answer.insufficient_data is True
    assert len(client.requests) == MAX_ITERATIONS


def test_tool_error_is_returned_to_model_not_raised() -> None:
    def failing(name: str, tool_input: dict[str, Any]) -> str:
        raise RuntimeError("database is down")

    client = FakeClient(
        _response(_tool_use("t1", "search_figures", {"question": "x"})),
        _response(_tool_use("t2", "submit_answer", {**GOOD_ANSWER, "insufficient_data": True})),
    )

    result = _answer(client, failing)

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "database is down" in tool_result["content"]
    assert result.answer.insufficient_data is True  # the loop carried on and finished


def test_invalid_submission_is_an_error_result_and_model_can_retry() -> None:
    client = FakeClient(
        _response(_tool_use("t1", "submit_answer", {"text": "no claims field"})),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER)),
    )

    result = _answer(client)

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert result.answer.text == GOOD_ANSWER["text"]


def test_refusal_becomes_insufficient_data() -> None:
    result = _answer(FakeClient(_response(stop="refusal")))
    assert result.answer.insufficient_data is True


def test_tokens_are_summed_over_every_model_call() -> None:
    client = FakeClient(
        _response(_tool_use("t1", "search_figures", {"question": "x"}), tokens=(100, 20)),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER), tokens=(250, 40)),
    )
    result = _answer(client)
    assert (result.usage.input_tokens, result.usage.output_tokens) == (350, 60)


# ---- ask(): verification and response metadata --------------------------------


def test_ask_verifies_claims_and_fills_every_response_field() -> None:
    wrong = {**GOOD_ANSWER["claims"][0], "statement": "x", "value": 9.9}
    submitted = {**GOOD_ANSWER, "claims": [*GOOD_ANSWER["claims"], wrong]}
    client = FakeClient(_response(_tool_use("t1", "submit_answer", submitted), tokens=(120, 30)))

    response = ask("q", client, _runner(), lambda _id: FIGURE, SETTINGS, CONFIG)  # type: ignore[arg-type]

    assert [c.passed for c in response.claims] == [True, False]
    assert response.citation_pass_rate == 0.5
    assert response.claims[0].source_url == FIGURE.source_url
    assert response.model == "test-model"
    assert response.prompt_version == hashlib.sha256(PROMPT.read_bytes()).hexdigest()[:8]
    assert len(response.prompt_version) == 8
    assert isinstance(response.latency_ms, int) and response.latency_ms >= 0
    assert (response.input_tokens, response.output_tokens) == (120, 30)


def test_ask_logs_latency_and_tokens_but_no_secrets(caplog: pytest.LogCaptureFixture) -> None:
    client = FakeClient(_response(_tool_use("t1", "submit_answer", GOOD_ANSWER), tokens=(7, 3)))
    with caplog.at_level("INFO", logger="cited_answers.answer"):
        ask("q", client, _runner(), lambda _id: FIGURE, SETTINGS, CONFIG)  # type: ignore[arg-type]

    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("ask:"))
    assert "latency_ms=" in line and "input_tokens=7" in line and "output_tokens=3" in line
    assert "pass_rate=1.00" in line
    assert "sk-" not in line


def test_prompt_version_changes_with_the_text() -> None:
    assert prompt_version("a") != prompt_version("b")
    assert prompt_version("a") == prompt_version("a")


# ---- tool definitions ---------------------------------------------------------


def test_tool_descriptions_are_the_exact_spec_text() -> None:
    descriptions = {t["name"]: t["description"] for t in TOOLS}
    assert descriptions["search_figures"] == (
        "Semantic search over World Bank indicator figures for Kenya, Uganda and "
        "Tanzania. Use for open or vague questions, or when you don't know the exact "
        "indicator code. Returns figures with figure_id and source_url."
    )
    assert descriptions["get_figure"] == (
        "Exact lookup of one figure. Use when the country, indicator and year are all "
        "known. Prefer this over search_figures for precise questions. Call "
        "list_indicators first if unsure of the indicator code."
    )
    assert descriptions["list_indicators"] == "List all available indicator codes, names and units."
    assert descriptions["submit_answer"] == (
        "Submit the final answer. You must call this exactly once to finish. Every "
        "number in text must appear in claims with its figure_id."
    )


def test_search_figures_has_optional_k_capped_at_20() -> None:
    schema = next(t for t in TOOLS if t["name"] == "search_figures")["input_schema"]
    assert schema["required"] == ["question"]  # k is optional
    assert schema["properties"]["k"]["maximum"] == MAX_K == 20


def test_get_figure_and_list_indicators_schemas() -> None:
    by_name = {t["name"]: t for t in TOOLS}
    get_schema = by_name["get_figure"]["input_schema"]
    assert get_schema["properties"]["country_code"]["enum"] == ["KEN", "UGA", "TZA"]
    assert set(get_schema["required"]) == {"country_code", "indicator_code", "year"}
    assert by_name["list_indicators"]["input_schema"]["properties"] == {}


def test_strict_is_on_where_every_property_is_required() -> None:
    strict = {t["name"]: t.get("strict", False) for t in TOOLS}
    assert strict == {
        "search_figures": False,  # optional k, see the comment on TOOLS
        "get_figure": True,
        "list_indicators": True,
        "submit_answer": True,
    }


def test_answer_schema_is_generated_from_the_model_and_in_sync() -> None:
    assert ANSWER_SCHEMA == build_answer_schema()
    assert TOOLS[-1]["input_schema"] is ANSWER_SCHEMA
    assert set(ANSWER_SCHEMA["properties"]) == set(Answer.model_fields)
    assert set(ANSWER_SCHEMA["required"]) == set(Answer.model_fields)
    claim_schema = ANSWER_SCHEMA["properties"]["claims"]["items"]
    assert set(claim_schema["properties"]) == set(Claim.model_fields)
    assert set(claim_schema["required"]) == set(Claim.model_fields)
    assert ANSWER_SCHEMA["additionalProperties"] is False
    assert claim_schema["additionalProperties"] is False


def test_generated_schema_is_flat_with_no_refs_defs_or_titles() -> None:
    def keywords(node: Any) -> set[str]:
        if isinstance(node, list):
            return set().union(*(keywords(item) for item in node)) if node else set()
        if not isinstance(node, dict):
            return set()
        found = set(node) - {"properties"}  # property names are data, not schema keywords
        for key, value in node.items():
            if key == "properties":
                for spec in value.values():
                    found |= keywords(spec)
            else:
                found |= keywords(value)
        return found

    assert not keywords(ANSWER_SCHEMA) & {"$ref", "$defs", "title"}


# ---- prompt -------------------------------------------------------------------


def test_prompt_is_loaded_from_the_configured_file() -> None:
    assert load_system_prompt(PROMPT) == PROMPT.read_text()
    assert PROMPT.name == "answer_system.md"


def test_every_tool_the_prompt_names_exists_and_vice_versa() -> None:
    prompt = PROMPT.read_text()
    names = r"\b(search_figures|get_figure|list_indicators|submit_answer)\b"
    assert set(re.findall(names, prompt)) == {t["name"] for t in TOOLS}


def test_prompt_has_the_nine_rules() -> None:
    prompt = PROMPT.read_text()
    assert [int(n) for n in re.findall(r"^(\d)\. ", prompt, re.MULTILINE)] == list(range(1, 10))
    assert "Use ONLY numbers returned by your tools" in prompt


# ---- run_tool -----------------------------------------------------------------


def test_run_tool_get_figure_found_and_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    lookups = iter([FIGURE, None])
    monkeypatch.setattr(answer_module.retrieval, "get_figure", lambda *_a: next(lookups))
    args = {"country_code": "KEN", "indicator_code": "X", "year": 2023}

    found = json.loads(run_tool(None, "get_figure", args))  # type: ignore[arg-type]
    missing = json.loads(run_tool(None, "get_figure", args))  # type: ignore[arg-type]

    assert found["figure"]["figure_id"] == FIGURE.figure_id
    assert found["figure"]["source_url"] == FIGURE.source_url
    assert missing["figure"] is None


@pytest.mark.parametrize(
    ("requested", "passed_on"), [(None, None), (5, 5), (50, 20), (0, 1), (-3, 1)]
)
def test_run_tool_search_clamps_k(
    monkeypatch: pytest.MonkeyPatch, requested: int | None, passed_on: int | None
) -> None:
    seen: dict[str, Any] = {}

    def fake_search(_conn: Any, question: str, **kwargs: Any) -> list[FigureResult]:
        seen.update(kwargs)
        return [FIGURE]

    monkeypatch.setattr(answer_module.retrieval, "search_figures", fake_search)
    tool_input: dict[str, Any] = {"question": "gdp"}
    if requested is not None:
        tool_input["k"] = requested

    found = json.loads(run_tool(None, "search_figures", tool_input))  # type: ignore[arg-type]

    assert seen["k"] == passed_on
    assert found["figures"][0]["figure_id"] == FIGURE.figure_id


def test_run_tool_list_indicators(monkeypatch: pytest.MonkeyPatch) -> None:
    info = IndicatorInfo(
        indicator_code="SP.POP.TOTL", indicator_name="Population, total", unit="people"
    )
    monkeypatch.setattr(answer_module.retrieval, "list_indicators", lambda _c: [info])

    result = json.loads(run_tool(None, "list_indicators", {}))  # type: ignore[arg-type]

    assert result == {"indicators": [info.model_dump()]}


def test_run_tool_unknown_tool_raises() -> None:
    with pytest.raises(ValueError, match="Unknown tool"):
        run_tool(None, "drop_tables", {})  # type: ignore[arg-type]


def test_prompt_tells_the_model_how_to_write_derived_claims() -> None:
    prompt = PROMPT.read_text()
    for phrase in ("derived claim", "difference", "percent_change", "ratio", "from", "negative"):
        assert phrase in prompt


def test_derived_claim_flows_through_ask_and_is_verified() -> None:
    other = FIGURE.model_copy(update={"figure_id": "KEN:NY.GDP.MKTP.KD.ZG:2022", "value": 4.86})
    figures = {FIGURE.figure_id: FIGURE, other.figure_id: other}
    submitted = {
        "text": "Growth rose from 4.86% to 5.72%, up 0.86 points.",
        "claims": [
            {**GOOD_ANSWER["claims"][0], "value": 4.86, "figure_id": other.figure_id},
            GOOD_ANSWER["claims"][0],
            {
                "statement": "up 0.86 points",
                "value": 0.86,
                "kind": "derived",
                "figure_id": None,
                "source_figure_ids": [other.figure_id, FIGURE.figure_id],
                "operation": "difference",
            },
        ],
        "insufficient_data": False,
    }
    client = FakeClient(_response(_tool_use("t1", "submit_answer", submitted)))

    response = ask("q", client, _runner(), figures.get, SETTINGS, CONFIG)  # type: ignore[arg-type]

    assert [c.passed for c in response.claims] == [True, True, True]
    derived = response.claims[2]
    assert derived.kind == "derived" and derived.source_url is None
    assert derived.source_urls == [other.source_url, FIGURE.source_url]
    assert abs(derived.actual_value - 0.86) < 1e-9  # type: ignore[operator]
    assert response.citation_pass_rate == 1.0


def test_a_malformed_derived_claim_is_an_error_result_and_can_be_fixed() -> None:
    bad = {
        **GOOD_ANSWER,
        "claims": [
            {
                "statement": "up",
                "value": 1.0,
                "kind": "derived",
                "figure_id": None,
                "source_figure_ids": ["only-one"],
                "operation": "difference",
            }
        ],
    }
    client = FakeClient(
        _response(_tool_use("t1", "submit_answer", bad)),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER)),
    )

    result = _answer(client)

    tool_result = client.requests[1]["messages"][-1]["content"][0]
    assert tool_result["is_error"] is True
    assert "exactly two source_figure_ids" in tool_result["content"]
    assert result.answer.text == GOOD_ANSWER["text"]
