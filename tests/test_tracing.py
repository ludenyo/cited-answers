import json
import logging
import time
from types import SimpleNamespace
from typing import Any

import pytest

from cited_answers import tracing
from cited_answers.answer import ask, prompt_version, run_tool
from cited_answers.embeddings import MODEL_NAME
from tests.conftest import Captured, axis_vector
from tests.test_answer import (
    CONFIG,
    FIGURE,
    GOOD_ANSWER,
    PROMPT,
    SETTINGS,
    FakeClient,
    _response,
    _runner,
    _tool_use,
)

# ---- on / off ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        ({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk"}, True),
        (
            {
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
                "LANGFUSE_TRACING_ENABLED": "false",
            },
            False,
        ),
        (
            {
                "LANGFUSE_PUBLIC_KEY": "pk",
                "LANGFUSE_SECRET_KEY": "sk",
                "LANGFUSE_TRACING_ENABLED": "FALSE",
            },
            False,
        ),
        ({"LANGFUSE_PUBLIC_KEY": "pk"}, False),
        ({"LANGFUSE_SECRET_KEY": "sk"}, False),
        ({"LANGFUSE_PUBLIC_KEY": "  ", "LANGFUSE_SECRET_KEY": "sk"}, False),
        ({}, False),
    ],
)
def test_tracing_enabled(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], expected: bool
) -> None:
    for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_TRACING_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert tracing.tracing_enabled() is expected


def test_when_off_everything_is_a_harmless_no_op() -> None:
    assert tracing._client is None
    ran: list[str] = []
    with tracing.observation("x", "span", input="i") as span, tracing.trace_attributes(tags=["t"]):
        span.update(output="o", level="ERROR")
        ran.append("body")
    tracing.score_current_trace("s", 1.0, "NUMERIC")
    tracing.shutdown_tracing()
    assert ran == ["body"]


def test_init_does_nothing_without_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert tracing.init_tracing() is False
    assert tracing._client is None


def test_init_failure_is_swallowed_and_leaves_tracing_off(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")

    def broken(**_kwargs: Any) -> None:
        raise RuntimeError("sdk exploded")

    monkeypatch.setattr("langfuse.Langfuse", broken)
    with caplog.at_level(logging.WARNING, logger="cited_answers.tracing"):
        assert tracing.init_tracing() is False
    assert tracing._client is None
    assert "could not start the Langfuse client" in caplog.text


def test_init_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(tracing, "_client", sentinel)
    assert tracing.init_tracing() is True
    assert tracing._client is sentinel


def test_shutdown_flushes_once_and_swallows_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Client:
        def shutdown(self) -> None:
            calls.append("shutdown")
            raise RuntimeError("network down")

    monkeypatch.setattr(tracing, "_client", Client())
    tracing.shutdown_tracing()  # must not raise
    tracing.shutdown_tracing()
    assert calls == ["shutdown"]
    assert tracing._client is None


# ---- tracing must never break a request ----------------------------------------------


class _Manager:
    """A fake SDK context manager that can fail at each stage."""

    def __init__(
        self, fail_enter: bool = False, fail_update: bool = False, fail_exit: bool = False
    ):
        self.fail_enter, self.fail_update, self.fail_exit = fail_enter, fail_update, fail_exit
        self.exited_with: list[Any] = []

    def __enter__(self) -> "_Manager":
        if self.fail_enter:
            raise RuntimeError("enter failed")
        return self

    def update(self, **_kwargs: Any) -> None:
        if self.fail_update:
            raise RuntimeError("update failed")

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.exited_with.append(exc_type)
        if self.fail_exit:
            raise RuntimeError("exit failed")


class _FakeClient:
    def __init__(self, manager: _Manager | None = None, start_raises: bool = False) -> None:
        self.manager = manager or _Manager()
        self.start_raises = start_raises

    def start_as_current_observation(self, **_kwargs: Any) -> _Manager:
        if self.start_raises:
            raise RuntimeError("langfuse is down")
        return self.manager

    def score_current_trace(self, **_kwargs: Any) -> None:
        raise RuntimeError("score upload failed")


@pytest.mark.parametrize(
    "client",
    [
        _FakeClient(start_raises=True),
        _FakeClient(_Manager(fail_enter=True)),
        _FakeClient(_Manager(fail_update=True)),
        _FakeClient(_Manager(fail_exit=True)),
        _FakeClient(_Manager(fail_update=True, fail_exit=True)),
    ],
)
def test_sdk_failures_never_reach_the_caller(
    monkeypatch: pytest.MonkeyPatch, client: _FakeClient
) -> None:
    monkeypatch.setattr(tracing, "_client", client)

    with tracing.observation("x", "span") as span:
        span.update(output="o")
        result = 40 + 2
    tracing.score_current_trace("s", 1.0, "NUMERIC")

    assert result == 42


def test_the_bodys_own_exception_passes_through_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    manager = _Manager(fail_exit=True)  # even a failing exit must not replace the real error
    monkeypatch.setattr(tracing, "_client", _FakeClient(manager))
    original = ValueError("the real error")

    with pytest.raises(ValueError) as caught, tracing.observation("x", "span"):
        raise original

    assert caught.value is original
    assert manager.exited_with == [ValueError]  # the SDK was told about it


def test_trace_attributes_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_client", _FakeClient())

    def broken(**_kwargs: Any) -> None:
        raise RuntimeError("propagate failed")

    monkeypatch.setattr("langfuse.propagate_attributes", broken)
    with tracing.trace_attributes(tags=["t"]):
        ran = True
    assert ran


# ---- masking -------------------------------------------------------------------------


def test_mask_redacts_keys_and_emails_anywhere_in_the_data() -> None:
    data = {
        "question": "Mail me at jane.doe@example.com about Kenya",
        "nested": [{"note": "key sk-ant-api03-AbCdEfGhIjKl and sk-lf-1234abcd5678"}, 3, None],
        "public": "pk-lf-abcdef123456",
    }
    masked = tracing.mask_sensitive(data=data)
    text = json.dumps(masked)
    assert "jane.doe@example.com" not in text and "[email]" in text
    assert "sk-ant" not in text and "sk-lf" not in text and "pk-lf" not in text
    assert masked["nested"][1:] == [3, None]
    assert "Kenya" in masked["question"]  # ordinary text is untouched


def test_mask_leaves_ordinary_data_alone() -> None:
    data = {"figure_id": "KEN:SP.POP.TOTL:2023", "value": 55339003.0, "ok": True}
    assert tracing.mask_sensitive(data=data) == data


# ---- the real SDK, with an in-memory exporter ------------------------------------------


def _scripted_run() -> Any:
    figure = json.dumps({"figure": FIGURE.model_dump()})
    return _runner(figure)


def _ask_with_one_tool_call() -> Any:
    get_args = {"country_code": "KEN", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2023}
    client = FakeClient(
        _response(_tool_use("t1", "get_figure", get_args), tokens=(100, 20)),
        _response(_tool_use("t2", "submit_answer", GOOD_ANSWER), tokens=(250, 40)),
    )
    return ask(
        "Kenya GDP growth 2023?", client, _scripted_run(), lambda _id: FIGURE, SETTINGS, CONFIG
    )  # type: ignore[arg-type]


def test_one_ask_is_one_trace_with_the_expected_spans(captured: Captured) -> None:
    _ask_with_one_tool_call()

    spans = captured.spans()
    assert sorted(s.name for s in spans) == [
        "answer-question",
        "call-model",
        "call-model",
        "classify-question",
        "verify-claims",
    ]
    assert len({s.context.trace_id for s in spans}) == 1  # a single trace
    [root] = captured.by_name("answer-question")
    assert root.parent is None
    assert all(captured.parent_name(s) == "answer-question" for s in spans if s is not root)


def test_observation_types(captured: Captured) -> None:
    _ask_with_one_tool_call()
    types = {s.name: s.attributes["langfuse.observation.type"] for s in captured.spans()}
    assert types == {
        "answer-question": "agent",
        "call-model": "generation",
        "classify-question": "generation",
        "verify-claims": "evaluator",
    }


def test_each_llm_call_records_model_tokens_and_messages(captured: Captured) -> None:
    _ask_with_one_tool_call()

    first, second = sorted(captured.by_name("call-model"), key=lambda s: s.start_time)
    for span in (first, second):
        assert span.attributes["langfuse.observation.model.name"] == "test-model"
    assert json.loads(first.attributes["langfuse.observation.usage_details"]) == {
        "input": 100,
        "output": 20,
    }
    assert json.loads(second.attributes["langfuse.observation.usage_details"]) == {
        "input": 250,
        "output": 40,
    }
    first_input = json.loads(first.attributes["langfuse.observation.input"])
    assert [m["role"] for m in first_input] == ["system", "user"]
    second_input = json.loads(second.attributes["langfuse.observation.input"])
    assert [m["role"] for m in second_input] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]  # what the model saw
    first_output = json.loads(first.attributes["langfuse.observation.output"])
    assert first_output["tool_calls"][0]["function"]["name"] == "get_figure"


def test_trace_input_output_name_and_prompt_version_tag(captured: Captured) -> None:
    _ask_with_one_tool_call()

    [root] = captured.by_name("answer-question")
    version = prompt_version(PROMPT.read_text())
    assert root.attributes["langfuse.observation.input"] == "Kenya GDP growth 2023?"
    assert root.attributes["langfuse.observation.output"] == GOOD_ANSWER["text"]
    assert root.attributes["langfuse.trace.name"] == "answer-question"
    assert tuple(root.attributes["langfuse.trace.tags"]) == (f"prompt:{version}",)
    # the tag reaches every span, so any observation can be filtered by prompt version
    assert all(
        tuple(s.attributes["langfuse.trace.tags"]) == (f"prompt:{version}",)
        for s in captured.spans()
    )


def test_root_metadata_summarises_the_request(captured: Captured) -> None:
    _ask_with_one_tool_call()
    [root] = captured.by_name("answer-question")
    metadata = {
        key.removeprefix("langfuse.observation.metadata."): json.loads(value)
        for key, value in root.attributes.items()
        if key.startswith("langfuse.observation.metadata.")
    }
    assert metadata["model"] == "test-model"
    assert metadata["input_tokens"] == 350 and metadata["output_tokens"] == 60
    assert metadata["claims_total"] == 1 and metadata["claims_passed"] == 1
    assert metadata["insufficient_data"] is False
    assert metadata["hit_iteration_limit"] is False
    assert metadata["prompt_version"] == prompt_version(PROMPT.read_text())


def test_both_scores_are_recorded_on_the_trace(captured: Captured) -> None:
    _ask_with_one_tool_call()
    assert captured.scores == [
        {"name": "citation_pass_rate", "value": 1.0, "data_type": "NUMERIC"},
        {"name": "insufficient_data", "value": 0.0, "data_type": "BOOLEAN"},
    ]


def test_scores_reflect_failed_checks_and_insufficient_data(captured: Captured) -> None:
    wrong = {
        **GOOD_ANSWER,
        "claims": [{**GOOD_ANSWER["claims"][0], "value": 99.0}],
        "insufficient_data": True,
    }
    # The same wrong claim is submitted twice: once, and again after the retry.
    client = FakeClient(
        _response(_tool_use("t1", "submit_answer", wrong)),
        _response(_tool_use("t2", "submit_answer", wrong)),
    )

    ask("q", client, _runner(), lambda _id: FIGURE, SETTINGS, CONFIG)  # type: ignore[arg-type]

    assert captured.scores == [
        {"name": "citation_pass_rate", "value": 0.0, "data_type": "NUMERIC"},
        {"name": "insufficient_data", "value": 1.0, "data_type": "BOOLEAN"},
    ]


def test_verify_span_shows_what_each_check_found(captured: Captured) -> None:
    _ask_with_one_tool_call()
    [verify] = captured.by_name("verify-claims")
    output = json.loads(verify.attributes["langfuse.observation.output"])
    assert output[0]["passed"] is True and output[0]["figure_id"] == FIGURE.figure_id


def test_an_exception_marks_the_span_as_an_error_and_still_propagates(captured: Captured) -> None:
    with pytest.raises(RuntimeError, match="kaboom"), tracing.observation("work", "span"):
        raise RuntimeError("kaboom")

    [span] = captured.by_name("work")
    assert span.attributes["langfuse.observation.level"] == "ERROR"
    assert "kaboom" in span.attributes["langfuse.observation.status_message"]


def test_a_failed_model_call_is_an_error_on_the_generation(captured: Captured) -> None:
    class Failing:
        messages = None

        def __init__(self) -> None:
            self.messages = self

        def create(self, **kwargs: Any) -> Any:
            if any(t["name"] == "label_question" for t in kwargs["tools"]):
                return FakeClient()._classify(kwargs)  # the classifier works; the answer call fails
            raise ConnectionError("anthropic unreachable")

    with pytest.raises(ConnectionError):
        ask("q", Failing(), _runner(), lambda _id: FIGURE, SETTINGS, CONFIG)  # type: ignore[arg-type]

    [generation] = captured.by_name("call-model")
    assert generation.attributes["langfuse.observation.level"] == "ERROR"
    [root] = captured.by_name("answer-question")
    assert root.attributes["langfuse.observation.level"] == "ERROR"
    assert captured.scores == []  # nothing to score when there was no answer


def test_the_iteration_limit_is_visible_on_the_trace(captured: Captured) -> None:
    search = _tool_use("t", "search_figures", {"question": "x"})
    client = FakeClient(*[_response(search)] * CONFIG.llm.max_tool_iterations)

    response = ask("q", client, _runner(), lambda _id: FIGURE, SETTINGS, CONFIG)  # type: ignore[arg-type]

    assert response.insufficient_data is True
    [root] = captured.by_name("answer-question")
    assert root.attributes["langfuse.observation.metadata.hit_iteration_limit"] == "true"
    assert len(captured.by_name("call-model")) == CONFIG.llm.max_tool_iterations


def test_tracing_does_not_change_the_response(
    captured: Captured, monkeypatch: pytest.MonkeyPatch
) -> None:
    traced = _ask_with_one_tool_call()
    monkeypatch.setattr(tracing, "_client", None)
    untraced = _ask_with_one_tool_call()
    drop = {"latency_ms"}
    assert traced.model_dump(exclude=drop) == untraced.model_dump(exclude=drop)


def test_api_key_never_reaches_a_span(captured: Captured) -> None:
    leaky = {
        **GOOD_ANSWER,
        "text": "Reach me at ceo@example.com, key sk-ant-api03-SECRETSECRET1234",
    }
    client = FakeClient(_response(_tool_use("t1", "submit_answer", leaky)))
    ask(
        "my key is sk-ant-api03-ABCDEFGH12345678",
        client,
        _runner(),
        lambda _id: FIGURE,
        SETTINGS,
        CONFIG,
    )  # type: ignore[arg-type]

    everything = json.dumps([dict(s.attributes) for s in captured.spans()])
    assert "sk-ant" not in everything
    assert "ceo@example.com" not in everything


# ---- retrieval spans (need the database) ---------------------------------------------


def test_search_records_a_retriever_with_a_nested_embedding(
    captured: Captured, seeded_db: Any
) -> None:
    with tracing.observation("answer-question", "agent"):
        run_tool(
            seeded_db,
            "search_figures",
            {"question": "inflation"},
            embed=lambda _t: [axis_vector(4)],
        )

    [search] = captured.by_name("search-figures")
    [embed] = captured.by_name("embed-question")
    assert search.attributes["langfuse.observation.type"] == "retriever"
    assert captured.parent_name(search) == "answer-question"
    assert embed.attributes["langfuse.observation.type"] == "embedding"
    assert captured.parent_name(embed) == "search-figures"
    assert embed.attributes["langfuse.observation.model.name"] == MODEL_NAME
    assert json.loads(embed.attributes["langfuse.observation.output"]) == {"dimensions": 384}
    assert embed.attributes["langfuse.observation.input"] == "inflation"
    output = json.loads(search.attributes["langfuse.observation.output"])
    assert output[0]["figure_id"] == "TZA:FP.CPI.TOTL.ZG:2023" and output[0]["distance"] < 1e-6
    assert output[0]["latest_year"] == 2023
    assert json.loads(search.attributes["langfuse.observation.input"])["question"] == "inflation"
    assert search.attributes["langfuse.observation.metadata.result_count"] == str(len(output))


def test_get_figure_and_list_indicators_record_retriever_spans(
    captured: Captured, seeded_db: Any
) -> None:
    args = {"country_code": "KEN", "indicator_code": "NY.GDP.MKTP.KD.ZG", "year": 2023}
    run_tool(seeded_db, "get_figure", args)
    run_tool(seeded_db, "get_figure", {**args, "year": 1900})
    run_tool(seeded_db, "list_indicators", {})

    hits = sorted(captured.by_name("get-figure"), key=lambda s: s.start_time)
    assert (
        json.loads(hits[0].attributes["langfuse.observation.output"])["figure_id"]
        == "KEN:NY.GDP.MKTP.KD.ZG:2023"
    )
    miss = json.loads(hits[1].attributes["langfuse.observation.output"])
    assert miss == {"found": False}  # a miss is stated outright, since the SDK drops None
    [indicators] = captured.by_name("list-indicators")
    assert "SP.POP.TOTL" in json.loads(indicators.attributes["langfuse.observation.output"])
    assert all(s.attributes["langfuse.observation.type"] == "retriever" for s in captured.spans())


# ---- Langfuse really unreachable -----------------------------------------------------


def test_an_unreachable_langfuse_does_not_slow_or_break_a_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "true")
    client = tracing.create_client(
        public_key="pk-lf-unreachable",
        secret_key="sk-lf-unreachable",
        base_url="http://127.0.0.1:9",  # connection refused
        timeout=1,
        flush_at=1,
    )
    monkeypatch.setattr(tracing, "_client", client)

    started = time.perf_counter()
    response = _ask_with_one_tool_call()  # real exporter, real (failing) network
    elapsed = time.perf_counter() - started

    assert response.answer == GOOD_ANSWER["text"] and response.citation_pass_rate == 1.0
    assert elapsed < 2.0  # export runs on a background thread
    monkeypatch.setattr(tracing, "_client", None)
    client.shutdown()


# ---- no double counting from the Anthropic SDK's own spans ---------------------------


def test_the_sdks_own_anthropic_span_is_not_exported(captured: Captured) -> None:
    provider = tracing._client._resources.tracer_provider  # type: ignore[union-attr]
    with tracing.observation("call-model", "generation", model="m"):
        native = provider.get_tracer("com.anthropic.sdk.python")
        with native.start_as_current_span("anthropic.messages.create"):
            pass

    assert [s.name for s in captured.spans()] == ["call-model"]  # one generation, not two


def test_export_filter_drops_only_the_anthropic_sdk_scope() -> None:
    def span(scope_name: str) -> Any:
        return SimpleNamespace(
            instrumentation_scope=SimpleNamespace(name=scope_name), attributes={}, name="x"
        )

    assert tracing.should_export_span(span("com.anthropic.sdk.python")) is False
    assert tracing.should_export_span(span("com.anthropic.anything")) is False


# ---- events --------------------------------------------------------------------------


def test_an_event_inside_a_span_is_recorded_under_it(captured: Captured) -> None:
    with tracing.observation("answer-question", "agent"):
        tracing.event(
            "guardrail-triggered", level="WARNING", metadata={"guardrail": "output_retry"}
        )

    [event] = captured.by_name("guardrail-triggered")
    assert event.attributes["langfuse.observation.type"] == "event"
    assert event.attributes["langfuse.observation.level"] == "WARNING"
    assert event.attributes["langfuse.observation.metadata.guardrail"] == '"output_retry"'
    assert captured.parent_name(event) == "answer-question"


def test_an_event_with_no_surrounding_span_becomes_its_own_trace(captured: Captured) -> None:
    tracing.event("guardrail-triggered", level="WARNING", metadata={"guardrail": "rate_limited"})
    [event] = captured.by_name("guardrail-triggered")
    assert event.parent is None


def test_events_are_a_no_op_when_tracing_is_off() -> None:
    assert tracing._client is None
    tracing.event("guardrail-triggered", level="WARNING")  # nothing to assert but no error


def test_an_event_failure_never_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    class Broken:
        def create_event(self, **_kw: Any) -> None:
            raise RuntimeError("langfuse is down")

    monkeypatch.setattr(tracing, "_client", Broken())
    with caplog.at_level(logging.WARNING, logger="cited_answers.tracing"):
        tracing.event("guardrail-triggered", level="WARNING")
    assert "could not record event" in caplog.text
