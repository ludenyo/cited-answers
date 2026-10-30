import logging
from collections.abc import Callable, Iterator

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from cited_answers.api import create_app, get_ask_fn
from cited_answers.models import AskResponse, VerifiedClaim
from cited_answers.settings import SettingsError

CANNED = AskResponse(
    answer="Kenya's GDP grew 5.72% in 2023.",
    insufficient_data=False,
    claims=[
        VerifiedClaim(
            statement="grew 5.72%",
            value=5.72,
            kind="figure",
            figure_id="KEN:NY.GDP.MKTP.KD.ZG:2023",
            source_figure_ids=[],
            operation=None,
            passed=True,
            actual_value=5.72,
            source_url="https://example.test/ken-2023",
            source_urls=["https://example.test/ken-2023"],
            reason=None,
        )
    ],
    citation_pass_rate=1.0,
    model="test-model",
    prompt_version="abcd1234",
    latency_ms=42,
    input_tokens=100,
    output_tokens=20,
)


@pytest.fixture
def client_with() -> Iterator[Callable[[Callable[[str], AskResponse]], TestClient]]:
    def build(ask_fn: Callable[[str], AskResponse]) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_ask_fn] = lambda: ask_fn
        return TestClient(app)

    yield build


def test_ask_happy_path_returns_answer_claims_and_source_url(client_with) -> None:  # type: ignore[no-untyped-def]
    seen: list[str] = []

    def fake_ask(question: str) -> AskResponse:
        seen.append(question)
        return CANNED

    response = client_with(fake_ask).post("/ask", json={"question": "Kenya GDP growth 2023?"})

    assert response.status_code == 200
    body = response.json()
    assert seen == ["Kenya GDP growth 2023?"]
    assert body["answer"] == CANNED.answer
    assert body["citation_pass_rate"] == 1.0
    assert body["claims"][0]["source_url"] == "https://example.test/ken-2023"
    assert body["claims"][0]["passed"] is True
    assert body["claims"][0]["kind"] == "figure"
    assert body["claims"][0]["source_urls"] == ["https://example.test/ken-2023"]
    assert body["model"] == "test-model"
    assert body["prompt_version"] == "abcd1234"
    assert (body["latency_ms"], body["input_tokens"], body["output_tokens"]) == (42, 100, 20)


@pytest.mark.parametrize("question", ["", "x" * 501])
def test_ask_rejects_bad_question_length(client_with, question: str) -> None:  # type: ignore[no-untyped-def]
    def never(_q: str) -> AskResponse:
        raise AssertionError("pipeline must not run for invalid input")

    response = client_with(never).post("/ask", json={"question": question})
    assert response.status_code == 422


def test_ask_maps_model_api_failure_to_502(client_with) -> None:  # type: ignore[no-untyped-def]
    def broken(_q: str) -> AskResponse:
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.APIConnectionError(request=request)

    response = client_with(broken).post("/ask", json={"question": "hi"})

    assert response.status_code == 502
    assert "sk-" not in response.text  # nothing sensitive in the error body


def test_app_refuses_to_start_without_a_model_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    # Using TestClient as a context manager runs startup, as uvicorn would.
    with pytest.raises(SettingsError, match="ANTHROPIC_MODEL is not set"), TestClient(create_app()):
        pass


def test_app_starts_when_the_model_name_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
    with TestClient(create_app()) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_startup_makes_our_info_logs_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "some-model")
    package_logger = logging.getLogger("cited_answers")
    package_logger.setLevel(logging.WARNING)  # the state uvicorn leaves it in
    with TestClient(create_app()):
        assert package_logger.isEnabledFor(logging.INFO)
        assert package_logger.handlers  # something actually prints the lines
