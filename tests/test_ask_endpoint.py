from collections.abc import Callable, Iterator

import anthropic
import httpx2
import pytest
from fastapi.testclient import TestClient

from cited_answers.api import create_app, get_ask_fn
from cited_answers.models import AskResponse, VerifiedClaim

CANNED = AskResponse(
    answer="Kenya's GDP grew 5.72% in 2023.",
    insufficient_data=False,
    claims=[
        VerifiedClaim(
            statement="grew 5.72%",
            value=5.72,
            figure_id="KEN:NY.GDP.MKTP.KD.ZG:2023",
            passed=True,
            actual_value=5.72,
            source_url="https://example.test/ken-2023",
            reason=None,
        )
    ],
    citation_pass_rate=1.0,
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
