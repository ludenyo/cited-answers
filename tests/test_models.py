from typing import Any

import pytest
from pydantic import ValidationError

from cited_answers.models import Answer, Claim, FigureResult

FIGURE_CLAIM: dict[str, Any] = {
    "statement": "grew 5.72%",
    "value": 5.72,
    "kind": "figure",
    "figure_id": "KEN:NY.GDP.MKTP.KD.ZG:2023",
    "source_figure_ids": [],
    "operation": None,
}
DERIVED_CLAIM: dict[str, Any] = {
    "statement": "up 1.56 points",
    "value": 1.56,
    "kind": "derived",
    "figure_id": None,
    "source_figure_ids": ["KEN:FP.CPI.TOTL.ZG:2021", "KEN:FP.CPI.TOTL.ZG:2023"],
    "operation": "difference",
}


def test_valid_figure_and_derived_claims() -> None:
    assert Claim.model_validate(FIGURE_CLAIM).kind == "figure"
    derived = Claim.model_validate(DERIVED_CLAIM)
    assert derived.operation == "difference"
    assert len(derived.source_figure_ids) == 2


@pytest.mark.parametrize(
    "override",
    [
        {"figure_id": None},  # figure claim without an id
        {"figure_id": ""},  # blank id
        {"operation": "difference"},  # figure claim with an operation
        {"source_figure_ids": ["a", "b"]},  # figure claim with sources
        {"kind": "guess"},  # unknown kind
    ],
)
def test_invalid_figure_claims_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate({**FIGURE_CLAIM, **override})


@pytest.mark.parametrize(
    "override",
    [
        {"figure_id": "KEN:X:2023"},  # derived claim with a figure id
        {"source_figure_ids": ["only-one"]},
        {"source_figure_ids": ["a", "b", "c"]},
        {"source_figure_ids": []},
        {"operation": None},
        {"operation": "average"},  # not one of the three operations
    ],
)
def test_invalid_derived_claims_are_rejected(override: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate({**DERIVED_CLAIM, **override})


def test_claims_must_be_complete_and_have_no_extra_fields() -> None:
    incomplete = {k: v for k, v in FIGURE_CLAIM.items() if k != "operation"}
    with pytest.raises(ValidationError):
        Claim.model_validate(incomplete)  # every field is required, as strict mode needs
    with pytest.raises(ValidationError):
        Claim.model_validate({**FIGURE_CLAIM, "confidence": 0.9})


def test_answer_accepts_a_mix_of_claim_kinds() -> None:
    answer = Answer.model_validate(
        {"text": "t", "claims": [FIGURE_CLAIM, DERIVED_CLAIM], "insufficient_data": False}
    )
    assert [c.kind for c in answer.claims] == ["figure", "derived"]


def test_figure_result_latest_year_is_optional_and_kept_when_given() -> None:
    base: dict[str, Any] = {
        "figure_id": "KEN:SP.POP.TOTL:2023",
        "country_code": "KEN",
        "country_name": "Kenya",
        "indicator_code": "SP.POP.TOTL",
        "indicator_name": "Population, total",
        "year": 2023,
        "value": 55339003.0,
        "unit": "people",
        "source_url": "https://example.test",
    }
    assert FigureResult.model_validate(base).latest_year is None
    assert FigureResult.model_validate({**base, "latest_year": 2025}).latest_year == 2025
