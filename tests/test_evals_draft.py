import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from cited_answers.models import Claim
from cited_answers.settings import get_app_config
from cited_answers.verify import compute_derived, values_match

EVAL_FILE = Path(__file__).parents[1] / "evals" / "questions_draft.yaml"
QUESTIONS: list[dict[str, Any]] = yaml.safe_load(EVAL_FILE.read_text())["questions"]
CATEGORIES = {"easy": 5, "compare": 5, "vague": 3, "latest": 2, "trap": 5}
FIGURE_ID = re.compile(r"^[A-Z]{3}:[A-Z0-9.]+:\d{4}$")


def test_ids_are_unique_and_categories_have_the_expected_counts() -> None:
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    counts = {c: sum(q["category"] == c for q in QUESTIONS) for c in CATEGORIES}
    assert counts == CATEGORIES
    assert all(q["id"].startswith(q["category"]) for q in QUESTIONS)


def test_every_question_has_text_and_an_insufficient_data_flag() -> None:
    for q in QUESTIONS:
        assert q["question"].strip(), q["id"]
        assert isinstance(q["expect"]["insufficient_data"], bool), q["id"]


def test_traps_expect_no_figures_unless_they_correct_a_premise() -> None:
    for q in QUESTIONS:
        if q["category"] == "trap" and q["expect"]["insufficient_data"]:
            assert q["expect"]["figure_ids"] == [], q["id"]
        if not q["expect"]["insufficient_data"]:
            assert q["expect"]["figure_ids"], q["id"]


def test_figure_ids_look_like_figure_ids_and_values_cover_them() -> None:
    for q in QUESTIONS:
        ids = q["expect"]["figure_ids"]
        assert all(FIGURE_ID.match(i) for i in ids), q["id"]
        assert set(q["expect"].get("values", {})) <= set(ids), q["id"]


def _derived() -> list[tuple[str, dict[str, Any]]]:
    return [(q["id"], d) for q in QUESTIONS for d in q["expect"].get("derived_claims", [])]


def test_the_three_comparison_questions_expect_derived_claims() -> None:
    by_id = {q["id"]: q["expect"].get("derived_claims", []) for q in QUESTIONS}
    assert [d["operation"] for d in by_id["compare-01"]] == ["difference"]
    assert [d["operation"] for d in by_id["compare-03"]] == ["percent_change"]
    assert [d["operation"] for d in by_id["compare-05"]] == ["difference"]


@pytest.mark.parametrize(("question_id", "derived"), _derived())
def test_derived_expectations_are_valid_claims_and_match_their_sources(
    question_id: str, derived: dict[str, Any]
) -> None:
    question = next(q for q in QUESTIONS if q["id"] == question_id)
    sources = derived["source_figure_ids"]
    assert set(sources) <= set(question["expect"]["figure_ids"])
    # The same validation a real submission goes through.
    Claim.model_validate(
        {
            "statement": "expected",
            "value": derived["value"],
            "kind": "derived",
            "figure_id": None,
            "source_figure_ids": sources,
            "operation": derived["operation"],
        }
    )
    # The stated value must agree with the stated source values, within the real tolerances.
    first, second = (Decimal(str(question["expect"]["values"][s])) for s in sources)
    computed = compute_derived(derived["operation"], first, second)
    assert computed is not None
    assert values_match(derived["value"], float(computed), get_app_config().verification)


def test_vague_questions_record_what_is_still_uncertain() -> None:
    for q in QUESTIONS:
        if q["category"] == "vague":
            assert q["todo"], q["id"]


def _latest_checks() -> list[tuple[str, dict[str, Any]]]:
    return [(q["id"], pair) for q in QUESTIONS for pair in q["expect"].get("latest_year_check", [])]


def test_the_two_latest_cases_carry_a_latest_year_check() -> None:
    latest = [q for q in QUESTIONS if q["category"] == "latest"]
    assert len(latest) == 2
    assert all(q["expect"].get("latest_year_check") for q in latest)


@pytest.mark.parametrize(("question_id", "pair"), _latest_checks())
def test_each_latest_check_names_a_pair_the_expected_figures_cover(
    question_id: str, pair: dict[str, Any]
) -> None:
    question = next(q for q in QUESTIONS if q["id"] == question_id)
    prefix = f"{pair['country']}:{pair['indicator']}:"
    expected = [i for i in question["expect"]["figure_ids"] if i.startswith(prefix)]
    assert expected, f"{question_id}: no expected figure for {prefix}"
    assert set(pair) == {"country", "indicator"}


def test_vague_03_keeps_its_label_and_states_the_head_count_rule() -> None:
    vague = next(q for q in QUESTIONS if q["id"] == "vague-03")
    assert vague["expect"]["insufficient_data"] is False
    assert vague["expect"]["figure_ids"] == ["UGA:EG.ELC.ACCS.ZS:2024"]
    assert vague["expect"]["values"] == {"UGA:EG.ELC.ACCS.ZS:2024": 55.3}
    assert "head count isn't available" in vague["notes"]
    assert "older year as the latest" in vague["notes"]
