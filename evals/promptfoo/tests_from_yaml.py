"""Builds the promptfoo test cases from evals/questions.yaml, the single source of truth."""

from pathlib import Path
from typing import Any

import yaml

QUESTIONS_FILE = Path(__file__).parents[1] / "questions.yaml"
ASSERTIONS = "file://assertions.py"

RUBRIC = "Answers the question directly; no number without a citation."
# The rubric grader reads the answer plus each claim's citation, not the raw response.
RUBRIC_TRANSFORM = "file://rubric_view.py:rubric_view"
LATENCY_MS = 20_000


def _asserts(expect: dict[str, Any]) -> list[dict[str, Any]]:
    refused = bool(expect.get("refused", False))
    asserts: list[dict[str, Any]] = [
        {"type": "python", "value": f"{ASSERTIONS}:claims_verified", "metric": "claims"},
        {
            "type": "python",
            "value": f"{ASSERTIONS}:insufficient_data_matches",
            "config": {"insufficient_data": expect["insufficient_data"]},
            "metric": "insufficient_data",
        },
        {
            "type": "python",
            "value": f"{ASSERTIONS}:refusal_matches",
            "config": {
                "refused": refused,
                "guardrails_include": expect.get("guardrails_include", []),
            },
            "metric": "guardrails",
        },
        {"type": "latency", "threshold": LATENCY_MS, "metric": "latency"},
    ]
    if expect.get("latest_year_check"):
        asserts.append(
            {
                "type": "python",
                "value": f"{ASSERTIONS}:latest_year_is_newest",
                "config": {"pairs": expect["latest_year_check"]},
                "metric": "latest_year",
            }
        )
    if not refused:  # a refusal does not answer the question, so the rubric cannot apply
        asserts.append(
            {
                "type": "llm-rubric",
                "value": RUBRIC,
                "transform": RUBRIC_TRANSFORM,
                "metric": "rubric",
            }
        )
    return asserts


def generate_tests(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    raw = yaml.safe_load(QUESTIONS_FILE.read_text())["questions"]
    return [
        {
            "description": f"{q['id']}: {q['question']}",
            "vars": {"question": q["question"]},
            "assert": _asserts(q["expect"]),
            "metadata": {"id": q["id"], "category": q["category"]},
        }
        for q in raw
    ]
