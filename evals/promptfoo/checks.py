"""The eval assertions as plain functions, so they can be tested without promptfoo.

promptfoo calls the wrappers at the bottom (see assertions.py); everything else takes its
inputs as arguments and has no side effects.
"""

import json
import re
from typing import Any

from cited_answers.models import Claim
from cited_answers.settings import VerificationConfig
from cited_answers.verify import Lookup, verify_claims

Check = tuple[bool, str]  # (passed, reason)
Pair = tuple[str, str]  # (country_code, indicator_code)

# The fields a Claim carries; the response's claims also hold the verification result.
CLAIM_FIELDS = ("statement", "value", "kind", "figure_id", "source_figure_ids", "operation")


def parse_response(output: Any) -> dict[str, Any]:
    """The provider returns the AskResponse as JSON text; promptfoo may also pass it parsed."""
    parsed = json.loads(output) if isinstance(output, str) else output
    if not isinstance(parsed, dict):
        raise ValueError("the provider output is not a JSON object")
    return parsed


def check_claims(response: dict[str, Any], lookup: Lookup, config: VerificationConfig) -> Check:
    """Run every returned claim, derived ones included, through verify.py again.

    The pipeline verified them once already; checking again against the database makes
    this test independent of the pipeline's own bookkeeping.
    """
    claims = [Claim.model_validate({k: raw[k] for k in CLAIM_FIELDS}) for raw in response["claims"]]
    failed = [v for v in verify_claims(claims, lookup, config) if not v.passed]
    if failed:
        return False, "; ".join(f"{v.statement!r}: {v.reason}" for v in failed)
    return True, f"{len(claims)} claims verified"


def check_flag(response: dict[str, Any], expected: bool) -> Check:
    actual = bool(response["insufficient_data"])
    return actual == expected, f"insufficient_data is {actual}, expected {expected}"


def check_refusal(
    response: dict[str, Any], expect_refused: bool, guardrails_include: list[str]
) -> Check:
    """Refused exactly when the eval file says so, and every named guardrail fired."""
    refused = bool(response.get("refused", False))
    if refused != expect_refused:
        return False, f"refused is {refused}, expected {expect_refused}"
    missing = sorted(set(guardrails_include) - set(response.get("guardrails", [])))
    if missing:
        return False, f"guardrails that did not fire: {missing}"
    return True, f"guardrails: {response.get('guardrails', [])}"


def cited_years(response: dict[str, Any], pair: Pair) -> list[int]:
    """Years of the cited figures for one country and indicator. Ids look like KEN:X.Y:2023."""
    pattern = re.compile(rf"^{re.escape(pair[0])}:{re.escape(pair[1])}:(\d{{4}})$")
    ids: list[str] = []
    for claim in response["claims"]:
        ids += [claim["figure_id"] or "", *claim["source_figure_ids"]]
    return [int(m.group(1)) for i in ids if (m := pattern.match(i))]


def check_latest_year(response: dict[str, Any], latest_years: dict[Pair, int]) -> Check:
    """The newest cited year for each pair must be the newest year in the database.

    Citing older years as well is fine (a trend); citing only older ones fails.
    """
    for pair, latest in latest_years.items():
        years = cited_years(response, pair)
        if not years or max(years) != latest:
            return (
                False,
                f"{pair[0]} {pair[1]}: cited years {sorted(set(years))}, latest is {latest}",
            )
    return True, "newest cited years match the database"


def rubric_view(response: dict[str, Any]) -> str:
    """What the rubric grader sees: the answer text and the citation behind each number.

    One explicit "cited" field per claim. Showing the raw figure_id and source_figure_ids
    fields made the grader read an empty source list on a figure claim as "no citation".
    """
    claims = [
        {
            "statement": raw["statement"],
            "value": raw["value"],
            "cited": raw["figure_id"] or raw["source_figure_ids"],
            "verified_against_database": raw["passed"],
        }
        for raw in response["claims"]
    ]
    return json.dumps({"answer": response["answer"], "claims": claims}, indent=2)
