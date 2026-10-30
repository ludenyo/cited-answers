"""promptfoo `python` assertions. Each takes (output, context) and returns a grading result."""

import sys
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import checks  # noqa: E402

from cited_answers import retrieval  # noqa: E402
from cited_answers.db import connect  # noqa: E402
from cited_answers.settings import get_app_config  # noqa: E402


def _result(check: checks.Check) -> dict[str, Any]:
    passed, reason = check
    return {"pass": passed, "score": 1.0 if passed else 0.0, "reason": reason}


def claims_verified(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    response = checks.parse_response(output)
    with connect() as conn:
        lookup = partial(retrieval.get_figure_by_id, conn)
        return _result(checks.check_claims(response, lookup, get_app_config().verification))


def insufficient_data_matches(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    expected = bool(context["config"]["insufficient_data"])
    return _result(checks.check_flag(checks.parse_response(output), expected))


def refusal_matches(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    config = context["config"]
    return _result(
        checks.check_refusal(
            checks.parse_response(output), bool(config["refused"]), config["guardrails_include"]
        )
    )


def _latest_years(pairs: list[dict[str, str]]) -> dict[checks.Pair, int]:
    with connect() as conn:
        return {
            (p["country"], p["indicator"]): conn.execute(
                "SELECT max(year) FROM figures WHERE country_code = %s AND indicator_code = %s",
                (p["country"], p["indicator"]),
            ).fetchone()[0]
            for p in pairs
        }


def latest_year_is_newest(output: Any, context: dict[str, Any]) -> dict[str, Any]:
    latest = _latest_years(context["config"]["pairs"])
    return _result(checks.check_latest_year(checks.parse_response(output), latest))
