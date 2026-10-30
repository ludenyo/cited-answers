from decimal import Decimal

from cited_answers.models import Claim, FigureResult
from cited_answers.settings import VerificationConfig
from cited_answers.verify import (
    compute_derived,
    pass_rate,
    values_match,
    verify_claims,
)

CONFIG = VerificationConfig(abs_tolerance=Decimal("0.05"), rel_tolerance=Decimal("0.001"))


def _figure(figure_id: str, value: float) -> FigureResult:
    return FigureResult(
        figure_id=figure_id,
        country_code="KEN",
        country_name="Kenya",
        indicator_code="X",
        indicator_name="X",
        year=2023,
        value=value,
        unit=None,
        source_url=f"https://example.test/{figure_id}",
    )


FIGURES = {
    "a": _figure("a", 5.72),
    "pop": _figure("pop", 55339003.0),
    "pop13": _figure("pop13", 44986791.0),
    "m2000": _figure("m2000", 128.0),
    "m2022": _figure("m2022", 40.1),
    "infl21": _figure("infl21", 6.10793603659176),
    "infl23": _figure("infl23", 7.67139634029402),
    "zero": _figure("zero", 0.0),
}


def _lookup(figure_id: str) -> FigureResult | None:
    return FIGURES.get(figure_id)


def _claim(figure_id: str, value: float) -> Claim:
    return Claim(
        statement="s",
        value=value,
        kind="figure",
        figure_id=figure_id,
        source_figure_ids=[],
        operation=None,
    )


def _derived(operation: str, sources: list[str], value: float) -> Claim:
    return Claim.model_validate(
        {
            "statement": "s",
            "value": value,
            "kind": "derived",
            "figure_id": None,
            "source_figure_ids": sources,
            "operation": operation,
        }
    )


# ---- figure claims ------------------------------------------------------------


def test_exact_match_passes() -> None:
    [result] = verify_claims([_claim("a", 5.72)], _lookup, CONFIG)
    assert result.passed and result.reason is None
    assert result.actual_value == 5.72
    assert result.source_url == "https://example.test/a"
    assert result.source_urls == ["https://example.test/a"]


def test_rounding_to_one_decimal_passes() -> None:
    assert verify_claims([_claim("a", 5.7)], _lookup, CONFIG)[0].passed


def test_mismatch_fails_with_reason_and_actual_value() -> None:
    [result] = verify_claims([_claim("a", 6.2)], _lookup, CONFIG)
    assert not result.passed
    assert result.reason == "value mismatch"
    assert result.actual_value == 5.72
    assert result.source_url == "https://example.test/a"


def test_missing_figure_fails() -> None:
    [result] = verify_claims([_claim("nope", 1.0)], _lookup, CONFIG)
    assert not result.passed
    assert result.reason == "figure not found"
    assert result.actual_value is None and result.source_url is None
    assert result.source_urls == []


def test_absolute_tolerance_boundary_is_exact_not_float_noise() -> None:
    # For 5.72 the relative term (0.00572) is smaller, so the absolute 0.05 decides.
    assert values_match(5.77, 5.72, CONFIG)
    assert not values_match(5.78, 5.72, CONFIG)


def test_relative_tolerance_applies_to_large_values() -> None:
    # Allowed is 0.1% of 55,339,003, about 55,339.
    assert values_match(55_300_000, 55_339_003, CONFIG)  # "55.3 million" spelled out
    assert not values_match(55_200_000, 55_339_003, CONFIG)  # off by about 0.25%
    assert not values_match(55.3, 55_339_003, CONFIG)  # rescaled to millions


def test_larger_of_the_two_tolerances_wins() -> None:
    # At 100 the relative term (0.1) beats the absolute term (0.05).
    assert values_match(100.1, 100.0, CONFIG)
    assert not values_match(100.2, 100.0, CONFIG)
    # Near zero the absolute term still gives some room.
    assert values_match(0.04, 0.0, CONFIG)
    assert not values_match(0.06, 0.0, CONFIG)


def test_tolerances_come_from_the_config_not_the_code() -> None:
    strict = VerificationConfig(abs_tolerance=Decimal("0"), rel_tolerance=Decimal("0"))
    assert not values_match(5.7, 5.72, strict)
    assert values_match(5.72, 5.72, strict)
    loose = VerificationConfig(abs_tolerance=Decimal("1"), rel_tolerance=Decimal("0"))
    assert values_match(6.5, 5.72, loose)


# ---- derived claims -----------------------------------------------------------


def test_compute_derived_definitions() -> None:
    assert compute_derived("difference", Decimal(10), Decimal(15)) == Decimal(5)
    assert compute_derived("percent_change", Decimal(50), Decimal(60)) == Decimal(20)
    assert compute_derived("ratio", Decimal(10), Decimal(4)) == Decimal("2.5")


def test_compute_derived_undefined_cases_return_none() -> None:
    assert compute_derived("percent_change", Decimal(0), Decimal(5)) is None
    assert compute_derived("ratio", Decimal(5), Decimal(0)) is None
    assert compute_derived("difference", Decimal(0), Decimal(0)) == Decimal(0)


def test_derived_difference_passes_and_reports_recomputed_value() -> None:
    [result] = verify_claims([_derived("difference", ["infl21", "infl23"], 1.56)], _lookup, CONFIG)
    assert result.passed and result.reason is None
    assert abs(result.actual_value - 1.5635) < 1e-3  # type: ignore[operator]
    assert result.source_url is None
    assert result.source_urls == ["https://example.test/infl21", "https://example.test/infl23"]


def test_derived_percent_change_on_large_values() -> None:
    claim = _derived("percent_change", ["pop13", "pop"], 23.0)
    [result] = verify_claims([claim], _lookup, CONFIG)
    assert result.passed
    assert abs(result.actual_value - 23.0117) < 1e-3  # type: ignore[operator]


def test_derived_fall_must_carry_the_formulas_sign() -> None:
    sources = ["m2000", "m2022"]
    assert verify_claims([_derived("difference", sources, -87.9)], _lookup, CONFIG)[0].passed
    wrong_sign = verify_claims([_derived("difference", sources, 87.9)], _lookup, CONFIG)[0]
    assert not wrong_sign.passed and wrong_sign.reason == "value mismatch"
    assert wrong_sign.actual_value == -87.9  # the report shows what the data says


def test_derived_source_order_matters() -> None:
    reversed_sources = ["m2022", "m2000"]  # to - from would now be +87.9
    assert not verify_claims([_derived("difference", reversed_sources, -87.9)], _lookup, CONFIG)[
        0
    ].passed


def test_derived_ratio() -> None:
    [result] = verify_claims([_derived("ratio", ["m2000", "m2022"], 3.19)], _lookup, CONFIG)
    assert result.passed  # 128 / 40.1 = 3.1920


def test_derived_wrong_value_fails() -> None:
    [result] = verify_claims([_derived("difference", ["infl21", "infl23"], 2.5)], _lookup, CONFIG)
    assert not result.passed and result.reason == "value mismatch"


def test_derived_with_a_missing_source_fails_and_names_it() -> None:
    [result] = verify_claims([_derived("difference", ["a", "ghost"], 1.0)], _lookup, CONFIG)
    assert not result.passed
    assert result.reason == "source figure not found: ghost"
    assert result.actual_value is None
    assert result.source_urls == ["https://example.test/a"]  # the one that was found


def test_derived_percent_change_from_zero_fails_cleanly() -> None:
    [result] = verify_claims([_derived("percent_change", ["zero", "a"], 10.0)], _lookup, CONFIG)
    assert not result.passed
    assert result.reason == "cannot compute: division by zero"


def test_derived_claims_use_the_configured_tolerances() -> None:
    claim = _derived("difference", ["infl21", "infl23"], 1.6)  # true value 1.5635, off by 0.0365
    assert verify_claims([claim], _lookup, CONFIG)[0].passed
    strict = VerificationConfig(abs_tolerance=Decimal("0.01"), rel_tolerance=Decimal("0"))
    assert not verify_claims([claim], _lookup, strict)[0].passed


# ---- mixed lists and pass rate ------------------------------------------------


def test_figure_and_derived_claims_verify_together() -> None:
    claims = [
        _claim("infl21", 6.11),
        _claim("infl23", 7.67),
        _derived("difference", ["infl21", "infl23"], 1.56),
    ]
    results = verify_claims(claims, _lookup, CONFIG)
    assert [r.passed for r in results] == [True, True, True]
    assert [r.kind for r in results] == ["figure", "figure", "derived"]
    assert pass_rate(results) == 1.0


def test_pass_rate() -> None:
    claims = [_claim("a", 5.72), _claim("a", 9.0), _claim("nope", 1)]
    results = verify_claims(claims, _lookup, CONFIG)
    assert pass_rate(results) == 1 / 3
    assert pass_rate([]) == 1.0
