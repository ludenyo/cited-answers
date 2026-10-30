from decimal import Decimal

from cited_answers.models import Claim, FigureResult
from cited_answers.settings import VerificationConfig
from cited_answers.verify import pass_rate, values_match, verify_claims

CONFIG = VerificationConfig(abs_tolerance=Decimal("0.05"), rel_tolerance=Decimal("0.005"))


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


FIGURES = {"a": _figure("a", 5.72), "pop": _figure("pop", 55339003.0)}


def _lookup(figure_id: str) -> FigureResult | None:
    return FIGURES.get(figure_id)


def _claim(figure_id: str, value: float) -> Claim:
    return Claim(statement="s", value=value, figure_id=figure_id)


def test_exact_match_passes() -> None:
    [result] = verify_claims([_claim("a", 5.72)], _lookup, CONFIG)
    assert result.passed and result.reason is None
    assert result.actual_value == 5.72
    assert result.source_url == "https://example.test/a"


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


def test_absolute_tolerance_boundary_is_exact_not_float_noise() -> None:
    # For 5.72 the relative term (0.0286) is smaller, so the absolute 0.05 decides.
    assert values_match(5.77, 5.72, CONFIG)
    assert not values_match(5.78, 5.72, CONFIG)


def test_relative_tolerance_applies_to_large_values() -> None:
    # Allowed is 0.5% of 55,339,003, about 276,695.
    assert values_match(55_300_000, 55_339_003, CONFIG)  # "55.3 million" spelled out
    assert not values_match(55_000_000, 55_339_003, CONFIG)  # off by about 0.6%
    assert not values_match(55.3, 55_339_003, CONFIG)  # rescaled to millions


def test_larger_of_the_two_tolerances_wins() -> None:
    # At 100 the relative term (0.5) beats the absolute term (0.05).
    assert values_match(100.5, 100.0, CONFIG)
    assert not values_match(100.6, 100.0, CONFIG)
    # Near zero the absolute term still gives some room.
    assert values_match(0.04, 0.0, CONFIG)
    assert not values_match(0.06, 0.0, CONFIG)


def test_tolerances_come_from_the_config_not_the_code() -> None:
    strict = VerificationConfig(abs_tolerance=Decimal("0"), rel_tolerance=Decimal("0"))
    assert not values_match(5.7, 5.72, strict)
    assert values_match(5.72, 5.72, strict)
    loose = VerificationConfig(abs_tolerance=Decimal("1"), rel_tolerance=Decimal("0"))
    assert values_match(6.5, 5.72, loose)


def test_pass_rate() -> None:
    claims = [_claim("a", 5.72), _claim("a", 9.0), _claim("nope", 1)]
    results = verify_claims(claims, _lookup, CONFIG)
    assert pass_rate(results) == 1 / 3
    assert pass_rate([]) == 1.0
