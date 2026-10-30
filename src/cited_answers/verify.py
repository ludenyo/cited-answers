"""Check every claim in an answer against the figures it cites."""

from collections.abc import Callable
from decimal import Decimal

from cited_answers.models import Claim, FigureResult, Operation, VerifiedClaim
from cited_answers.settings import VerificationConfig

Lookup = Callable[[str], FigureResult | None]


def _within_tolerance(claimed: Decimal, actual: Decimal, config: VerificationConfig) -> bool:
    allowed = max(config.abs_tolerance, config.rel_tolerance * abs(actual))
    return abs(claimed - actual) <= allowed


def values_match(claimed: float, actual: float, config: VerificationConfig) -> bool:
    """Within max(abs_tolerance, rel_tolerance * |actual|) of the stored value.

    The absolute term covers small values such as percentages, where rounding to
    a decimal place matters. The relative term covers large values such as
    population, where an absolute 0.05 would reject any rounding at all.
    str() before Decimal avoids float noise at the boundary.
    """
    return _within_tolerance(Decimal(str(claimed)), Decimal(str(actual)), config)


def compute_derived(operation: Operation, first: Decimal, second: Decimal) -> Decimal | None:
    """Apply an operation to [from, to]. None when it is undefined (divide by zero)."""
    if operation == "difference":
        return second - first
    if operation == "percent_change":
        return None if first == 0 else (second - first) / first * 100
    return None if second == 0 else first / second  # ratio


def _failed(
    claim: Claim, reason: str, urls: list[str], actual: float | None = None
) -> VerifiedClaim:
    return VerifiedClaim(
        **claim.model_dump(),
        passed=False,
        actual_value=actual,
        source_url=None,
        source_urls=urls,
        reason=reason,
    )


def _verify_figure_claim(claim: Claim, lookup: Lookup, config: VerificationConfig) -> VerifiedClaim:
    assert claim.figure_id is not None  # guaranteed by Claim's validator for kind "figure"
    figure = lookup(claim.figure_id)
    if figure is None:
        return _failed(claim, "figure not found", [])
    matches = values_match(claim.value, figure.value, config)
    return VerifiedClaim(
        **claim.model_dump(),
        passed=matches,
        actual_value=figure.value,
        source_url=figure.source_url,
        source_urls=[figure.source_url],
        reason=None if matches else "value mismatch",
    )


def _verify_derived_claim(
    claim: Claim, lookup: Lookup, config: VerificationConfig
) -> VerifiedClaim:
    """Recompute the value from the stored source figures and compare it to the claim."""
    assert claim.operation is not None  # guaranteed by Claim's validator for kind "derived"
    figures = [lookup(figure_id) for figure_id in claim.source_figure_ids]
    urls = [f.source_url for f in figures if f is not None]
    for figure_id, figure in zip(claim.source_figure_ids, figures, strict=True):
        if figure is None:
            return _failed(claim, f"source figure not found: {figure_id}", urls)
    first, second = (Decimal(str(f.value)) for f in figures if f is not None)
    computed = compute_derived(claim.operation, first, second)
    if computed is None:
        return _failed(claim, "cannot compute: division by zero", urls)
    matches = _within_tolerance(Decimal(str(claim.value)), computed, config)
    return VerifiedClaim(
        **claim.model_dump(),
        passed=matches,
        actual_value=float(computed),
        source_url=None,
        source_urls=urls,
        reason=None if matches else "value mismatch",
    )


def verify_claim(claim: Claim, lookup: Lookup, config: VerificationConfig) -> VerifiedClaim:
    if claim.kind == "derived":
        return _verify_derived_claim(claim, lookup, config)
    return _verify_figure_claim(claim, lookup, config)


def verify_claims(
    claims: list[Claim], lookup: Lookup, config: VerificationConfig
) -> list[VerifiedClaim]:
    """The lookup is injected so this stays testable without a database."""
    return [verify_claim(claim, lookup, config) for claim in claims]


def pass_rate(verified: list[VerifiedClaim]) -> float:
    """Share of claims that passed; 1.0 when there are none to check."""
    if not verified:
        return 1.0
    return sum(claim.passed for claim in verified) / len(verified)
