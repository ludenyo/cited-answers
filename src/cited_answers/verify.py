"""Check every claim in an answer against the figure it cites."""

from collections.abc import Callable
from decimal import Decimal

from cited_answers.models import Claim, FigureResult, VerifiedClaim
from cited_answers.settings import VerificationConfig

Lookup = Callable[[str], FigureResult | None]


def values_match(claimed: float, actual: float, config: VerificationConfig) -> bool:
    """Within max(abs_tolerance, rel_tolerance * |actual|) of the stored value.

    The absolute term covers small values such as percentages, where rounding to
    a decimal place matters. The relative term covers large values such as
    population, where an absolute 0.05 would reject any rounding at all.
    str() before Decimal avoids float noise at the boundary.
    """
    actual_d = Decimal(str(actual))
    allowed = max(config.abs_tolerance, config.rel_tolerance * abs(actual_d))
    return abs(Decimal(str(claimed)) - actual_d) <= allowed


def verify_claim(claim: Claim, lookup: Lookup, config: VerificationConfig) -> VerifiedClaim:
    figure = lookup(claim.figure_id)
    if figure is None:
        return VerifiedClaim(
            **claim.model_dump(),
            passed=False,
            actual_value=None,
            source_url=None,
            reason="figure not found",
        )
    matches = values_match(claim.value, figure.value, config)
    return VerifiedClaim(
        **claim.model_dump(),
        passed=matches,
        actual_value=figure.value,
        source_url=figure.source_url,
        reason=None if matches else "value mismatch",
    )


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
