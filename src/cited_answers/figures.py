"""The Figure record and the pure helpers that build its id, unit and sentence."""

import re
from dataclasses import dataclass
from decimal import Decimal

# Matches the last parenthesised group at the end of an indicator name,
# e.g. "GDP growth (annual %)" -> "annual %".
_UNIT_PATTERN = re.compile(r"\(([^()]*)\)\s*$")


@dataclass(frozen=True)
class Figure:
    id: str
    country_code: str
    country_name: str
    indicator_code: str
    indicator_name: str
    year: int
    value: Decimal
    unit: str
    source_url: str


def make_figure_id(country_code: str, indicator_code: str, year: int) -> str:
    """Deterministic id, so re-ingesting the same figure always hits the same row."""
    return f"{country_code}:{indicator_code}:{year}"


def derive_unit(indicator_code: str, indicator_name: str, overrides: dict[str, str]) -> str:
    """Explicit override first, then the parenthesised part of the name, else empty."""
    if indicator_code in overrides:
        return overrides[indicator_code]
    match = _UNIT_PATTERN.search(indicator_name)
    return match.group(1) if match else ""


def format_value(value: Decimal) -> str:
    """Readable number for the sentence: thousands separators, up to 2 decimals.

    Rounding here only affects the text that gets embedded; the exact value is
    stored in the figures table.
    """
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def build_sentence(figure: Figure) -> str:
    """e.g. "Kenya's GDP growth (annual %) in 2023 was 5.6." """
    return (
        f"{figure.country_name}'s {figure.indicator_name} in {figure.year} "
        f"was {format_value(figure.value)}."
    )
