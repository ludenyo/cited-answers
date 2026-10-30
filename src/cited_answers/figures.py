"""The Figure record and the pure helpers that build its id, unit and sentence."""

import re
from dataclasses import dataclass
from decimal import Decimal

# Every parenthesised group in an indicator name, e.g. "GDP growth (annual %)".
_GROUP_PATTERN = re.compile(r"\(([^()]*)\)")
# Names can end with a source qualifier after the unit, e.g.
# "Unemployment, total (% of total labor force) (modeled ILO estimate)".
# A group counts as a unit only if it contains one of these markers.
_UNIT_HINTS = ("%", "per ", "us$", "years", "current", "constant")


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
    """Explicit override first, then the last unit-like bracketed group of the name.

    Falls back to the last bracketed group when none looks like a unit, and to
    an empty string when the name has no brackets at all.
    """
    if indicator_code in overrides:
        return overrides[indicator_code]
    groups = _GROUP_PATTERN.findall(indicator_name)
    for group in reversed(groups):
        if any(hint in group.lower() for hint in _UNIT_HINTS):
            return group
    return groups[-1] if groups else ""


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
