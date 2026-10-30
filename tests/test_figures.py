from decimal import Decimal

from cited_answers.figures import (
    Figure,
    build_sentence,
    derive_unit,
    format_value,
    make_figure_id,
)


def _figure(value: str, name: str = "GDP growth (annual %)", year: int = 2023) -> Figure:
    return Figure(
        id="KEN:NY.GDP.MKTP.KD.ZG:2023",
        country_code="KEN",
        country_name="Kenya",
        indicator_code="NY.GDP.MKTP.KD.ZG",
        indicator_name=name,
        year=year,
        value=Decimal(value),
        unit="annual %",
        source_url="https://example.test",
    )


def test_make_figure_id_is_deterministic() -> None:
    assert make_figure_id("KEN", "NY.GDP.MKTP.KD.ZG", 2023) == "KEN:NY.GDP.MKTP.KD.ZG:2023"
    assert make_figure_id("KEN", "NY.GDP.MKTP.KD.ZG", 2023) == make_figure_id(
        "KEN", "NY.GDP.MKTP.KD.ZG", 2023
    )


def test_build_sentence_matches_spec_example() -> None:
    assert build_sentence(_figure("5.6")) == "Kenya's GDP growth (annual %) in 2023 was 5.6."


def test_build_sentence_formats_large_numbers() -> None:
    sentence = build_sentence(_figure("55100000", name="Population, total"))
    assert sentence == "Kenya's Population, total in 2023 was 55,100,000."


def test_format_value_trims_trailing_zeros_and_rounds() -> None:
    assert format_value(Decimal("5.60")) == "5.6"
    assert format_value(Decimal("5")) == "5"
    assert format_value(Decimal("5.678")) == "5.68"


def test_derive_unit_from_parenthesised_name() -> None:
    assert derive_unit("X", "GDP growth (annual %)", {}) == "annual %"
    assert derive_unit("X", "Mortality, under-5 (per 1,000 live births)", {}) == (
        "per 1,000 live births"
    )


def test_derive_unit_skips_trailing_source_qualifier() -> None:
    name = "Unemployment, total (% of total labor force) (modeled ILO estimate)"
    assert derive_unit("SL.UEM.TOTL.ZS", name, {}) == "% of total labor force"


def test_derive_unit_falls_back_to_last_group_when_none_looks_like_a_unit() -> None:
    assert derive_unit("X", "Index of things (base 2010)", {}) == "base 2010"


def test_derive_unit_prefers_override_and_handles_missing() -> None:
    assert derive_unit("SP.POP.TOTL", "Population, total", {"SP.POP.TOTL": "people"}) == "people"
    assert derive_unit("SP.POP.TOTL", "Population, total", {}) == ""
