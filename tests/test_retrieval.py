import psycopg

from cited_answers.retrieval import (
    get_figure,
    get_figure_by_id,
    list_indicators,
    search_figures,
)
from tests.conftest import axis_vector


def test_get_figure_returns_the_right_row(seeded_db: psycopg.Connection) -> None:
    figure = get_figure(seeded_db, "KEN", "NY.GDP.MKTP.KD.ZG", 2023)

    assert figure is not None
    assert figure.figure_id == "KEN:NY.GDP.MKTP.KD.ZG:2023"
    assert figure.value == 5.72
    assert figure.country_name == "Kenya"
    assert figure.unit == "annual %"
    assert figure.source_url.endswith("date=2023")
    assert figure.distance is None


def test_get_figure_missing_returns_none(seeded_db: psycopg.Connection) -> None:
    assert get_figure(seeded_db, "KEN", "NY.GDP.MKTP.KD.ZG", 1999) is None
    assert get_figure(seeded_db, "UGA", "NO.SUCH.CODE", 2023) is None


def test_get_figure_by_id(seeded_db: psycopg.Connection) -> None:
    found = get_figure_by_id(seeded_db, "KEN:SP.POP.TOTL:2023")
    assert found is not None and found.value == 55339003
    assert get_figure_by_id(seeded_db, "KEN:SP.POP.TOTL:1900") is None


def test_lookup_is_not_vulnerable_to_injection(seeded_db: psycopg.Connection) -> None:
    assert get_figure_by_id(seeded_db, "x' OR '1'='1") is None


def test_search_orders_by_cosine_distance_and_limits_k(seeded_db: psycopg.Connection) -> None:
    results = search_figures(
        seeded_db, "inflation", k=3, max_distance=2.0, embed=lambda _t: [axis_vector(4)]
    )

    assert len(results) == 3
    assert results[0].figure_id == "TZA:FP.CPI.TOTL.ZG:2023"
    assert results[0].distance is not None and results[0].distance < 1e-6
    assert all(r.distance is not None and abs(r.distance - 1.0) < 1e-6 for r in results[1:])
    assert [r.distance for r in results] == sorted(r.distance for r in results)  # type: ignore[type-var]


def test_search_drops_results_beyond_the_configured_max_distance(
    seeded_db: psycopg.Connection,
) -> None:
    # Unit vectors on different axes are at distance 1.0, beyond the 0.6 in app.yaml,
    # so only the exact match survives when no explicit limits are given.
    results = search_figures(seeded_db, "inflation", embed=lambda _t: [axis_vector(4)])

    assert [r.figure_id for r in results] == ["TZA:FP.CPI.TOTL.ZG:2023"]


def test_search_returns_nothing_when_nothing_is_close_enough(
    seeded_db: psycopg.Connection,
) -> None:
    far_away = [1.0 if i == 100 else 0.0 for i in range(384)]  # no figure sits on axis 100
    assert search_figures(seeded_db, "unrelated", embed=lambda _t: [far_away]) == []


def test_search_k_defaults_to_the_configured_top_k_and_can_be_overridden(
    seeded_db: psycopg.Connection,
) -> None:
    embed = lambda _t: [axis_vector(0)]  # noqa: E731
    assert len(search_figures(seeded_db, "q", max_distance=2.0, embed=embed)) == 5  # 5 < top_k 10
    assert len(search_figures(seeded_db, "q", k=2, max_distance=2.0, embed=embed)) == 2


def test_list_indicators_returns_each_indicator_once_in_code_order(
    seeded_db: psycopg.Connection,
) -> None:
    indicators = list_indicators(seeded_db)

    assert [i.indicator_code for i in indicators] == [
        "FP.CPI.TOTL.ZG",
        "NY.GDP.MKTP.KD.ZG",  # two fixture rows share this code but it is listed once
        "SP.DYN.LE00.IN",
        "SP.POP.TOTL",
    ]
    gdp = next(i for i in indicators if i.indicator_code == "NY.GDP.MKTP.KD.ZG")
    assert gdp.indicator_name == "GDP growth (annual %)"
    assert gdp.unit == "annual %"
