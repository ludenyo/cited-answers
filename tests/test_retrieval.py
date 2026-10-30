import psycopg

from cited_answers.retrieval import get_figure, get_figure_by_id, search_figures
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
    results = search_figures(seeded_db, "inflation", k=3, embed=lambda _t: [axis_vector(4)])

    assert len(results) == 3
    assert results[0].figure_id == "TZA:FP.CPI.TOTL.ZG:2023"
    assert results[0].distance is not None and results[0].distance < 1e-6
    assert all(r.distance is not None and abs(r.distance - 1.0) < 1e-6 for r in results[1:])
    assert [r.distance for r in results] == sorted(r.distance for r in results)  # type: ignore[type-var]
