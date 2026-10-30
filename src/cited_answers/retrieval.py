"""Reading figures back out of Postgres: semantic search and exact lookups.

Every query is parameterised; no value is ever formatted into the SQL text.
"""

from collections.abc import Callable
from typing import Any

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row

from cited_answers import tracing
from cited_answers.embeddings import MODEL_NAME, embed_texts
from cited_answers.models import FigureResult, IndicatorInfo
from cited_answers.settings import get_app_config

EmbedFn = Callable[[list[str]], list[list[float]]]

_COLUMNS = """f.id AS figure_id, f.country_code, f.country_name, f.indicator_code,
       f.indicator_name, f.year, f.value, f.unit, f.source_url,
       (SELECT max(l.year) FROM figures l
         WHERE l.country_code = f.country_code
           AND l.indicator_code = f.indicator_code) AS latest_year"""

_SEARCH = f"""
SELECT {_COLUMNS}, c.embedding <=> %(vec)s AS distance
FROM figure_chunks c
JOIN figures f ON f.id = c.figure_id
WHERE (c.embedding <=> %(vec)s) <= %(max_distance)s
ORDER BY c.embedding <=> %(vec)s
LIMIT %(k)s
"""

_INDICATORS = """
SELECT DISTINCT ON (indicator_code) indicator_code, indicator_name, unit
FROM figures
ORDER BY indicator_code
"""

_BY_KEY = f"""
SELECT {_COLUMNS} FROM figures f
WHERE f.country_code = %s AND f.indicator_code = %s AND f.year = %s
"""

_BY_ID = f"SELECT {_COLUMNS} FROM figures f WHERE f.id = %s"

_LATEST_YEAR = "SELECT max(year) FROM figures WHERE country_code = %s AND indicator_code = %s"


def _to_figure(row: dict[str, Any]) -> FigureResult:
    # numeric columns arrive as Decimal; the model and JSON use plain floats.
    return FigureResult.model_validate({**row, "value": float(row["value"])})


def _fetch_one(conn: psycopg.Connection, sql: str, params: tuple[Any, ...]) -> FigureResult | None:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(sql, params).fetchone()
    return _to_figure(row) if row else None


def _embed_question(question: str, embed: EmbedFn) -> list[float]:
    with tracing.observation(
        "embed-question", "embedding", model=MODEL_NAME, input=question
    ) as span:
        vector = embed([question])[0]
        span.update(output={"dimensions": len(vector)})  # the vector itself is just noise
    return vector


def search_figures(
    conn: psycopg.Connection,
    question: str,
    k: int | None = None,
    max_distance: float | None = None,
    embed: EmbedFn = embed_texts,
) -> list[FigureResult]:
    """Up to k figures closest to the question by cosine distance, none beyond max_distance.

    k and max_distance default to retrieval.top_k and retrieval.max_distance in
    config/app.yaml. The same embedding function and model as ingestion is used
    on purpose: vectors from different models are not comparable.
    """
    defaults = get_app_config().retrieval
    k = k if k is not None else defaults.top_k
    max_distance = max_distance if max_distance is not None else defaults.max_distance
    trace_input = {"question": question, "k": k, "max_distance": max_distance}
    with tracing.observation("search-figures", "retriever", input=trace_input) as span:
        params = {
            "vec": Vector(_embed_question(question, embed)),
            "k": k,
            "max_distance": max_distance,
        }
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(_SEARCH, params).fetchall()
        figures = [_to_figure(row) for row in rows]
        span.update(
            output=[
                {
                    "figure_id": f.figure_id,
                    "value": f.value,
                    "latest_year": f.latest_year,
                    "distance": f.distance,
                }
                for f in figures
            ],
            metadata={"result_count": len(figures)},
        )
    return figures


def list_indicators(conn: psycopg.Connection) -> list[IndicatorInfo]:
    """Every indicator in the database, once each, ordered by code."""
    with tracing.observation("list-indicators", "retriever") as span:
        with conn.cursor(row_factory=dict_row) as cur:
            rows = cur.execute(_INDICATORS).fetchall()
        indicators = [IndicatorInfo.model_validate(row) for row in rows]
        span.update(output=[i.indicator_code for i in indicators])
    return indicators


def _lookup_output(figure: FigureResult | None) -> dict[str, Any]:
    """Trace output for an exact lookup. A miss is stated outright: the SDK drops None."""
    if figure is None:
        return {"found": False}
    return {
        "found": True,
        "figure_id": figure.figure_id,
        "value": figure.value,
        "latest_year": figure.latest_year,
    }


def get_figure(
    conn: psycopg.Connection, country_code: str, indicator_code: str, year: int
) -> FigureResult | None:
    """Exact lookup by country, indicator and year."""
    trace_input = {"country_code": country_code, "indicator_code": indicator_code, "year": year}
    with tracing.observation("get-figure", "retriever", input=trace_input) as span:
        figure = _fetch_one(conn, _BY_KEY, (country_code, indicator_code, year))
        span.update(output=_lookup_output(figure))
    return figure


def get_latest_year(conn: psycopg.Connection, country_code: str, indicator_code: str) -> int | None:
    """Newest year with data for one country and indicator, or None if there is none."""
    with conn.cursor() as cur:
        row = cur.execute(_LATEST_YEAR, (country_code, indicator_code)).fetchone()
    return row[0] if row else None


def get_figure_by_id(conn: psycopg.Connection, figure_id: str) -> FigureResult | None:
    """Exact lookup by figure id, used to check citations."""
    return _fetch_one(conn, _BY_ID, (figure_id,))
