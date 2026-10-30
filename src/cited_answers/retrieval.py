"""Reading figures back out of Postgres: semantic search and exact lookups.

Every query is parameterised; no value is ever formatted into the SQL text.
"""

from collections.abc import Callable
from typing import Any

import psycopg
from pgvector import Vector
from psycopg.rows import dict_row

from cited_answers.embeddings import embed_texts
from cited_answers.models import FigureResult

EmbedFn = Callable[[list[str]], list[list[float]]]

_COLUMNS = """f.id AS figure_id, f.country_code, f.country_name, f.indicator_code,
       f.indicator_name, f.year, f.value, f.unit, f.source_url"""

_SEARCH = f"""
SELECT {_COLUMNS}, c.embedding <=> %(vec)s AS distance
FROM figure_chunks c
JOIN figures f ON f.id = c.figure_id
ORDER BY c.embedding <=> %(vec)s
LIMIT %(k)s
"""

_BY_KEY = f"""
SELECT {_COLUMNS} FROM figures f
WHERE f.country_code = %s AND f.indicator_code = %s AND f.year = %s
"""

_BY_ID = f"SELECT {_COLUMNS} FROM figures f WHERE f.id = %s"


def _to_figure(row: dict[str, Any]) -> FigureResult:
    # numeric columns arrive as Decimal; the model and JSON use plain floats.
    return FigureResult.model_validate({**row, "value": float(row["value"])})


def _fetch_one(conn: psycopg.Connection, sql: str, params: tuple[Any, ...]) -> FigureResult | None:
    with conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(sql, params).fetchone()
    return _to_figure(row) if row else None


def search_figures(
    conn: psycopg.Connection,
    question: str,
    k: int = 10,
    embed: EmbedFn = embed_texts,
) -> list[FigureResult]:
    """The k figures whose sentence is closest to the question by cosine distance.

    The same embedding function and model as ingestion is used on purpose:
    vectors from different models are not comparable.
    """
    vector = Vector(embed([question])[0])
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(_SEARCH, {"vec": vector, "k": k}).fetchall()
    return [_to_figure(row) for row in rows]


def get_figure(
    conn: psycopg.Connection, country_code: str, indicator_code: str, year: int
) -> FigureResult | None:
    """Exact lookup by country, indicator and year."""
    return _fetch_one(conn, _BY_KEY, (country_code, indicator_code, year))


def get_figure_by_id(conn: psycopg.Connection, figure_id: str) -> FigureResult | None:
    """Exact lookup by figure id, used to check citations."""
    return _fetch_one(conn, _BY_ID, (figure_id,))
