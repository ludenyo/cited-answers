"""Ingest World Bank indicator figures into Postgres, with one embedded sentence each.

Run with: uv run python -m cited_answers.ingest_worldbank
Safe to re-run: figures and chunks are upserted, and unchanged chunks are not re-embedded.
"""

import argparse
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path

import httpx2
import psycopg
import yaml
from pgvector import Vector
from pydantic import BaseModel

from cited_answers.db import connect
from cited_answers.embeddings import embed_texts
from cited_answers.figures import Figure, build_sentence
from cited_answers.worldbank import fetch_figures

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "indicators.yaml"

logger = logging.getLogger(__name__)

EmbedFn = Callable[[list[str]], list[list[float]]]

_UPSERT_FIGURE = """
INSERT INTO figures (id, country_code, country_name, indicator_code, indicator_name,
                     year, value, unit, source_url, retrieved_at)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
ON CONFLICT (id) DO UPDATE SET
    country_name = EXCLUDED.country_name,
    indicator_name = EXCLUDED.indicator_name,
    value = EXCLUDED.value,
    unit = EXCLUDED.unit,
    source_url = EXCLUDED.source_url,
    retrieved_at = now()
"""

_UPSERT_CHUNK = """
INSERT INTO figure_chunks (figure_id, text, embedding)
VALUES (%s, %s, %s)
ON CONFLICT (figure_id) DO UPDATE SET text = EXCLUDED.text, embedding = EXCLUDED.embedding
"""


class IngestConfig(BaseModel):
    countries: list[str]
    indicators: list[str]
    start_year: int = 2000
    units: dict[str, str] = {}


def load_config(path: Path) -> IngestConfig:
    return IngestConfig.model_validate(yaml.safe_load(path.read_text()))


def upsert_figures(conn: psycopg.Connection, figures: list[Figure]) -> None:
    rows = [
        (
            f.id,
            f.country_code,
            f.country_name,
            f.indicator_code,
            f.indicator_name,
            f.year,
            f.value,
            f.unit,
            f.source_url,
        )
        for f in figures
    ]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_FIGURE, rows)


def _existing_chunk_texts(conn: psycopg.Connection, ids: list[str]) -> dict[str, str]:
    rows = conn.execute(
        "SELECT figure_id, text FROM figure_chunks WHERE figure_id = ANY(%s)", (ids,)
    )
    return dict(rows.fetchall())


def sync_chunks(conn: psycopg.Connection, figures: list[Figure], embed: EmbedFn) -> int:
    """Embed and upsert chunks whose sentence is new or changed. Returns how many."""
    existing = _existing_chunk_texts(conn, [f.id for f in figures])
    pending = [(f.id, build_sentence(f)) for f in figures]
    pending = [(fid, text) for fid, text in pending if existing.get(fid) != text]
    vectors = embed([text for _, text in pending])  # one batched call
    rows = [(fid, text, Vector(vec)) for (fid, text), vec in zip(pending, vectors, strict=True)]
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_CHUNK, rows)
    return len(rows)


def fetch_all(client: httpx2.Client, config: IngestConfig, end_year: int) -> list[Figure]:
    figures: list[Figure] = []
    for indicator in config.indicators:
        found = fetch_figures(
            client, config.countries, indicator, config.start_year, end_year, config.units
        )
        logger.info("%s: %d figures", indicator, len(found))
        figures.extend(found)
    return figures


def run(
    conn: psycopg.Connection,
    client: httpx2.Client,
    config: IngestConfig,
    embed: EmbedFn = embed_texts,
    end_year: int | None = None,
) -> tuple[int, int]:
    """Fetch, upsert and embed. Returns (figures stored, chunks embedded)."""
    figures = fetch_all(client, config, end_year or date.today().year)
    with conn.transaction():
        upsert_figures(conn, figures)
        embedded = sync_chunks(conn, figures, embed)
    return len(figures), embedded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = load_config(args.config)
    with connect() as conn, httpx2.Client(timeout=30.0) as client:
        stored, embedded = run(conn, client, config)
    logger.info("Stored %d figures, embedded %d new or changed chunks", stored, embedded)


if __name__ == "__main__":
    main()
