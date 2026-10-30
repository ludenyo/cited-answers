import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from pgvector import Vector

from cited_answers.db import connect
from cited_answers.migrate import apply_migrations

FIXTURE_FILE = Path(__file__).parent / "fixtures" / "figures.json"
EMBEDDING_DIM = 384


def axis_vector(axis: int) -> list[float]:
    """A unit vector along one axis: distinct, with an obvious nearest neighbour."""
    vector = [0.0] * EMBEDDING_DIM
    vector[axis] = 1.0
    return vector


def load_fixture_figures() -> list[dict[str, Any]]:
    return json.loads(FIXTURE_FILE.read_text())


@pytest.fixture
def db() -> Iterator[psycopg.Connection]:
    """A connection with a fresh, throwaway schema containing the migrated tables.

    Skipped unless TEST_DATABASE_URL is set. A per-test schema keeps tests
    isolated without needing a separate database per run.
    """
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    schema = f"test_{uuid.uuid4().hex[:8]}"
    # autocommit so DDL and the extension creation are visible immediately
    with connect(url, register=False) as setup:
        setup.autocommit = True
        setup.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn = connect(url)
    conn.autocommit = True
    conn.execute(f"CREATE SCHEMA {schema}")
    # public stays on the path so the vector type from the extension resolves
    conn.execute(f"SET search_path TO {schema}, public")
    apply_migrations(conn)
    try:
        yield conn
    finally:
        conn.execute(f"DROP SCHEMA {schema} CASCADE")
        conn.close()


@pytest.fixture
def seeded_db(db: psycopg.Connection) -> psycopg.Connection:
    """The db fixture plus the five fixture figures and one chunk each.

    Chunk i has the unit vector along axis i, so a query vector on axis i
    has distance 0 to figure i and distance 1 to the others.
    """
    for fig in load_fixture_figures():
        db.execute(
            "INSERT INTO figures (id, country_code, country_name, indicator_code, "
            "indicator_name, year, value, unit, source_url) "
            "VALUES (%(id)s, %(country_code)s, %(country_name)s, %(indicator_code)s, "
            "%(indicator_name)s, %(year)s, %(value)s, %(unit)s, %(source_url)s)",
            fig,
        )
        db.execute(
            "INSERT INTO figure_chunks (figure_id, text, embedding) VALUES (%s, %s, %s)",
            (fig["id"], f"chunk {fig['id']}", Vector(axis_vector(fig["axis"]))),
        )
    return db
