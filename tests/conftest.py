import os
import uuid
from collections.abc import Iterator

import psycopg
import pytest

from cited_answers.db import connect
from cited_answers.migrate import apply_migrations


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
