"""Prepare the database the evals run against, and print its URL.

Run with: uv run python scripts/eval_db.py
It migrates, then replaces the contents of the figures tables with tests/fixtures/eval_figures.sql.

Which database: EVAL_DATABASE_URL when set (CI). Otherwise the DATABASE_URL database with
"_eval" added to its name, created if missing, so a local run never wipes the real data.
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from cited_answers.db import connect
from cited_answers.migrate import apply_migrations

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "eval_figures.sql"


def eval_url(environ: dict[str, str] | os._Environ[str] = os.environ) -> str:
    if environ.get("EVAL_DATABASE_URL"):
        return environ["EVAL_DATABASE_URL"]
    base = urlsplit(environ["DATABASE_URL"])
    # Edit the URL itself so it stays a URL: other tools read this value as DATABASE_URL.
    return urlunsplit(base._replace(path=f"{base.path}_eval"))


def ensure_database(url: str) -> None:
    """Create the database if it does not exist, by connecting to the server's own 'postgres'."""
    name = conninfo_to_dict(url)["dbname"]
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def replace_figures(conn: psycopg.Connection, fixture: Path = FIXTURE) -> int:
    """Empty the figures tables and load the fixture. Returns the number of figures."""
    conn.execute("TRUNCATE figure_chunks, figures")
    conn.execute(fixture.read_text())
    count = conn.execute("SELECT count(*) FROM figures").fetchone()
    return int(count[0]) if count else 0


def load_fixture(url: str, fixture: Path = FIXTURE) -> int:
    """Migrate, then replace the figures with the fixture's. Returns the number of figures."""
    with connect(url, register=False) as conn:
        apply_migrations(conn)  # the vector type exists only after the first migration
    with connect(url) as conn:
        return replace_figures(conn, fixture)


def main() -> None:
    url = eval_url()
    ensure_database(url)
    count = load_fixture(url)
    print(f"{count} figures loaded", file=sys.stderr)
    print(url)  # the only thing on stdout, so a shell can capture it


if __name__ == "__main__":
    main()
