"""Minimal SQL migration runner: applies migrations/*.sql in name order, once each."""

import logging
from pathlib import Path

import psycopg

from cited_answers.db import connect

# Resolved from the source tree; fine because the project always runs via uv
# from a checkout. Pass a directory explicitly if that ever changes.
MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"

logger = logging.getLogger(__name__)


def list_migrations(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.sql"))


def _applied_versions(conn: psycopg.Connection) -> set[str]:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        "version text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
    )
    return {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}


def _apply_one(conn: psycopg.Connection, path: Path) -> None:
    # One transaction per file: a failing migration leaves nothing half-applied.
    with conn.transaction():
        conn.execute(path.read_text())
        conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (path.name,))


def apply_migrations(conn: psycopg.Connection, directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every migration not yet recorded. Returns the names applied."""
    done = _applied_versions(conn)
    applied: list[str] = []
    for path in list_migrations(directory):
        if path.name not in done:
            _apply_one(conn, path)
            applied.append(path.name)
    return applied


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    with connect(register=False) as conn:
        applied = apply_migrations(conn)
    logger.info("Applied migrations: %s", applied or "none (up to date)")


if __name__ == "__main__":
    main()
