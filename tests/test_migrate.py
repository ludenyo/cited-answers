from pathlib import Path

import psycopg

from cited_answers.migrate import MIGRATIONS_DIR, apply_migrations, list_migrations


def test_list_migrations_is_sorted_and_sql_only(tmp_path: Path) -> None:
    (tmp_path / "002_b.sql").write_text("")
    (tmp_path / "001_a.sql").write_text("")
    (tmp_path / "notes.txt").write_text("")
    assert [p.name for p in list_migrations(tmp_path)] == ["001_a.sql", "002_b.sql"]


def test_initial_migration_exists() -> None:
    assert (MIGRATIONS_DIR / "001_init.sql").is_file()


def test_apply_is_idempotent_and_creates_tables(db: psycopg.Connection) -> None:
    # The db fixture already applied everything once.
    assert apply_migrations(db) == []
    tables = {
        row[0]
        for row in db.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = current_schema()"
        )
    }
    assert {"figures", "figure_chunks", "schema_migrations"} <= tables


def test_hnsw_cosine_index_exists(db: psycopg.Connection) -> None:
    row = db.execute(
        "SELECT indexdef FROM pg_indexes WHERE indexname = 'figure_chunks_embedding_idx'"
    ).fetchone()
    assert row is not None
    assert "hnsw" in row[0] and "vector_cosine_ops" in row[0]
