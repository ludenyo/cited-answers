import json
from pathlib import Path
from typing import Any

import httpx2
import psycopg

from cited_answers.ingest_worldbank import IngestConfig, load_config, run

CONFIG = IngestConfig(countries=["KEN"], indicators=["NY.GDP.MKTP.KD.ZG"], start_year=2020)


def _record(year: str, value: float | None) -> dict[str, Any]:
    return {
        "indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
        "country": {"id": "KE", "value": "Kenya"},
        "countryiso3code": "KEN",
        "date": year,
        "value": value,
    }


def _client(rows: list[dict[str, Any]]) -> httpx2.Client:
    def handler(request: httpx2.Request) -> httpx2.Response:
        meta = {"page": 1, "pages": 1, "per_page": 1000, "total": len(rows)}
        return httpx2.Response(200, content=json.dumps([meta, rows]))

    return httpx2.Client(transport=httpx2.MockTransport(handler))


class _CountingEmbed:
    def __init__(self) -> None:
        self.texts_embedded = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        return [[0.1] * 384 for _ in texts]


def _count(db: psycopg.Connection, table: str) -> int:
    row = db.execute(f"SELECT count(*) FROM {table}").fetchone()  # noqa: S608 - fixed names
    assert row is not None
    return int(row[0])


def test_load_config_reads_shipped_file() -> None:
    config = load_config(Path(__file__).parents[1] / "config" / "indicators.yaml")
    assert config.countries == ["KEN", "UGA", "TZA"]
    assert len(config.indicators) == 17
    assert config.units["SP.POP.TOTL"] == "people"


def test_run_skips_null_values_and_stores_rest(db: psycopg.Connection) -> None:
    rows = [_record("2021", 5.0), _record("2022", 4.8), _record("2023", None)]
    embed = _CountingEmbed()

    stored, embedded = run(db, _client(rows), CONFIG, embed=embed, end_year=2024)

    assert (stored, embedded) == (2, 2)
    assert _count(db, "figures") == 2
    assert _count(db, "figure_chunks") == 2


def test_run_is_idempotent(db: psycopg.Connection) -> None:
    rows = [_record("2021", 5.0), _record("2022", 4.8)]
    embed = _CountingEmbed()

    run(db, _client(rows), CONFIG, embed=embed, end_year=2024)
    _, embedded_again = run(db, _client(rows), CONFIG, embed=embed, end_year=2024)

    assert _count(db, "figures") == 2
    assert _count(db, "figure_chunks") == 2
    assert embedded_again == 0  # unchanged sentences are not re-embedded
    assert embed.texts_embedded == 2


def test_changed_value_updates_row_and_reembeds(db: psycopg.Connection) -> None:
    embed = _CountingEmbed()
    run(db, _client([_record("2021", 5.0)]), CONFIG, embed=embed, end_year=2024)

    run(db, _client([_record("2021", 5.5)]), CONFIG, embed=embed, end_year=2024)

    value = db.execute("SELECT value FROM figures WHERE id = 'KEN:NY.GDP.MKTP.KD.ZG:2021'")
    row = value.fetchone()
    assert row is not None and float(row[0]) == 5.5
    text = db.execute("SELECT text FROM figure_chunks").fetchone()
    assert text is not None and text[0].endswith("was 5.5.")
    assert _count(db, "figures") == 1
