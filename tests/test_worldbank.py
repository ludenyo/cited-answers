import json
from decimal import Decimal
from typing import Any

import httpx2
import pytest

from cited_answers.worldbank import (
    MAX_ATTEMPTS,
    WorldBankError,
    fetch_figures,
    fetch_records,
    parse_record,
)


def _record(year: str, value: float | None, country: str = "KEN") -> dict[str, Any]:
    return {
        "indicator": {"id": "NY.GDP.MKTP.KD.ZG", "value": "GDP growth (annual %)"},
        "country": {"id": "KE", "value": "Kenya"},
        "countryiso3code": country,
        "date": year,
        "value": value,
    }


def _client(handler: Any) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _paged_handler(pages: list[list[dict[str, Any]]], seen: list[int]) -> Any:
    def handler(request: httpx2.Request) -> httpx2.Response:
        page = int(request.url.params["page"])
        seen.append(page)
        meta = {"page": page, "pages": len(pages), "per_page": 1000, "total": 3}
        return httpx2.Response(200, content=json.dumps([meta, pages[page - 1]]))

    return handler


def test_parse_record_builds_figure() -> None:
    figure = parse_record(_record("2023", 5.6), {})
    assert figure is not None
    assert figure.id == "KEN:NY.GDP.MKTP.KD.ZG:2023"
    assert figure.value == Decimal("5.6")
    assert figure.unit == "annual %"
    assert figure.country_name == "Kenya"
    assert "NY.GDP.MKTP.KD.ZG" in figure.source_url


def test_parse_record_skips_null_value() -> None:
    assert parse_record(_record("2024", None), {}) is None


def test_zero_is_kept_not_treated_as_null() -> None:
    figure = parse_record(_record("2020", 0.0), {})
    assert figure is not None
    assert figure.value == Decimal("0.0")


def test_fetch_records_follows_pagination() -> None:
    seen: list[int] = []
    pages = [[_record("2021", 1.0)], [_record("2022", 2.0)], [_record("2023", 3.0)]]
    with _client(_paged_handler(pages, seen)) as client:
        records = fetch_records(client, ["KEN"], "NY.GDP.MKTP.KD.ZG", 2000, 2024)
    assert seen == [1, 2, 3]
    assert [r["date"] for r in records] == ["2021", "2022", "2023"]


def test_fetch_figures_drops_nulls_across_pages() -> None:
    pages = [[_record("2022", 2.0), _record("2023", None)], [_record("2021", 1.0)]]
    with _client(_paged_handler(pages, [])) as client:
        figures = fetch_figures(client, ["KEN"], "NY.GDP.MKTP.KD.ZG", 2000, 2024, {})
    assert sorted(f.year for f in figures) == [2021, 2022]


def test_request_uses_date_range_and_joined_countries() -> None:
    captured: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured.append(request)
        meta = {"page": 1, "pages": 1, "per_page": 1000, "total": 0}
        return httpx2.Response(200, content=json.dumps([meta, []]))

    with _client(handler) as client:
        fetch_records(client, ["KEN", "UGA"], "SP.POP.TOTL", 2000, 2024)
    assert "KEN;UGA" in captured[0].url.path or "KEN%3BUGA" in str(captured[0].url)
    assert captured[0].url.params["date"] == "2000:2024"


def test_timeout_is_retried_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cited_answers.worldbank.time.sleep", lambda _s: None)
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if len(calls) < 3:
            raise httpx2.ReadTimeout("stalled", request=request)
        meta = {"page": 1, "pages": 1, "per_page": 1000, "total": 1}
        return httpx2.Response(200, content=json.dumps([meta, [_record("2023", 5.6)]]))

    with _client(handler) as client:
        records = fetch_records(client, ["KEN"], "NY.GDP.MKTP.KD.ZG", 2000, 2024)
    assert len(calls) == 3
    assert len(records) == 1


def test_persistent_timeout_raises_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("cited_answers.worldbank.time.sleep", lambda _s: None)
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        raise httpx2.ReadTimeout("stalled", request=request)

    with _client(handler) as client, pytest.raises(httpx2.ReadTimeout):
        fetch_records(client, ["KEN"], "NY.GDP.MKTP.KD.ZG", 2000, 2024)
    assert len(calls) == MAX_ATTEMPTS


def test_client_error_is_not_retried() -> None:
    calls: list[int] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        return httpx2.Response(404, content="not found")

    with _client(handler) as client, pytest.raises(httpx2.HTTPStatusError):
        fetch_records(client, ["KEN"], "NY.GDP.MKTP.KD.ZG", 2000, 2024)
    assert len(calls) == 1


def test_error_payload_raises() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, content=json.dumps([{"message": [{"id": "120"}]}]))

    with _client(handler) as client, pytest.raises(WorldBankError):
        fetch_records(client, ["KEN"], "BAD.CODE", 2000, 2024)
