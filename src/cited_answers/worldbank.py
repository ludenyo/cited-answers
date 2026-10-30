"""Client for the World Bank API v2 (JSON): pagination and record parsing."""

import logging
import time
from decimal import Decimal
from typing import Any

import httpx2

from cited_answers.figures import Figure, derive_unit, make_figure_id

BASE_URL = "https://api.worldbank.org/v2"
PER_PAGE = 1000
MAX_ATTEMPTS = 4
RETRY_BACKOFF_SECONDS = 2.0

logger = logging.getLogger(__name__)


class WorldBankError(RuntimeError):
    """The API answered with an error payload instead of data."""


def figure_source_url(country_code: str, indicator_code: str, year: int) -> str:
    """URL that returns exactly this one figure, so a citation is checkable."""
    return f"{BASE_URL}/country/{country_code}/indicator/{indicator_code}?format=json&date={year}"


def _get_with_retry(client: httpx2.Client, url: str, params: dict[str, Any]) -> httpx2.Response:
    """GET with retries. The World Bank API occasionally stalls on a single request,
    so timeouts, connection errors and 5xx responses are retried with backoff.
    Other errors (4xx) are real and raised immediately."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code < 500:
                response.raise_for_status()
                return response
            error: Exception = httpx2.HTTPStatusError(
                f"server error {response.status_code}", request=response.request, response=response
            )
        except httpx2.TransportError as exc:
            error = exc
        if attempt == MAX_ATTEMPTS:
            raise error
        logger.warning("Attempt %d/%d failed (%s); retrying", attempt, MAX_ATTEMPTS, error)
        time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
    raise AssertionError("unreachable")  # loop always returns or raises


def _get_page(
    client: httpx2.Client,
    countries: list[str],
    indicator: str,
    date_range: str,
    page: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    url = f"{BASE_URL}/country/{';'.join(countries)}/indicator/{indicator}"
    params = {"format": "json", "per_page": PER_PAGE, "date": date_range, "page": page}
    payload = _get_with_retry(client, url, params).json()
    # Errors (e.g. an unknown indicator) come back as HTTP 200 with a single
    # {"message": [...]} element instead of the usual [meta, rows] pair.
    if not isinstance(payload, list) or len(payload) < 2:
        raise WorldBankError(f"Unexpected response for {indicator}: {payload!r}")
    return payload[0], payload[1] or []


def fetch_records(
    client: httpx2.Client,
    countries: list[str],
    indicator: str,
    start_year: int,
    end_year: int,
) -> list[dict[str, Any]]:
    """All raw records for one indicator, following pagination to the last page."""
    date_range = f"{start_year}:{end_year}"
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        meta, rows = _get_page(client, countries, indicator, date_range, page)
        records.extend(rows)
        if page >= int(meta["pages"]):
            return records
        page += 1


def parse_record(record: dict[str, Any], unit_overrides: dict[str, str]) -> Figure | None:
    """Turn one API record into a Figure, or None when the value is null."""
    if record["value"] is None:
        return None
    country_code = record["countryiso3code"]
    indicator_code = record["indicator"]["id"]
    indicator_name = record["indicator"]["value"]
    year = int(record["date"])
    return Figure(
        id=make_figure_id(country_code, indicator_code, year),
        country_code=country_code,
        country_name=record["country"]["value"],
        indicator_code=indicator_code,
        indicator_name=indicator_name,
        year=year,
        # str() first: Decimal(float) would carry binary rounding noise.
        value=Decimal(str(record["value"])),
        unit=derive_unit(indicator_code, indicator_name, unit_overrides),
        source_url=figure_source_url(country_code, indicator_code, year),
    )


def fetch_figures(
    client: httpx2.Client,
    countries: list[str],
    indicator: str,
    start_year: int,
    end_year: int,
    unit_overrides: dict[str, str],
) -> list[Figure]:
    """Fetch one indicator for all countries, dropping null values."""
    records = fetch_records(client, countries, indicator, start_year, end_year)
    parsed = (parse_record(record, unit_overrides) for record in records)
    return [figure for figure in parsed if figure is not None]
