import asyncio
from contextlib import nullcontext
from typing import Any

import psycopg
import pytest
from mcp import Client

from cited_answers.answer import TOOLS
from cited_answers.mcp_server import build_server, tool_description
from tests.conftest import axis_vector


async def _call_async(server: Any, name: str, arguments: dict[str, Any]) -> Any:
    async with Client(server) as client:
        return await client.call_tool(name, arguments)


def _call(server: Any, name: str, arguments: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """(is_error, structured result) for one tool call, through an in-memory MCP client."""
    result = asyncio.run(_call_async(server, name, arguments))
    return bool(result.is_error), result.structured_content or {}


@pytest.fixture
def server(seeded_db: psycopg.Connection) -> Any:
    # The fixture's connection is shared, so it must not be closed after each call.
    # Figure 4 (Tanzania inflation) is the one the fake embedder points at.
    return build_server(lambda: nullcontext(seeded_db), embed=lambda _t: [axis_vector(4)])


def test_the_server_exposes_exactly_the_three_data_tools(server: Any) -> None:
    tools = asyncio.run(server.list_tools())
    assert sorted(t.name for t in tools) == ["get_figure", "list_indicators", "search_figures"]


def test_descriptions_are_the_ones_the_answering_model_sees(server: Any) -> None:
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    for tool in TOOLS:
        if tool["name"] in by_name:
            assert by_name[tool["name"]].description == tool["description"]
    assert tool_description("get_figure").startswith("Exact lookup of one figure")


def test_input_schemas_match_the_answering_tools(server: Any) -> None:
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    get_figure = by_name["get_figure"].input_schema["properties"]
    assert get_figure["country_code"]["enum"] == ["KEN", "UGA", "TZA"]
    assert set(by_name["get_figure"].input_schema["required"]) == {
        "country_code",
        "indicator_code",
        "year",
    }
    assert by_name["search_figures"].input_schema["required"] == ["question"]


# ---- search_figures ------------------------------------------------------------------


def test_search_figures_returns_the_nearest_figure_with_its_citation_fields(server: Any) -> None:
    is_error, result = _call(server, "search_figures", {"question": "inflation in Tanzania"})
    assert not is_error
    [figure] = result["figures"]
    assert figure["figure_id"] == "TZA:FP.CPI.TOTL.ZG:2023"
    assert figure["source_url"].startswith("https://api.worldbank.org/")
    assert figure["latest_year"] == 2023 and figure["value"] == 3.8


def test_search_figures_accepts_and_clamps_k(server: Any) -> None:
    assert not _call(server, "search_figures", {"question": "q", "k": 3})[0]
    assert not _call(server, "search_figures", {"question": "q", "k": 500})[0]  # clamped to 20


def test_search_figures_with_nothing_close_returns_an_empty_list(
    seeded_db: psycopg.Connection,
) -> None:
    far = [0.0] * 384
    far[100] = 1.0
    server = build_server(lambda: nullcontext(seeded_db), embed=lambda _t: [far])
    assert _call(server, "search_figures", {"question": "unrelated"}) == (False, {"figures": []})


def test_search_figures_requires_a_question(server: Any) -> None:
    is_error, _ = _call(server, "search_figures", {})
    assert is_error


# ---- get_figure ----------------------------------------------------------------------


def test_get_figure_returns_the_exact_row_with_source_url_and_latest_year(server: Any) -> None:
    is_error, result = _call(
        server,
        "get_figure",
        {"country_code": "UGA", "indicator_code": "SP.DYN.LE00.IN", "year": 2023},
    )
    assert not is_error
    figure = result["figure"]
    assert figure["figure_id"] == "UGA:SP.DYN.LE00.IN:2023"
    assert figure["value"] == 68.6
    assert figure["source_url"].endswith("date=2023")
    assert figure["latest_year"] == 2023


def test_get_figure_miss_says_how_far_the_series_goes(server: Any) -> None:
    is_error, result = _call(
        server,
        "get_figure",
        {"country_code": "UGA", "indicator_code": "SP.DYN.LE00.IN", "year": 1990},
    )
    assert not is_error  # a missing year is an answer, not a failure
    assert result["figure"] is None and result["latest_year"] == 2023


def test_get_figure_miss_for_an_unknown_indicator_has_no_latest_year(server: Any) -> None:
    _, result = _call(
        server, "get_figure", {"country_code": "KEN", "indicator_code": "NOPE", "year": 2023}
    )
    assert result["figure"] is None and "latest_year" not in result


@pytest.mark.parametrize(
    "arguments",
    [
        {"country_code": "RWA", "indicator_code": "SP.POP.TOTL", "year": 2022},  # not a country
        {"country_code": "KEN", "indicator_code": "SP.POP.TOTL", "year": "recent"},
        {"country_code": "KEN", "indicator_code": "SP.POP.TOTL"},
    ],
)
def test_get_figure_rejects_bad_arguments(server: Any, arguments: dict[str, Any]) -> None:
    assert _call(server, "get_figure", arguments)[0]


# ---- list_indicators -----------------------------------------------------------------


def test_list_indicators_returns_each_code_once_with_name_and_unit(server: Any) -> None:
    is_error, result = _call(server, "list_indicators", {})
    assert not is_error
    codes = [i["indicator_code"] for i in result["indicators"]]
    assert codes == sorted(set(codes)) and "SP.POP.TOTL" in codes
    assert set(result["indicators"][0]) == {"indicator_code", "indicator_name", "unit"}


# ---- failures ------------------------------------------------------------------------


def test_a_database_failure_is_reported_as_a_tool_error_not_a_crash() -> None:
    def broken() -> Any:
        raise psycopg.OperationalError("connection refused")

    server = build_server(broken)
    is_error, _ = _call(server, "list_indicators", {})
    assert is_error


def test_a_crash_does_not_leak_its_details_to_the_client() -> None:
    def broken() -> Any:
        raise RuntimeError("password=hunter2")

    result = asyncio.run(_call_async(build_server(broken), "list_indicators", {}))
    assert result.is_error and "hunter2" not in str(result.content)


def test_each_call_opens_its_own_connection(seeded_db: psycopg.Connection) -> None:
    opened: list[int] = []

    def open_conn() -> Any:
        opened.append(1)
        return nullcontext(seeded_db)

    server = build_server(open_conn)
    _call(server, "list_indicators", {})
    _call(server, "list_indicators", {})
    assert len(opened) == 2
