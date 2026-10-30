"""MCP server (stdio) exposing the three data tools, so any MCP client can query the figures.

Run with: uv run --env-file .env python -m cited_answers.mcp_server  (or: make mcp)

The tools are the ones the answering model uses: descriptions come from answer.TOOLS and the
lookups go through answer.run_tool, so both sides always return the same figures with the same
fields (figure_id, source_url, latest_year). Over stdio, stdout carries the protocol itself,
so nothing in here may print to it; logging goes to stderr.

MCPServer is the official Python SDK's FastMCP-style server; mcp 2.x renamed FastMCP to it.
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, Literal

import psycopg
from mcp.server.mcpserver import MCPServer

from cited_answers import retrieval
from cited_answers.answer import TOOLS, run_tool
from cited_answers.db import connect

ConnectFn = Callable[[], AbstractContextManager[psycopg.Connection]]


def tool_description(name: str) -> str:
    """The description the answering model sees for this tool; not rewritten for MCP."""
    return next(tool["description"] for tool in TOOLS if tool["name"] == name)


def build_server(
    open_conn: ConnectFn = connect, embed: retrieval.EmbedFn | None = None
) -> MCPServer:
    """Both arguments exist so tests can use a throwaway database and a fake embedder."""
    server = MCPServer("cited-answers")

    def run(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        # One short-lived connection per call, like the HTTP API; a stdio server may sit idle
        # for hours, and a held connection would not survive a database restart.
        with open_conn() as conn:
            result: dict[str, Any] = json.loads(run_tool(conn, name, tool_input, embed))
        return result

    @server.tool(name="search_figures", description=tool_description("search_figures"))
    def search_figures(question: str, k: int | None = None) -> dict[str, Any]:
        tool_input: dict[str, Any] = {"question": question}
        if k is not None:
            tool_input["k"] = k  # run_tool clamps it to 1..20
        return run("search_figures", tool_input)

    @server.tool(name="get_figure", description=tool_description("get_figure"))
    def get_figure(
        country_code: Literal["KEN", "UGA", "TZA"], indicator_code: str, year: int
    ) -> dict[str, Any]:
        return run(
            "get_figure",
            {"country_code": country_code, "indicator_code": indicator_code, "year": year},
        )

    @server.tool(name="list_indicators", description=tool_description("list_indicators"))
    def list_indicators() -> dict[str, Any]:
        return run("list_indicators", {})

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
