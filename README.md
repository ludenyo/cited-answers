# cited-answers

A Python service that answers questions about Kenyan development indicators from the
World Bank. Every number in an answer cites the database row it came from.

See `CLAUDE.md` for the stack, folder layout and project rules.

## Quick start

```
cp .env.example .env     # then fill in ANTHROPIC_API_KEY when it is needed
make up                  # start Postgres + pgvector
make ingest              # apply migrations, fetch World Bank data, embed it
make test
```

## Why the database is on port 5435

Docker publishes Postgres on host port **5435**, not the default 5432. A Postgres
installed directly on your machine often owns 5432 and answers before Docker does,
which shows up as `role "cited" does not exist`. Ports 5433 and 5434 are used by
other local projects' databases, so 5435 is the next free one. The container still
listens on 5432 internally, and `DATABASE_URL` in `.env.example` already points at 5435.

## MCP server

`src/cited_answers/mcp_server.py` is an [MCP](https://modelcontextprotocol.io) server over
stdio. It gives any MCP client the same three tools the answering model uses, with the same
descriptions (taken from `answer.py`, not rewritten):

| Tool | Use |
|---|---|
| `search_figures(question, k?)` | semantic search over the figures; `k` is clamped to 1-20 |
| `get_figure(country_code, indicator_code, year)` | exact lookup; `country_code` is KEN, UGA or TZA |
| `list_indicators()` | every indicator code, name and unit |

Every figure comes back with `figure_id`, `source_url` and `latest_year` (the newest year the
database has for that series). A `get_figure` miss is not an error: it returns
`{"figure": null, "latest_year": ...}` so the client knows how far the series goes.

It needs only `DATABASE_URL` (no Anthropic key), so run `make up` and `make ingest` first.
The server talks over stdout, so do not run it by hand to "see output"; a client starts it.
`make mcp` is the same command the clients below run.

The command to register is the same everywhere. Use absolute paths, because clients do not
start in the project directory:

```
uv run --directory /path/to/cited-answers --env-file /path/to/cited-answers/.env \
  python -m cited_answers.mcp_server
```

### Claude Code

```
claude mcp add cited-answers -- uv run --directory /path/to/cited-answers \
  --env-file /path/to/cited-answers/.env python -m cited_answers.mcp_server
claude mcp list        # cited-answers should show as connected
```

Add `--scope user` to make it available in every project, not just the current one.

### Claude Desktop

Edit `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows:
`%APPDATA%\Claude\`), then restart Claude Desktop. The desktop app does not use your shell's
`PATH`, so give the full path to `uv` (`which uv`):

```json
{
  "mcpServers": {
    "cited-answers": {
      "command": "/full/path/to/uv",
      "args": [
        "run",
        "--directory", "/path/to/cited-answers",
        "--env-file", "/path/to/cited-answers/.env",
        "python", "-m", "cited_answers.mcp_server"
      ]
    }
  }
}
```

### Testing with the MCP Inspector

The [Inspector](https://github.com/modelcontextprotocol/inspector) is a browser UI that starts
the server and lets you call each tool by hand. It needs Node:

```
npx @modelcontextprotocol/inspector \
  uv run --directory /path/to/cited-answers --env-file /path/to/cited-answers/.env \
  python -m cited_answers.mcp_server
```

Open the URL it prints (it includes a session token), choose **Tools**, then **List Tools**.
Try `list_indicators`, then `get_figure` with `KEN`, `SP.POP.TOTL`, `2023`, and `search_figures`
with "Kenya inflation". The first search takes a few seconds while the embedding model loads.
Server logs appear in the Inspector's notifications pane; they go to stderr, never stdout.
