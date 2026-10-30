# cited-answers

## Goal

A Python service that answers natural-language questions about Kenyan development
indicators from the World Bank. Every number in an answer cites the database row it
came from, so a reader can verify each figure against its source.

## Stack

- Python 3.12, managed with `uv` (`.python-version` pins the interpreter)
- FastAPI + uvicorn for the HTTP API
- PostgreSQL 16 with pgvector, run via `docker-compose.yml`
- Anthropic API for answer generation
- Langfuse for tracing
- pytest for tests, ruff for lint and formatting

## Folder layout

```
cited-answers/
├── src/cited_answers/     # application package (src layout)
│   ├── api.py             # FastAPI app factory and routes
│   ├── db.py              # Postgres connection from DATABASE_URL
│   ├── migrate.py         # applies migrations/*.sql once each
│   ├── figures.py         # Figure record, id, unit and sentence helpers
│   ├── worldbank.py       # World Bank API v2 client and record parsing
│   ├── embeddings.py      # bge-small-en-v1.5 batch embedding (384 dims)
│   ├── ingest_worldbank.py  # fetch, upsert and embed (idempotent)
│   ├── models.py          # Claim, Answer, VerifiedClaim, AskResponse, FigureResult
│   ├── settings.py        # env settings + typed config/app.yaml
│   ├── retrieval.py       # search_figures (pgvector cosine), get_figure, get_figure_by_id, list_indicators
│   ├── answer.py          # tool loop; the answer comes back via a submit_answer tool
│   ├── mcp_server.py      # MCP server (stdio): the three data tools, built on answer.run_tool
│   ├── tracing.py         # Langfuse wrapper: spans, scores, masking; never raises
│   ├── trace_format.py    # Anthropic messages -> the role format Langfuse renders
│   └── verify.py          # checks every claim against its cited figure
├── prompts/answer_system.md  # system prompt, versioned on its own
├── migrations/            # numbered SQL files
├── config/indicators.yaml # ingestion: countries, indicators, start year, unit overrides
├── config/app.yaml        # retrieval, llm loop, verification tolerances, prompt file
├── scripts/export_eval_fixture.py  # full DB -> tests/fixtures/eval_figures.sql (needs real data)
├── scripts/eval_db.py     # migrate + load the fixture into the eval database
├── scripts/eval_gate.py   # exit 1 when any deterministic eval assertion fails
├── scripts/run_eval.sh    # the eval run, shared by make eval and CI
├── scripts/load_test.py   # sends every eval question 5 times to a running /ask
├── evals/questions.yaml  # 30 eval questions (easy, compare, vague, latest, trap, guardrail)
├── evals/promptfoo/      # promptfoo suite: provider, assertions, case generator, rubric grader
├── .github/workflows/evals.yml  # runs the suite on pull requests; fails on any hard gate
├── tests/                 # pytest suite, mirrors the package
├── docker-compose.yml     # Postgres + pgvector
├── .env.example           # required env vars; copy to .env (gitignored)
├── Makefile               # the commands below
└── pyproject.toml         # deps, ruff and pytest config
```

## Commands

First time: `cp .env.example .env` and fill in `ANTHROPIC_API_KEY`, then `make up` and
`make ingest`.

```
make up       # start Postgres
make down     # stop Postgres
make test     # pytest, loading .env so the DB tests run (they skip without TEST_DATABASE_URL)
make lint     # ruff check + ruff format --check
make migrate  # apply SQL migrations
make ingest   # migrate, then fetch World Bank data and embed it
make mcp        # MCP server over stdio (clients start it; see MCP server)
make eval       # promptfoo suite, 1 run per case, against the fixture database (cited_answers_eval)
make load-test  # 30 eval questions x 5 against a running server (150 requests; refused ones make no answering call)
```

## Evals (promptfoo)

`evals/questions.yaml` is the only source of cases and expected values; the promptfoo config
builds its tests from it. Each case runs once through `answer.ask()` directly (no server) and
must pass: every claim re-verified by `verify.py`, `insufficient_data`, `refused` and
`guardrails_include`, the latest-year check, latency under 20 s and, for non-refused cases, an
`llm-rubric`. `scripts/eval_gate.py` makes the deterministic ones hard gates (claims, flag,
latest year, refusal and guardrails) that must pass in every run of every case. The rubric,
latency and, for cases marked `guardrails_report_only` (a retry that fires only sometimes), the
guardrails that must fire are report-only. `EVAL_REPEAT=3 make eval` runs each case three times.
The workflow runs on pull requests and on pushes to `main` that touch `src/`, `prompts/`,
`config/` or `evals/`, and can be started by hand (`workflow_dispatch`).

- **Fixture database.** CI never calls the World Bank API. It loads
  `tests/fixtures/eval_figures.sql`: the full series (all years) for every figure the cases expect,
  plus any series a case lists under `requires_series`,
  with stored embeddings. Regenerate with `scripts/export_eval_fixture.py` against a fully
  ingested DB whenever the cases change. Fewer distractors than production, so retrieval is
  easier than the real thing. `make eval` uses a separate `<db>_eval` database, never your data.
- **The rubric grader** (`grader.py`) calls the SDK directly and sends no temperature, which
  Haiku 5.5 rejects. It sees the answer plus one `cited` field per claim. It also flags numbers in
  the answer text that have no claim, which `verify.py` does not parse.
- Needs `npm ci` (promptfoo is pinned in `package.json`) and `ANTHROPIC_API_KEY`. In CI the key
  comes from the repository secret `ANTHROPIC_API_KEY`; pull requests from forks get no secrets.

## Database port

The Docker database is published on host port **5435**, not the Postgres default
5432. A Postgres installed directly on the developer's machine commonly owns 5432
and answers before Docker does, which looks like "role does not exist" errors.
5433 and 5434 are taken by other local projects' databases, so 5435 is the next free one.
`DATABASE_URL` and `TEST_DATABASE_URL` in `.env.example` use 5435 to match.

## Model configuration

- `ANTHROPIC_MODEL` is required and has no default. If it is missing the app refuses to
  start with a clear error. `ANTHROPIC_MAX_TOKENS` defaults to 2048 because thinking
  tokens count against it.
- `config/app.yaml` has no `temperature` setting on purpose. Haiku 5.5 rejects any
  non-default `temperature` (and `top_p` / `top_k`) with a 400, and the Anthropic SDK 2.x
  has no such parameter. Output is steered from the prompt instead.
- `search_figures` is the one tool without `strict: true`, because its optional `k`
  input is not confirmed to work under strict schemas. The code clamps `k` to 20.

## Claims

Every number in an answer is a claim. A `figure` claim cites one stored figure and is
checked against its value. A `derived` claim is computed from two figures listed as
[from, to], and `verify.py` recomputes it from the stored values:

- `difference` = to - from
- `percent_change` = (to - from) / from * 100
- `ratio` = from / to

The sign follows the formula, so a fall is negative; a claim of +87.9 for a fall of
87.9 fails on purpose. Both kinds use the tolerances in `config/app.yaml`. All `Claim`
fields are required (null or empty when unused) so the strict `submit_answer` schema
works. Text numbers are not parsed: only the claims list is verified.

## Tracing (Langfuse)

Each `/ask` request is one trace named `answer-question`, tagged `prompt:<hash>` (the
first 8 characters of the SHA-256 of `prompts/answer_system.md`) so prompt versions can
be compared. Spans, all under the root `agent` span:

| Span | Type | What it holds |
|---|---|---|
| `call-model` (one per model call) | generation | model, input and output tokens, the full message history, tool calls |
| `search-figures`, `get-figure`, `list-indicators` | retriever | the lookup and what it returned; a miss is `{"found": false}` |
| `embed-question` (inside `search-figures`) | embedding | the embedding model and vector size |
| `verify-claims` | evaluator | each claim and whether it passed |

Two scores go on every trace: `citation_pass_rate` (numeric) and `insufficient_data`
(boolean). The environment comes from `LANGFUSE_TRACING_ENVIRONMENT`.

- **Tracing never breaks a request.** Every SDK call goes through `tracing.py`, which
  logs and carries on if Langfuse is down or the SDK fails. Export runs on a background
  thread. Errors raised by the traced code itself still propagate unchanged.
- **Off by default in tests.** `tests/conftest.py` forces `LANGFUSE_TRACING_ENABLED=false`
  because `make test` loads `.env`, which holds real keys. Tests that need tracing build
  a client with an in-memory exporter via `tracing.create_client`.
- **The Anthropic SDK 2.x emits its own `anthropic.messages.create` span.** It is dropped
  at export (`tracing.should_export_span`) because `call-model` already records the call
  with the messages; keeping both would double every token and cost total.
- **Masking:** API keys and email addresses are redacted from input and output. The SDK's
  mask does not cover metadata, so never put user text or secrets in metadata.
- Not done: prompts are files in this repo, not Langfuse-managed, so traces are not linked
  to a Langfuse prompt object. There is no `user_id` (no auth) or `session_id` (single
  turn). Thinking text is omitted by default on Haiku 5.5, so generations carry none.

## Rules

- Type hints everywhere: every function signature has parameter and return types.
- Small functions: one job each; split rather than nest.
- No secrets in code: read keys and URLs from environment variables, never literals.
- Write a test for every new module under `tests/`.
- Explain any non-obvious decision in a short comment next to the code.

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
`make mcp` runs the same command the clients below use. Give them absolute paths, because
clients do not start in the project directory.

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
