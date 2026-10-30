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
│   ├── settings.py        # ANTHROPIC_MODEL / ANTHROPIC_MAX_TOKENS from env
│   ├── retrieval.py       # search_figures (pgvector cosine), get_figure, get_figure_by_id
│   ├── answer.py          # tool loop; the answer comes back via a submit_answer tool
│   └── verify.py          # checks every claim against its cited figure
├── prompts/answer_system.md  # system prompt, versioned on its own
├── migrations/            # numbered SQL files
├── config/indicators.yaml # countries, indicators, start year, unit overrides
├── tests/                 # pytest suite, mirrors the package
├── docker-compose.yml     # Postgres + pgvector
├── .env.example           # required env vars; copy to .env (gitignored)
├── Makefile               # up, down, test, lint, migrate, ingest
└── pyproject.toml         # deps, ruff and pytest config
```

## Commands

```
make up       # start Postgres
make down     # stop Postgres
make test     # uv run pytest (DB tests skip unless TEST_DATABASE_URL is set)
make lint     # ruff check + ruff format --check
make migrate  # apply SQL migrations
make ingest   # migrate, then fetch World Bank data and embed it
```

## Database port

The Docker database is published on host port **5435**, not the Postgres default
5432. A Postgres installed directly on the developer's machine commonly owns 5432
and answers before Docker does, which looks like "role does not exist" errors.
5433 and 5434 are taken by other local projects' databases, so 5435 is the next free one.
`DATABASE_URL` and `TEST_DATABASE_URL` in `.env.example` use 5435 to match.

## Rules

- Type hints everywhere: every function signature has parameter and return types.
- Small functions: one job each; split rather than nest.
- No secrets in code: read keys and URLs from environment variables, never literals.
- Write a test for every new module under `tests/`.
- Explain any non-obvious decision in a short comment next to the code.
