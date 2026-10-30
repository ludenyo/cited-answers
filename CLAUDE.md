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
│   ├── __init__.py
│   └── api.py             # FastAPI app factory and routes
├── tests/                 # pytest suite, mirrors the package
├── docker-compose.yml     # Postgres + pgvector
├── .env.example           # required env vars; copy to .env (gitignored)
├── Makefile               # up, down, test, lint
└── pyproject.toml         # deps, ruff and pytest config
```

## Commands

```
make up     # start Postgres
make down   # stop Postgres
make test   # uv run pytest
make lint   # ruff check + ruff format --check
```

## Rules

- Type hints everywhere: every function signature has parameter and return types.
- Small functions: one job each; split rather than nest.
- No secrets in code: read keys and URLs from environment variables, never literals.
- Write a test for every new module under `tests/`.
- Explain any non-obvious decision in a short comment next to the code.
