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
│   └── verify.py          # checks every claim against its cited figure
├── prompts/answer_system.md  # system prompt, versioned on its own
├── migrations/            # numbered SQL files
├── config/indicators.yaml # ingestion: countries, indicators, start year, unit overrides
├── config/app.yaml        # retrieval, llm loop, verification tolerances, prompt file
├── evals/questions_draft.yaml  # 18 draft eval questions (easy, compare, vague, trap), not yet run
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
make test     # pytest, loading .env so the DB tests run (they skip without TEST_DATABASE_URL)
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

## Rules

- Type hints everywhere: every function signature has parameter and return types.
- Small functions: one job each; split rather than nest.
- No secrets in code: read keys and URLs from environment variables, never literals.
- Write a test for every new module under `tests/`.
- Explain any non-obvious decision in a short comment next to the code.
