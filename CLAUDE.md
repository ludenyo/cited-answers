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
│   ├── tracing.py         # Langfuse wrapper: spans, scores, masking; never raises
│   ├── trace_format.py    # Anthropic messages -> the role format Langfuse renders
│   └── verify.py          # checks every claim against its cited figure
├── prompts/answer_system.md  # system prompt, versioned on its own
├── migrations/            # numbered SQL files
├── config/indicators.yaml # ingestion: countries, indicators, start year, unit overrides
├── config/app.yaml        # retrieval, llm loop, verification tolerances, prompt file
├── scripts/export_eval_fixture.py  # full DB -> tests/fixtures/eval_figures.sql (needs real data)
├── scripts/eval_db.py     # migrate + load the fixture into the eval database
├── scripts/eval_gate.py   # exit 1 when the eval pass rate is below the threshold
├── scripts/run_eval.sh    # the eval run, shared by make eval and CI
├── scripts/load_test.py   # sends every eval question 5 times to a running /ask
├── evals/questions.yaml  # 30 eval questions (easy, compare, vague, latest, trap, guardrail)
├── evals/promptfoo/      # promptfoo suite: provider, assertions, case generator, rubric grader
├── .github/workflows/evals.yml  # runs the suite on pull requests, fails below 90%
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
make eval       # promptfoo suite, 1 run per case, against the fixture database (cited_answers_eval)
make load-test  # 30 eval questions x 5 against a running server (150 requests; refused ones make no answering call)
```

## Evals (promptfoo)

`evals/questions.yaml` is the only source of cases and expected values; the promptfoo config
builds its tests from it. Each case runs once through `answer.ask()` directly (no server) and
must pass: every claim re-verified by `verify.py`, `insufficient_data`, `refused` and
`guardrails_include`, the latest-year check, latency under 20 s and, for non-refused cases, an
`llm-rubric`. A case passes only when all of them do; `scripts/eval_gate.py` fails below 90%.

- **Fixture database.** CI never calls the World Bank API. It loads
  `tests/fixtures/eval_figures.sql`: the full series (all years) for every figure the cases expect,
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
