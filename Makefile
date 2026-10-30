.PHONY: up down test lint migrate ingest load-test eval mcp

up:
	docker compose up -d

down:
	docker compose down

# Loads .env when present so the database tests see TEST_DATABASE_URL instead of skipping.
test:
	uv run $(if $(wildcard .env),--env-file .env,) pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# migrate and ingest read DATABASE_URL from .env (copy .env.example first)
migrate:
	uv run --env-file .env python -m cited_answers.migrate

ingest: migrate
	uv run --env-file .env python -m cited_answers.ingest_worldbank

# Needs the server running: uv run --env-file .env uvicorn cited_answers.api:app
# Makes about 90 real model calls (a few US cents on Haiku).
load-test:
	uv run python scripts/load_test.py

# Runs the promptfoo evals against a fixture database (cited_answers_eval), not your real data.
# One run per case, about 30 questions: a few US cents on Haiku. Needs `make up` and `npm ci`.
eval:
	set -a; . ./.env; set +a; ./scripts/run_eval.sh

# MCP server over stdio, for MCP clients (see README). Reads DATABASE_URL from .env.
# Not meant to be run by hand: an MCP client starts it. Use the Inspector to try it.
mcp:
	uv run --env-file .env python -m cited_answers.mcp_server
