.PHONY: up down test lint migrate ingest

up:
	docker compose up -d

down:
	docker compose down

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

# migrate and ingest read DATABASE_URL from .env (copy .env.example first)
migrate:
	uv run --env-file .env python -m cited_answers.migrate

ingest: migrate
	uv run --env-file .env python -m cited_answers.ingest_worldbank
