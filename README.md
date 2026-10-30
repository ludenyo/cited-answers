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
