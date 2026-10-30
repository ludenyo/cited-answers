CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE figures (
    id             text PRIMARY KEY,   -- deterministic: COUNTRY:INDICATOR:YEAR
    country_code   text NOT NULL,
    country_name   text NOT NULL,
    indicator_code text NOT NULL,
    indicator_name text NOT NULL,
    year           int NOT NULL,
    value          numeric NOT NULL,
    unit           text,
    source_url     text NOT NULL,
    retrieved_at   timestamptz NOT NULL DEFAULT now()
);

-- figure_id is the primary key so each figure has exactly one chunk,
-- which lets ingestion upsert chunks the same way it upserts figures.
CREATE TABLE figure_chunks (
    figure_id text PRIMARY KEY REFERENCES figures (id),
    text      text NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX figure_chunks_embedding_idx
    ON figure_chunks USING hnsw (embedding vector_cosine_ops);
