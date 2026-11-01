# Demo image for Cloud Run. Build for linux/amd64 (Cloud Build does this; on Apple silicon
# pass --platform linux/amd64).

# ---- build: dependencies and the embedding model ------------------------------------------
FROM python:3.12-slim AS builder
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencies first so code changes do not reinstall them. pyproject.toml pins torch to the
# CPU-only index on Linux, which keeps the image about 2 GB smaller than the CUDA wheels.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev

# Bake the embedding model in, so a cold start does not download it.
ENV HF_HOME=/opt/hf
RUN .venv/bin/python -c "from cited_answers.embeddings import warm_up; warm_up()"

# ---- run ---------------------------------------------------------------------------------
FROM python:3.12-slim
RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /opt/hf /opt/hf
# The package is installed in editable mode, so it must stay at /app/src. It finds
# config/ and prompts/ relative to itself.
COPY src ./src
COPY config ./config
COPY prompts ./prompts
COPY migrations ./migrations

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    WARM_EMBEDDING_MODEL=true \
    TRUST_PROXY_HOPS=1 \
    RATE_LIMIT_REQUESTS=10 \
    RATE_LIMIT_WINDOW_SECONDS=3600 \
    DAILY_ASK_CAP=300
# Secrets and the model name are not baked in: ANTHROPIC_API_KEY, DATABASE_URL and
# ANTHROPIC_MODEL come from the environment. Langfuse stays off unless its keys are set.

USER app
EXPOSE 8080
# Cloud Run sets $PORT; 8080 is the fallback for running the image by hand.
CMD ["sh", "-c", "exec uvicorn cited_answers.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
