#!/usr/bin/env bash
# Run the promptfoo eval suite: the same steps locally (make eval) and in CI.
# Needs ANTHROPIC_API_KEY and ANTHROPIC_MODEL, and DATABASE_URL (or EVAL_DATABASE_URL).
set -euo pipefail
cd "$(dirname "$0")/.."

export LANGFUSE_TRACING_ENABLED=false
# Keeps promptfoo's own cache and database inside the project instead of in ~/.promptfoo.
export PROMPTFOO_CONFIG_DIR="$PWD/.promptfoo"
export PROMPTFOO_DISABLE_TELEMETRY=1
# The provider imports cited_answers, so promptfoo must use the project's interpreter.
export PROMPTFOO_PYTHON="$PWD/.venv/bin/python"

DATABASE_URL="$(uv run python scripts/eval_db.py)"
export DATABASE_URL

out=evals/promptfoo/output
mkdir -p "$out"
# --repeat 1: each case runs once, to keep the cost to a few cents.
# No `set -e` abort on failed cases: the gate below decides pass or fail.
npx promptfoo eval -c evals/promptfoo/promptfooconfig.yaml --repeat 1 --no-cache \
  --max-concurrency 4 --output "$out/results.json" || true
uv run python scripts/eval_gate.py "$out/results.json"
