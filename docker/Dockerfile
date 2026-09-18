# Serves the FastAPI layer only (serving/api) — not the ingestion or
# training pipelines, which are run manually via `uv run python -m ...`
# against a local checkout, not from inside a container (see PRD Section
# 15 for the not-yet-built Airflow-orchestrated retraining pipeline that
# will eventually own that job).
#
# The DuckDB warehouse, MLflow tracking DB, and MLflow artifact store are
# NOT baked into this image — they're real, large, frequently-regenerated
# local files (data/bronze/raceiq.duckdb, mlflow.db, mlartifacts/), all
# already gitignored for the same reason. They're mounted as read-only
# volumes at runtime instead (see docker-compose.yml) — this container is
# the serving code, not the data.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

# Install dependencies before copying the rest of the source so an
# application code change doesn't invalidate this (much slower — torch,
# catboost, lightgbm are large downloads) layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY pipelines/ pipelines/
COPY models/ models/
COPY strategy_engine/ strategy_engine/
COPY serving/ serving/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "serving.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
