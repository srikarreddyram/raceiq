"""Read-only connection to the Gold-layer DuckDB warehouse for training scripts."""

from __future__ import annotations

import duckdb

from pipelines.bronze.db import bronze_db_path


def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(bronze_db_path()), read_only=True)
