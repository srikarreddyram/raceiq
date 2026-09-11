"""Connection helper for the Gold layer — same DuckDB file, `gold` schema."""

from __future__ import annotations

import duckdb

from pipelines.bronze.db import get_connection as get_bronze_connection


def get_connection() -> duckdb.DuckDBPyConnection:
    con = get_bronze_connection()
    con.execute("CREATE SCHEMA IF NOT EXISTS gold")
    return con
