"""Connection helper for the Silver layer.

Silver lives in the same DuckDB file as Bronze (see pipelines/bronze/db.py),
just under a different schema. One physical database keeps every layer
joinable with plain SQL and is exactly what dbt will point at once Silver/
Gold move to dbt models — this hand-written version is the reference
implementation that dbt will later replace.
"""

from __future__ import annotations

import duckdb

from pipelines.bronze.db import get_connection as get_bronze_connection


def get_connection() -> duckdb.DuckDBPyConnection:
    con = get_bronze_connection()
    con.execute("CREATE SCHEMA IF NOT EXISTS silver")
    return con
