"""Shared write path for Silver builders.

Unlike Bronze (pipelines/bronze/common.py), Silver tables are derived by a
SQL query each run, not accumulated from timestamped ingestion events —
there's no `_ingested_at` to dedup on. Silver's job is to make sure that
query is already correct (a real GROUP BY / DISTINCT / window function),
then materialize the result as-is.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def write_silver_table(con: duckdb.DuckDBPyConnection, table_name: str, frame: pd.DataFrame) -> None:
    if frame.empty:
        logger.warning("No rows for silver.%s — skipping", table_name)
        return

    con.register("_incoming", frame)
    con.execute(f"CREATE OR REPLACE TABLE silver.{table_name} AS SELECT * FROM _incoming")
    con.unregister("_incoming")
    logger.info("silver.%s: %s rows", table_name, len(frame))
