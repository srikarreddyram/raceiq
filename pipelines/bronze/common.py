"""Shared write path for every Bronze loader.

Encapsulates the two things every source needs after it has parsed its own
raw files into a DataFrame: deduplicate on its natural key, then materialize
into the Bronze schema. Bronze rebuilds each table from all of Raw on every
run (`CREATE OR REPLACE`) rather than appending — Raw is the source of
truth and is cheap to re-read, so there is no incremental state to manage
or get wrong here.
"""

from __future__ import annotations

import logging

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def write_bronze_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    frame: pd.DataFrame,
    dedup_keys: list[str],
) -> None:
    if frame.empty:
        logger.warning("No rows for bronze.%s — skipping", table_name)
        return

    before = len(frame)
    frame = (
        frame.sort_values("_ingested_at")
        .drop_duplicates(subset=dedup_keys, keep="last")
        .reset_index(drop=True)
    )
    if len(frame) < before:
        logger.info("bronze.%s: dropped %s duplicate rows on %s", table_name, before - len(frame), dedup_keys)

    con.register("_incoming", frame)
    con.execute(f"CREATE OR REPLACE TABLE bronze.{table_name} AS SELECT * FROM _incoming")
    con.unregister("_incoming")
    logger.info("bronze.%s: %s rows", table_name, len(frame))
