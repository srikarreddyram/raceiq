"""Connection helper for the Bronze-layer DuckDB warehouse.

Bronze is stored as a single local DuckDB file rather than more raw files,
because from this point on every layer (Bronze -> Silver -> Gold) needs to
be queried and joined with SQL. DuckDB gives us that for free, locally,
with zero infrastructure — dbt will point its `silver`/`gold` models at
this same file (see dbt/profiles.yml once that's set up).
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def bronze_db_path() -> Path:
    return Path(os.environ.get("RACEIQ_BRONZE_DB", REPO_ROOT / "data" / "bronze" / "raceiq.duckdb"))


def get_connection() -> duckdb.DuckDBPyConnection:
    db_path = bronze_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS bronze")
    return con
