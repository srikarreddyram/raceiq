"""Silver Constructors dimension (PRD Section 9.1).

Ergast's constructor_id is already the canonical id used everywhere else
in Silver (see id_mappings.py) — this table is just Ergast's constructor
standings collapsed to one row per (constructor, season).
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table


def build(con: duckdb.DuckDBPyConnection) -> None:
    constructors = con.execute(
        """
        SELECT DISTINCT
            constructor_id,
            constructor_name AS name,
            season
        FROM bronze.ergast_constructor_standings
        """
    ).df()
    write_silver_table(con, "constructors", constructors)
