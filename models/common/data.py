"""Shared data loading for all six models — everything reads from
`gold.race_features`, never from Bronze/Silver directly (see
pipelines/gold/race_features.py's docstring: that table is the contract
between the data pipeline and everything downstream).
"""

from __future__ import annotations

import duckdb
import pandas as pd

from models.common.db import get_connection


def load_race_features(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """The full Gold feature store, with `season`/`round` parsed out of `race_id`
    and `circuit_id` joined in from `silver.races` — every model benefits from
    knowing which circuit a lap belongs to, even before track_maps/ exists to
    describe *why* circuits differ.
    """
    owns_connection = con is None
    con = con or get_connection()
    try:
        df = con.execute(
            """
            SELECT lf.*, r.circuit_id
            FROM gold.race_features lf
            LEFT JOIN silver.races r ON r.race_id = lf.race_id
            """
        ).df()
    finally:
        if owns_connection:
            con.close()

    season_round = df["race_id"].str.split("_", n=1, expand=True)
    df["season"] = season_round[0].astype(int)
    df["round"] = season_round[1].astype(int)
    return df


def load_final_classifications(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """One row per (season, round, driver_id) -> final race position.

    Ergast's `position` is only populated for classified finishers;
    `positionText` covers DNF/DSQ/etc. as non-numeric codes, which is why
    this is a left-as-null numeric column rather than something coerced
    into a fake position.
    """
    owns_connection = con is None
    con = con or get_connection()
    try:
        return con.execute(
            "SELECT season, round, driver_id, position AS final_position "
            "FROM bronze.ergast_results"
        ).df()
    finally:
        if owns_connection:
            con.close()
