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


def is_classified(status) -> bool:
    """Whether a driver was classified in a race, from Ergast's `status`:
    finishers, and lapped finishers ("+1 Lap", or "Lapped" in the 2026
    feed). Everything else — Retired, Accident, Disqualified, Did not
    start — wasn't."""
    return isinstance(status, str) and (status == "Finished" or status == "Lapped" or status.startswith("+"))


def load_final_classifications(con: duckdb.DuckDBPyConnection | None = None) -> pd.DataFrame:
    """One row per (season, round, driver_id): `final_position` and
    `classified`.

    `final_position` is Ergast's `position`, which in this data is the
    finishing ORDER and is populated for every starter, retirements
    included (a lap-3 retirement reads P22). An earlier version of this
    docstring said non-classified drivers had a null position; they
    don't, and the Final Race Position model was trained on retirement
    order because of it. `classified` says which rows are real finishing
    positions — see models/race_position/train.py for how it's used.
    """
    owns_connection = con is None
    con = con or get_connection()
    try:
        df = con.execute(
            "SELECT season, round, driver_id, position AS final_position, status "
            "FROM bronze.ergast_results"
        ).df()
    finally:
        if owns_connection:
            con.close()
    df["classified"] = df["status"].map(is_classified)
    return df.drop(columns="status")
