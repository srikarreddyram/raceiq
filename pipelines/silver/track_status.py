"""Silver TrackStatus fact table (PRD Section 9.1).

FastF1 emits one row per track-status *change* (a point event), not a
span — `duration_laps` has to be derived by pairing each event with the
next one for the same race. `lap_number` itself isn't in the raw data
either; it's attached via the shared lap timeline (see lap_timeline.py).
The final status of a race is left with a null `duration_laps` since there
is no "next" event to bound it.
"""

from __future__ import annotations

import logging

import duckdb

from pipelines.silver.common import write_silver_table
from pipelines.silver.lap_timeline import attach_lap_number, build_lap_boundaries

logger = logging.getLogger(__name__)

STATUS_CODE_MAP = {
    "1": "Clear",
    "2": "Yellow",
    "4": "SafetyCar",
    "5": "Red",
    "6": "VSC",
    "7": "VSCEnding",
}


def build(con: duckdb.DuckDBPyConnection) -> None:
    status = con.execute(
        "SELECT * FROM bronze.fastf1_track_status WHERE session_type = 'R'"
    ).df()
    if status.empty:
        return

    status["race_id"] = status["season"].astype(str) + "_" + status["round"].astype(str)
    status["status_type"] = status["status_code"].astype(str).map(STATUS_CODE_MAP)
    unknown = status["status_type"].isna()
    if unknown.any():
        logger.warning(
            "Unrecognised track status codes: %s", sorted(status.loc[unknown, "status_code"].unique())
        )
        status.loc[unknown, "status_type"] = "Unknown"

    boundaries = build_lap_boundaries(con)
    status["lap_number"] = attach_lap_number(boundaries, status["race_id"], status["session_offset_seconds"])

    status = status.sort_values(["race_id", "session_offset_seconds"])
    status["duration_laps"] = status.groupby("race_id")["lap_number"].shift(-1) - status["lap_number"]

    status["status_id"] = (
        status["race_id"] + "_" + status["session_offset_seconds"].astype(str)
    )

    final = status[["status_id", "race_id", "lap_number", "status_type", "duration_laps"]]
    write_silver_table(con, "track_status", final)
