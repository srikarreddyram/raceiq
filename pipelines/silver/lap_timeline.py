"""Maps a race-session timestamp (seconds since session start) to the lap
number in progress at that moment.

Track status events and weather samples are race-wide, not per-driver, but
FastF1 only gives lap-start timestamps per driver. This builds one
canonical lap timeline per race by taking the median lap-start time across
all drivers for each lap number — robust to any single driver's pit stops
or retirement — then locates events against it with a binary search. This
is what lets Silver attach `lap_number` to weather samples and compute
`duration_laps` for track-status events (PRD Section 9.1), which the raw
per-event timestamps alone don't give us.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd


def build_lap_boundaries(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per (race_id, lap_number) -> the median lap-start time across drivers."""
    laps = con.execute(
        """
        SELECT season, round, lap_number, lap_start_time_seconds
        FROM bronze.fastf1_laps
        WHERE session_type = 'R' AND lap_start_time_seconds IS NOT NULL
        """
    ).df()
    laps["race_id"] = laps["season"].astype(str) + "_" + laps["round"].astype(str)

    boundaries = (
        laps.groupby(["race_id", "lap_number"])["lap_start_time_seconds"]
        .median()
        .reset_index()
        .sort_values(["race_id", "lap_number"])
    )
    return boundaries


def attach_lap_number(
    boundaries: pd.DataFrame, race_ids: pd.Series, offsets_seconds: pd.Series
) -> pd.Series:
    """For each (race_id, offset) pair, the lap number whose boundary it falls on or after.

    Returns 0 for anything before lap 1 started (formation lap / grid).
    """
    result = pd.Series(index=offsets_seconds.index, dtype="Int64")

    for race_id, group_index in pd.Series(race_ids.values, index=race_ids.index).groupby(race_ids).groups.items():
        race_boundaries = boundaries[boundaries["race_id"] == race_id].sort_values("lap_number")
        if race_boundaries.empty:
            continue

        boundary_values = race_boundaries["lap_start_time_seconds"].to_numpy()
        lap_numbers = race_boundaries["lap_number"].to_numpy()

        group_offsets = offsets_seconds.loc[group_index].to_numpy()
        idx = np.searchsorted(boundary_values, group_offsets, side="right") - 1
        mapped = np.where(idx < 0, 0, lap_numbers[np.clip(idx, 0, len(lap_numbers) - 1)])
        result.loc[group_index] = mapped

    return result
