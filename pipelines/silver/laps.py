"""Silver Laps fact table (PRD Section 9.1) — the busiest table in Silver.

Three things happen here that don't happen in Bronze:

1. **ID normalisation**: `driver` (a code like "VER") is resolved to
   Ergast's canonical `driver_id`, scoped by season; `team` (a display
   name like "Red Bull Racing") is resolved to Ergast's `constructor_id`
   via fuzzy matching (see id_mappings.py).
2. **Pit stop duration**: FastF1 marks `PitInTime` on the lap a driver
   enters the pits and `PitOutTime` on the *next* lap (the out-lap) — the
   stop's duration is the gap between those two timestamps, which requires
   looking at adjacent rows for the same driver.
3. **Gap to leader**: derived from each driver's cumulative race time
   (running sum of lap times) compared to the minimum cumulative time at
   that lap number. This is an approximation — it compares cars at the
   same lap *count*, which is exact for lead-lap cars but not fully
   correct once a car is lapped. A precise version would use OpenF1's
   `/intervals` endpoint (not yet ingested) rather than derive it from lap
   times.
4. **Stints that are really stints**: FastF1 starts a new `Stint` every
   time a car goes through the pit lane, tyre change or not. When a race
   is neutralised and the field is routed through the pits — Canada 2025
   finished that way — every car picks up a "stint" per lap on the same
   tyres. 540 of 6,480 stint changes in 2018-2026 are like that (12% in
   2025), in 64 of 187 races, and everything that counts stops from stint
   numbers counted them: rival stop patterns, stint-length limits, the
   validations. Here a stint changes only when the tyres do (see
   `_real_stint_numbers`).
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np

from pipelines.silver.common import write_silver_table
from pipelines.silver.id_mappings import build_team_id_map

logger = logging.getLogger(__name__)


def _real_stint_numbers(laps):
    """Stint numbers that count tyre changes, not pit-lane passages.

    A FastF1 stint change is kept unless the tyres are provably the same
    set: same compound, and a tyre age that carried on rising across it.
    An unknown age or compound on either side keeps FastF1's change.
    `laps` must be sorted by lap within (race_id, driver).
    """
    by_car = laps.groupby(["race_id", "driver"], sort=False)
    prev_stint = by_car["stint_number"].shift(1)
    prev_age = by_car["tyre_age_laps"].shift(1)
    prev_compound = by_car["compound"].shift(1)
    changed = laps["stint_number"].notna() & prev_stint.notna() & (laps["stint_number"] != prev_stint)
    same_tyres = (laps["tyre_age_laps"] > prev_age) & (laps["compound"] == prev_compound)
    real_change = (changed & ~same_tyres.fillna(False)).astype("int64")
    stint = 1 + real_change.groupby([laps["race_id"], laps["driver"]], sort=False).cumsum()
    return stint.where(laps["stint_number"].notna()).astype("Int64")


def build(con: duckdb.DuckDBPyConnection) -> None:
    laps = con.execute("SELECT * FROM bronze.fastf1_laps WHERE session_type = 'R'").df()
    if laps.empty:
        return

    laps["race_id"] = laps["season"].astype(str) + "_" + laps["round"].astype(str)

    driver_lookup = con.execute(
        "SELECT DISTINCT season, driver_code, driver_id FROM bronze.ergast_results"
    ).df()
    laps = laps.merge(
        driver_lookup, left_on=["season", "driver"], right_on=["season", "driver_code"], how="left"
    )
    unmatched = laps["driver_id"].isna().sum()
    if unmatched:
        logger.warning("%s lap rows had no matching Ergast driver_id for their (season, code)", unmatched)

    team_map = build_team_id_map(con)
    laps["team_id"] = laps["team"].map(team_map)

    laps = laps.sort_values(["race_id", "driver", "lap_number"])
    laps["stint_number"] = _real_stint_numbers(laps)
    laps["_next_pit_out"] = laps.groupby(["race_id", "driver"])["pit_out_time_seconds"].shift(-1)
    laps["pit_stop_duration"] = np.where(
        laps["pit_in_time_seconds"].notna(), laps["_next_pit_out"] - laps["pit_in_time_seconds"], np.nan
    )

    laps["cumulative_time_seconds"] = laps.groupby(["race_id", "driver"])["lap_time_seconds"].cumsum()
    leader_time = laps.groupby(["race_id", "lap_number"])["cumulative_time_seconds"].transform("min")
    laps["gap_to_leader"] = laps["cumulative_time_seconds"] - leader_time

    laps["lap_id"] = laps["race_id"] + "_" + laps["driver"] + "_" + laps["lap_number"].astype(str)

    final = laps.rename(
        columns={
            "driver_id": "driver_id",
            "tyre_age_laps": "tyre_age",
            "track_status": "track_status_code",
        }
    )[
        [
            "lap_id",
            "race_id",
            "driver_id",
            "team_id",
            "lap_number",
            "lap_time_seconds",
            "position",
            "compound",
            "tyre_age",
            "stint_number",
            "is_pit_lap",
            "pit_stop_duration",
            "track_status_code",
            "gap_to_leader",
            # Carried through for car_profiles/ (PRD Section 8.1), whose
            # `downforce_proxy` characteristic is defined as sector 2
            # (high-downforce) pace delta versus sectors 1+3 (low-downforce)
            # relative to the field — it can't be derived from whole-lap
            # times, and FastF1 has had these in Bronze all along.
            "sector1_time_seconds",
            "sector2_time_seconds",
            "sector3_time_seconds",
        ]
    ]
    write_silver_table(con, "laps", final)
