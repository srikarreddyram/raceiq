"""Silver PitStops fact table (PRD Section 9.1).

Built from `silver.laps` (must run after `laps.build`): a pit stop is any
lap flagged `is_pit_lap`, with `compound_in` being the compound run into
the pits on that lap and `compound_out` the compound fitted for the next
lap (via `LEAD` over each driver's laps, ordered by lap number).
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table


def build(con: duckdb.DuckDBPyConnection) -> None:
    pit_stops = con.execute(
        """
        WITH with_next_compound AS (
            SELECT
                race_id,
                driver_id,
                lap_number,
                compound AS compound_in,
                LEAD(compound) OVER (
                    PARTITION BY race_id, driver_id ORDER BY lap_number
                ) AS compound_out,
                pit_stop_duration AS stop_duration_seconds
            FROM silver.laps
            WHERE is_pit_lap
        )
        SELECT
            race_id || '_' || driver_id || '_' || lap_number AS pitstop_id,
            race_id,
            driver_id,
            lap_number,
            compound_in,
            compound_out,
            stop_duration_seconds
        FROM with_next_compound
        """
    ).df()
    write_silver_table(con, "pit_stops", pit_stops)
