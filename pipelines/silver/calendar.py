"""Silver Calendar — every round of every season, run or not.

silver.races only holds races with results (it's built from FastF1 race
sessions), which is right for everything that analyses a race. The race
weekend planner also has to plan races that haven't happened, so it reads
this instead. `race_id` follows silver.races' `{season}_{round}`, and
`has_results` says whether the round has been run and ingested.
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table


def build(con: duckdb.DuckDBPyConnection) -> None:
    tables = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'bronze'"
    ).fetchall()}
    if "ergast_schedule" not in tables:
        return
    calendar = con.execute(
        """
        SELECT s.season || '_' || s.round AS race_id, s.season, s.round, s.race_name AS name,
               s.circuit_id, s.circuit_name, s.locality, s.country, s.latitude, s.longitude,
               CAST(s.date AS DATE) AS date, s.time_utc,
               r.race_id IS NOT NULL AS has_results
        FROM bronze.ergast_schedule s
        LEFT JOIN silver.races r ON r.race_id = s.season || '_' || s.round
        """
    ).df()
    write_silver_table(con, "calendar", calendar)
