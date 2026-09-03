"""Silver Weather fact table (PRD Section 9.1).

Sourced from FastF1's own session weather samples (higher resolution and
already session-timed, unlike the hourly Open-Meteo observations in
`bronze.weather_observations`, which are a supplementary Gold-layer
cross-check rather than the primary record). `lap_number` is attached via
the shared lap timeline (see lap_timeline.py) so weather can be joined to
laps directly instead of only by raw timestamp.
"""

from __future__ import annotations

import duckdb

from pipelines.silver.common import write_silver_table
from pipelines.silver.lap_timeline import attach_lap_number, build_lap_boundaries


def build(con: duckdb.DuckDBPyConnection) -> None:
    weather = con.execute("SELECT * FROM bronze.fastf1_weather WHERE session_type = 'R'").df()
    if weather.empty:
        return

    weather["race_id"] = weather["season"].astype(str) + "_" + weather["round"].astype(str)

    boundaries = build_lap_boundaries(con)
    weather["lap_number"] = attach_lap_number(
        boundaries, weather["race_id"], weather["session_offset_seconds"]
    )

    weather["weather_id"] = weather["race_id"] + "_" + weather["session_offset_seconds"].astype(str)

    final = weather[
        [
            "weather_id",
            "race_id",
            "lap_number",
            "air_temp",
            "track_temp",
            "humidity",
            "wind_speed",
            "wind_direction",
            "rainfall",
        ]
    ]
    write_silver_table(con, "weather", final)
