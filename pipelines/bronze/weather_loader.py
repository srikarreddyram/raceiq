"""Bronze loader for Open-Meteo historical weather (PRD Section 7).

Raw files live at data/raw/weather/{circuit_id}/{start}_{end}.json. The
payload is Open-Meteo's "parallel arrays" shape — one `hourly.time` array
plus one array per variable, all the same length — which this loader
transposes into one row per hourly observation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from pipelines.bronze.common import write_bronze_table

logger = logging.getLogger(__name__)


def load_observations(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    frames = []
    for path in sorted(raw_root.glob("weather/*/*.json")):
        circuit_id = path.parts[-2]
        envelope = json.loads(path.read_text())
        payload, ingested_at = envelope["payload"], envelope["ingested_at"]
        hourly = payload["hourly"]

        frame = pd.DataFrame(
            {
                "circuit_id": circuit_id,
                "time": pd.to_datetime(hourly["time"]),
                "latitude": payload["latitude"],
                "longitude": payload["longitude"],
                "air_temp": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "precipitation": hourly["precipitation"],
                "rain": hourly["rain"],
                "wind_speed": hourly["wind_speed_10m"],
                "wind_direction": hourly["wind_direction_10m"],
            }
        )
        frame["_source"] = "open_meteo"
        frame["_ingested_at"] = ingested_at
        frames.append(frame)

    if not frames:
        logger.warning("No weather raw files found under %s", raw_root)
        return

    observations = pd.concat(frames, ignore_index=True)
    write_bronze_table(con, "weather_observations", observations, dedup_keys=["circuit_id", "time"])


def load_all(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    load_observations(raw_root, con)
