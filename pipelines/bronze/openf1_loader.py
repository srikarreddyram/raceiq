"""Bronze loader for OpenF1 (PRD Section 7).

Raw files live at data/raw/openf1/{session_key}/{table}.json, where
`payload` is already a flat list of records — OpenF1's JSON responses need
far less reshaping than Ergast's, just column selection and typing.

Note on `pit`: OpenF1's historical coverage of pit-stop data is sparse (see
ingestion/openf1/ingest.py, which already treats a 404 there as "no data"
rather than a failure), so this loader is written against OpenF1's
documented schema for that endpoint even where a live sample was not
available to verify against during development.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from pipelines.bronze.common import write_bronze_table

logger = logging.getLogger(__name__)


def _load_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    envelope = json.loads(path.read_text())
    return envelope["payload"], envelope["ingested_at"]


def _read_table(raw_root: Path, table: str, columns: list[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(raw_root.glob(f"openf1/*/{table}.json")):
        records, ingested_at = _load_records(path)
        if not records:
            continue
        frame = pd.DataFrame(records)[columns]
        frame["_source"] = "openf1"
        frame["_ingested_at"] = ingested_at
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_laps(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    laps = _read_table(
        raw_root,
        "laps",
        [
            "session_key",
            "meeting_key",
            "driver_number",
            "lap_number",
            "date_start",
            "duration_sector_1",
            "duration_sector_2",
            "duration_sector_3",
            "lap_duration",
            "i1_speed",
            "i2_speed",
            "st_speed",
            "is_pit_out_lap",
        ],
    )
    write_bronze_table(
        con, "openf1_laps", laps, dedup_keys=["session_key", "driver_number", "lap_number"]
    )


def load_pit(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    pit = _read_table(
        raw_root, "pit", ["session_key", "meeting_key", "driver_number", "lap_number", "date", "pit_duration"]
    )
    write_bronze_table(con, "openf1_pit", pit, dedup_keys=["session_key", "driver_number", "lap_number"])


def load_position(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    position = _read_table(
        raw_root, "position", ["session_key", "meeting_key", "driver_number", "date", "position"]
    )
    write_bronze_table(
        con, "openf1_position", position, dedup_keys=["session_key", "driver_number", "date"]
    )


def load_weather(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    weather = _read_table(
        raw_root,
        "weather",
        [
            "session_key",
            "meeting_key",
            "date",
            "air_temperature",
            "track_temperature",
            "humidity",
            "pressure",
            "rainfall",
            "wind_speed",
            "wind_direction",
        ],
    )
    write_bronze_table(con, "openf1_weather", weather, dedup_keys=["session_key", "date"])


def load_race_control(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    race_control = _read_table(
        raw_root,
        "race_control",
        [
            "session_key",
            "meeting_key",
            "date",
            "driver_number",
            "lap_number",
            "category",
            "flag",
            "scope",
            "message",
        ],
    )
    # (session_key, date, message) is the best natural key OpenF1 gives us here —
    # race control has no numeric id of its own.
    write_bronze_table(
        con, "openf1_race_control", race_control, dedup_keys=["session_key", "date", "message"]
    )


def load_all(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    load_laps(raw_root, con)
    load_pit(raw_root, con)
    load_position(raw_root, con)
    load_weather(raw_root, con)
    load_race_control(raw_root, con)
