"""Bronze loader for FastF1 (PRD Section 7 — schema enforcement, typing,
deduplication on the Raw Layer's FastF1 output).

Every raw file lives at data/raw/fastf1/{season}/{round}/{session_type}/{table}.parquet
(see ingestion/fastf1/ingest.py). `season`/`round` are re-derived from the
path here rather than trusted from in-file columns — the path is a
structural guarantee, the columns are not. This is also where FastF1's
Timedelta columns (LapTime, sector times, weather sample offsets) become
plain float seconds, since DuckDB/dbt work with those far more naturally
than pandas Timedeltas.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from pipelines.bronze.common import write_bronze_table

logger = logging.getLogger(__name__)


def _read_all(raw_root: Path, table: str) -> pd.DataFrame:
    """Read every raw file for one FastF1 table, tagging season/round/session_type from its path."""
    frames = []
    for path in sorted(raw_root.glob(f"fastf1/*/*/*/{table}.parquet")):
        season, round_number, session_type = path.parts[-4], path.parts[-3], path.parts[-2]
        frame = pd.read_parquet(path)
        frame["season"] = int(season)
        frame["round"] = int(round_number)
        frame["session_type"] = session_type
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _seconds(series: pd.Series) -> pd.Series:
    """Convert a pandas Timedelta column to float seconds, preserving NaT as NaN."""
    return series.dt.total_seconds()


def load_laps(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    laps = _read_all(raw_root, "laps")
    if laps.empty:
        return

    bronze = pd.DataFrame(
        {
            "season": laps["season"],
            "round": laps["round"],
            "session_type": laps["session_type"],
            "driver": laps["Driver"],
            "driver_number": laps["DriverNumber"],
            "team": laps["Team"],
            "lap_number": laps["LapNumber"].astype("Int64"),
            "lap_time_seconds": _seconds(laps["LapTime"]),
            "sector1_time_seconds": _seconds(laps["Sector1Time"]),
            "sector2_time_seconds": _seconds(laps["Sector2Time"]),
            "sector3_time_seconds": _seconds(laps["Sector3Time"]),
            "compound": laps["Compound"],
            "tyre_age_laps": laps["TyreLife"],
            "stint_number": laps["Stint"].astype("Int64"),
            "is_fresh_tyre": laps["FreshTyre"],
            "is_pit_lap": laps["PitInTime"].notna() | laps["PitOutTime"].notna(),
            "track_status": laps["TrackStatus"],
            "position": laps["Position"],
            "is_accurate": laps["IsAccurate"],
            "_source": "fastf1",
            "_ingested_at": laps["_ingested_at"],
        }
    )
    write_bronze_table(
        con, "fastf1_laps", bronze, dedup_keys=["season", "round", "session_type", "driver", "lap_number"]
    )


def load_weather(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    weather = _read_all(raw_root, "weather")
    if weather.empty:
        return

    bronze = pd.DataFrame(
        {
            "season": weather["season"],
            "round": weather["round"],
            "session_type": weather["session_type"],
            "session_offset_seconds": _seconds(weather["Time"]),
            "air_temp": weather["AirTemp"],
            "track_temp": weather["TrackTemp"],
            "humidity": weather["Humidity"],
            "pressure": weather["Pressure"],
            "rainfall": weather["Rainfall"],
            "wind_direction": weather["WindDirection"],
            "wind_speed": weather["WindSpeed"],
            "_source": "fastf1",
            "_ingested_at": weather["_ingested_at"],
        }
    )
    write_bronze_table(
        con,
        "fastf1_weather",
        bronze,
        dedup_keys=["season", "round", "session_type", "session_offset_seconds"],
    )


def load_track_status(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    status = _read_all(raw_root, "track_status")
    if status.empty:
        return

    bronze = pd.DataFrame(
        {
            "season": status["season"],
            "round": status["round"],
            "session_type": status["session_type"],
            "session_offset_seconds": _seconds(status["Time"]),
            "status_code": status["Status"],
            "message": status["Message"],
            "_source": "fastf1",
            "_ingested_at": status["_ingested_at"],
        }
    )
    write_bronze_table(
        con,
        "fastf1_track_status",
        bronze,
        dedup_keys=["season", "round", "session_type", "session_offset_seconds", "status_code"],
    )


def load_results(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    results = _read_all(raw_root, "results")
    if results.empty:
        return

    bronze = pd.DataFrame(
        {
            "season": results["season"],
            "round": results["round"],
            "session_type": results["session_type"],
            "driver_number": results["DriverNumber"],
            "driver_code": results["Abbreviation"],
            "full_name": results["FullName"],
            "team_name": results["TeamName"],
            "grid_position": results["GridPosition"],
            "classified_position": results["ClassifiedPosition"],
            "status": results["Status"],
            "points": results["Points"],
            "laps_completed": results["Laps"],
            "_source": "fastf1",
            "_ingested_at": results["_ingested_at"],
        }
    )
    write_bronze_table(
        con,
        "fastf1_results",
        bronze,
        dedup_keys=["season", "round", "session_type", "driver_number"],
    )


def load_race_meta(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    meta = _read_all(raw_root, "race_meta")
    if meta.empty:
        return

    bronze = pd.DataFrame(
        {
            "season": meta["season"],
            "round": meta["round"],
            "session_type": meta["session_type"],
            "circuit_id": meta["circuit_id"],
            "name": meta["name"],
            "date": meta["date"],
            "_source": "fastf1",
            "_ingested_at": meta["_ingested_at"],
        }
    )
    write_bronze_table(con, "fastf1_race_meta", bronze, dedup_keys=["season", "round", "session_type"])


def load_all(raw_root: Path, con: duckdb.DuckDBPyConnection) -> None:
    load_race_meta(raw_root, con)
    load_laps(raw_root, con)
    load_weather(raw_root, con)
    load_track_status(raw_root, con)
    load_results(raw_root, con)
