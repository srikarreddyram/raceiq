"""Session and telemetry extraction from FastF1 (PRD Section 6.1).

This module only knows how to pull data OUT of the FastF1 Python API and
shape it into the DataFrames described by the Laps / Telemetry / Weather /
TrackStatus / Races tables in Section 7.1. It does not know about our
storage layout — see `ingest.py` for that.
"""

from __future__ import annotations

from pathlib import Path

import fastf1
import pandas as pd

_cache_enabled = False


def enable_cache(cache_dir: Path) -> None:
    """Enable FastF1's own response cache exactly once per process.

    This is distinct from our Raw Layer: FastF1 caches the unparsed API/
    telemetry payloads it fetches internally so re-running ingestion for a
    session already on disk doesn't re-hit the FIA/timing endpoints.
    """
    global _cache_enabled
    if not _cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
        _cache_enabled = True


def load_session(
    season: int, round_number: int, session_type: str = "R", with_telemetry: bool = False
) -> fastf1.core.Session:
    """Load one session (e.g. 2023 round 1 Race) with laps/weather/track status.

    FastF1's own `telemetry` flag controls whether it fetches car/position
    data (speed, throttle, brake, GPS, ...) from the API at all — separate
    from our `extract_telemetry`, which only shapes that data once it's
    already loaded. Every call to the FIA data feed counts against FastF1's
    own rate limit (500/hour), so when nothing downstream needs telemetry
    (the default), skip fetching it rather than pulling and discarding it.
    """
    session = fastf1.get_session(season, round_number, session_type)
    session.load(telemetry=with_telemetry)
    return session


def extract_race_meta(session: fastf1.core.Session) -> pd.DataFrame:
    """One row describing the race itself — feeds the `Races` table."""
    event = session.event
    return pd.DataFrame(
        [
            {
                "season": int(event["EventDate"].year),
                "round": int(event["RoundNumber"]),
                "circuit_id": event["Location"],
                "name": event["EventName"],
                "date": event["EventDate"],
                "session_type": session.name,
            }
        ]
    )


def extract_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """Lap-level data — feeds the `Laps` table.

    FastF1's `session.laps` already carries `Driver`, `LapNumber`, `LapTime`,
    `Compound`, `TyreLife`, `Stint`, `PitInTime`/`PitOutTime`, `Position`,
    and `TrackStatus` per lap, which map directly onto Section 7.1's schema.
    """
    laps = session.laps.reset_index(drop=True).copy()
    laps["season"] = session.event["EventDate"].year
    laps["round"] = session.event["RoundNumber"]
    return laps


def extract_weather(session: fastf1.core.Session) -> pd.DataFrame:
    """Session weather samples — feeds the `Weather` table."""
    weather = session.weather_data.reset_index(drop=True).copy()
    weather["season"] = session.event["EventDate"].year
    weather["round"] = session.event["RoundNumber"]
    return weather


def extract_track_status(session: fastf1.core.Session) -> pd.DataFrame:
    """SC/VSC/Yellow/Red events — feeds the `TrackStatus` table."""
    status = session.track_status.reset_index(drop=True).copy()
    status["season"] = session.event["EventDate"].year
    status["round"] = session.event["RoundNumber"]
    return status


def extract_results(session: fastf1.core.Session) -> pd.DataFrame:
    """Classification/results for the session — grid, finish, points."""
    results = session.results.reset_index(drop=True).copy()
    results["season"] = session.event["EventDate"].year
    results["round"] = session.event["RoundNumber"]
    return results


def extract_telemetry(session: fastf1.core.Session) -> pd.DataFrame:
    """Per-lap telemetry channels — feeds the `Telemetry` table.

    This is the expensive extraction: it pulls speed/throttle/brake/DRS/gear/
    rpm traces for every lap of every driver, so callers gate it behind an
    explicit flag (see `ingest.py --with-telemetry`) rather than always
    fetching it.
    """
    frames: list[pd.DataFrame] = []
    for _, lap in session.laps.iterlaps():
        try:
            car_data = lap.get_car_data().add_distance()
        except Exception:
            # Some laps (in/out laps, red-flag-affected laps) have no
            # recorded telemetry channel — skip rather than fail the batch.
            continue

        car_data = car_data.rename(
            columns={
                "Time": "timestamp",
                "Speed": "speed",
                "Throttle": "throttle",
                "Brake": "brake",
                "DRS": "drs",
                "nGear": "gear",
                "RPM": "rpm",
                "Distance": "distance_on_lap",
            }
        )
        car_data["driver_id"] = lap["Driver"]
        car_data["lap_number"] = lap["LapNumber"]
        frames.append(car_data)

    if not frames:
        return pd.DataFrame()

    telemetry = pd.concat(frames, ignore_index=True)
    telemetry["season"] = session.event["EventDate"].year
    telemetry["round"] = session.event["RoundNumber"]
    return telemetry
