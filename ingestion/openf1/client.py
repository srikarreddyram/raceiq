"""OpenF1 client for live/near-live session data (PRD Section 6.3).

OpenF1 is queried during active sessions to reduce latency relative to
FastF1, which typically lags live timing by design. Every endpoint here
is keyed by `session_key` — OpenF1's identifier for one race weekend
session, discoverable via `get_sessions`.
"""

from __future__ import annotations

from typing import Any

from ingestion.config import IngestionConfig
from ingestion.http import get_json


def get_sessions(config: IngestionConfig, **filters: Any) -> list[dict[str, Any]]:
    """Look up session_key values, e.g. get_sessions(config, year=2024, country_name="Bahrain")."""
    return get_json(f"{config.openf1_base_url}/sessions", config, params=filters)


def get_laps(session_key: int, config: IngestionConfig) -> list[dict[str, Any]]:
    return get_json(f"{config.openf1_base_url}/laps", config, params={"session_key": session_key})


def get_pit(session_key: int, config: IngestionConfig) -> list[dict[str, Any]]:
    return get_json(f"{config.openf1_base_url}/pit", config, params={"session_key": session_key})


def get_position(session_key: int, config: IngestionConfig) -> list[dict[str, Any]]:
    return get_json(f"{config.openf1_base_url}/position", config, params={"session_key": session_key})


def get_weather(session_key: int, config: IngestionConfig) -> list[dict[str, Any]]:
    return get_json(f"{config.openf1_base_url}/weather", config, params={"session_key": session_key})


def get_race_control(session_key: int, config: IngestionConfig) -> list[dict[str, Any]]:
    """Flags, safety car / VSC deployments, and other track-status events."""
    return get_json(
        f"{config.openf1_base_url}/race_control", config, params={"session_key": session_key}
    )
