"""Shared configuration for all ingestion sources.

Every ingestion client reads its settings from here rather than touching
`os.environ` directly, so the whole ingestion layer has one place that
defines where things live and which knobs are configurable via `.env`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class IngestionConfig:
    # Root directory for the Raw Layer (Section 7). Never committed to git —
    # see .gitignore. Each source writes into its own subdirectory here.
    raw_data_root: Path

    # FastF1 maintains its own on-disk cache of unparsed API responses,
    # separate from our Raw Layer, so repeated `uv run` calls don't re-hit
    # the FIA/timing endpoints.
    fastf1_cache_dir: Path

    # Ergast has signalled deprecation (PRD 6.2 / 18). Kept as a single
    # override point so a mirror or the official F1 API can be swapped in
    # without touching client code.
    ergast_base_url: str

    openf1_base_url: str

    # Open-Meteo's historical + forecast weather API. Free, keyless, and
    # queried by raw lat/lon + timestamp, which matches how we key weather
    # to circuits (Section 6.4). Swap this if the team standardises on a
    # different provider later.
    weather_base_url: str

    request_timeout_seconds: float
    max_retries: int


    # Which FastF1 sessions to ingest. Defaults to the race alone, which is
    # what every model and the whole Gold layer have been built on. Practice
    # and qualifying are what race_plan/tyre_allocation.py needs, and are
    # opt-in because each extra session type roughly multiplies the number of
    # FastF1 calls a backfill makes against a ~500/hour limit.
    fastf1_session_types: tuple[str, ...] = ("R",)


def load_config() -> IngestionConfig:
    return IngestionConfig(
        raw_data_root=Path(os.environ.get("RACEIQ_DATA_ROOT", REPO_ROOT / "data" / "raw")),
        fastf1_cache_dir=Path(
            os.environ.get("FASTF1_CACHE_DIR", REPO_ROOT / ".fastf1_cache")
        ),
        ergast_base_url=os.environ.get("ERGAST_BASE_URL", "https://api.jolpi.ca/ergast/f1"),
        openf1_base_url=os.environ.get("OPENF1_BASE_URL", "https://api.openf1.org/v1"),
        weather_base_url=os.environ.get(
            "WEATHER_BASE_URL", "https://archive-api.open-meteo.com/v1/archive"
        ),
        request_timeout_seconds=float(os.environ.get("RACEIQ_REQUEST_TIMEOUT", "30")),
        max_retries=int(os.environ.get("RACEIQ_MAX_RETRIES", "3")),
        fastf1_session_types=tuple(
            part.strip() for part in os.environ.get("RACEIQ_FASTF1_SESSIONS", "R").split(",") if part.strip()
        ),
    )
