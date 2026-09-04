"""Historical backfill orchestrator.

Drives all four ingestion sources across a season range, using Ergast's
season schedule (not a hardcoded round count) to discover which rounds
actually exist. Per the project's data-lookback decision, the default
range is 2018-present: FastF1's full-telemetry coverage doesn't reliably
extend earlier, and pre-2018 seasons would only ever populate
results-level tables, not the lap/tyre/telemetry ones every model needs.

Resumable by design: every step checks whether its raw output file already
exists before making a network call, so a re-run after a partial failure
or an interruption only fetches what's missing. Errors on a single race
are logged and skipped rather than aborting the whole run — a bad session
FastF1 can't load, or a round Jolpica hasn't backfilled yet, shouldn't
lose everything else.

Usage:
    uv run python -m ingestion.backfill --start-season 2018 --end-season 2026
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime

from fastf1.exceptions import RateLimitExceededError

from ingestion.config import IngestionConfig, load_config
from ingestion.ergast import client as ergast_client
from ingestion.ergast.ingest import ingest_circuits, ingest_round
from ingestion.fastf1.ingest import ingest_session as ingest_fastf1_session
from ingestion.openf1 import client as openf1_client
from ingestion.openf1.ingest import ingest_session as ingest_openf1_session
from ingestion.weather.ingest import ingest_circuit_weather

logger = logging.getLogger(__name__)

REQUEST_SPACING_SECONDS = 0.5

# FastF1 self-enforces 500 calls/hour against the FIA data feed (a hard
# external constraint, not something we control). A backfill running at
# full speed burns through that budget in well under an hour, so on
# RateLimitExceededError we back off substantially and retry rather than
# silently losing that race's data - which is what happened before this
# was added: two full seasons (2021-2022) failed instantly, call after
# call, until the sliding window aged out on its own an hour later.
FASTF1_RATE_LIMIT_BACKOFF_SECONDS = 20 * 60
FASTF1_RATE_LIMIT_MAX_RETRIES = 4


def _exists(config: IngestionConfig, *relative_parts: str) -> bool:
    return (config.raw_data_root.joinpath(*relative_parts)).exists()


def _backfill_ergast(season: int, round_number: int, config: IngestionConfig) -> None:
    if _exists(config, "ergast", str(season), str(round_number), "results.json"):
        return
    ingest_round(season, round_number)
    time.sleep(REQUEST_SPACING_SECONDS)


def _backfill_fastf1(season: int, round_number: int, config: IngestionConfig) -> None:
    if _exists(config, "fastf1", str(season), str(round_number), "R", "results.parquet"):
        return

    for attempt in range(1, FASTF1_RATE_LIMIT_MAX_RETRIES + 1):
        try:
            ingest_fastf1_session(season, round_number, "R", with_telemetry=False)
            return
        except RateLimitExceededError:
            if attempt == FASTF1_RATE_LIMIT_MAX_RETRIES:
                raise
            logger.warning(
                "FastF1 rate limit hit on %s round %s (attempt %s/%s) — sleeping %ss",
                season,
                round_number,
                attempt,
                FASTF1_RATE_LIMIT_MAX_RETRIES,
                FASTF1_RATE_LIMIT_BACKOFF_SECONDS,
            )
            time.sleep(FASTF1_RATE_LIMIT_BACKOFF_SECONDS)


def _find_openf1_session_key(season: int, race_date: str, config: IngestionConfig) -> int | None:
    sessions = openf1_client.get_sessions(config, year=season, session_type="Race")
    for session in sessions:
        if session["date_start"].startswith(race_date):
            return session["session_key"]
    return None


def _backfill_openf1(season: int, round_number: int, race_date: str, config: IngestionConfig) -> None:
    try:
        session_key = _find_openf1_session_key(season, race_date, config)
    except Exception as error:  # OpenF1 has no data at all for many older seasons
        logger.warning("OpenF1 session lookup failed for %s round %s: %s", season, round_number, error)
        return

    if session_key is None:
        logger.info("No OpenF1 session found for %s round %s (%s) — likely pre-OpenF1 coverage", season, round_number, race_date)
        return
    if _exists(config, "openf1", str(session_key), "laps.json"):
        return
    ingest_openf1_session(session_key)


def _backfill_weather(circuit_id: str, latitude: float, longitude: float, race_date: str, config: IngestionConfig) -> None:
    if _exists(config, "weather", circuit_id, f"{race_date}_{race_date}.json"):
        return
    ingest_circuit_weather(circuit_id, latitude, longitude, race_date, race_date)
    time.sleep(REQUEST_SPACING_SECONDS)


def backfill_season(season: int, config: IngestionConfig) -> None:
    ingest_circuits(season)

    schedule = ergast_client.get_season_schedule(season, config)
    races = schedule["MRData"]["RaceTable"]["Races"]
    today = date.today()

    for race in races:
        round_number = int(race["round"])
        race_date = race["date"]
        if datetime.strptime(race_date, "%Y-%m-%d").date() > today:
            logger.info("Skipping %s round %s — scheduled for the future (%s)", season, round_number, race_date)
            continue

        circuit = race["Circuit"]
        circuit_id = circuit["circuitId"]
        latitude, longitude = float(circuit["Location"]["lat"]), float(circuit["Location"]["long"])

        logger.info("=== %s round %s: %s (%s) ===", season, round_number, race["raceName"], race_date)
        try:
            _backfill_ergast(season, round_number, config)
        except Exception:
            logger.exception("Ergast ingestion failed for %s round %s", season, round_number)

        try:
            _backfill_fastf1(season, round_number, config)
        except Exception:
            logger.exception("FastF1 ingestion failed for %s round %s", season, round_number)

        try:
            _backfill_openf1(season, round_number, race_date, config)
        except Exception:
            logger.exception("OpenF1 ingestion failed for %s round %s", season, round_number)

        try:
            _backfill_weather(circuit_id, latitude, longitude, race_date, config)
        except Exception:
            logger.exception("Weather ingestion failed for %s round %s", season, round_number)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Backfill all ingestion sources across a season range")
    parser.add_argument("--start-season", type=int, default=2018)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    args = parser.parse_args()

    config = load_config()
    for season in range(args.start_season, args.end_season + 1):
        logger.info("########## Backfilling season %s ##########", season)
        backfill_season(season, config)


if __name__ == "__main__":
    main()
