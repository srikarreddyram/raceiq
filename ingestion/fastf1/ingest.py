"""CLI entry point for FastF1 ingestion.

Usage:
    uv run python -m ingestion.fastf1.ingest --season 2023 --round 1
    uv run python -m ingestion.fastf1.ingest --season 2023 --round 1 --with-telemetry
"""

from __future__ import annotations

import argparse
import logging

from ingestion.config import load_config
from ingestion.fastf1 import client
from ingestion.raw_writer import write_table

logger = logging.getLogger(__name__)


def ingest_session(season: int, round_number: int, session_type: str, with_telemetry: bool) -> None:
    config = load_config()
    client.enable_cache(config.fastf1_cache_dir)

    logger.info("Loading FastF1 session season=%s round=%s type=%s", season, round_number, session_type)
    session = client.load_session(season, round_number, session_type)

    base_path = f"{season}/{round_number}/{session_type}"
    tables = {
        "race_meta": client.extract_race_meta(session),
        "laps": client.extract_laps(session),
        "weather": client.extract_weather(session),
        "track_status": client.extract_track_status(session),
        "results": client.extract_results(session),
    }
    if with_telemetry:
        tables["telemetry"] = client.extract_telemetry(session)

    for table_name, frame in tables.items():
        if frame.empty:
            logger.warning("No data for table=%s season=%s round=%s", table_name, season, round_number)
            continue
        destination = write_table(config.raw_data_root, "fastf1", f"{base_path}/{table_name}", frame)
        logger.info("Wrote %s rows to %s", len(frame), destination)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest a FastF1 session into the Raw Layer")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_number")
    parser.add_argument("--session", type=str, default="R", help="R, Q, FP1, FP2, FP3, S")
    parser.add_argument("--with-telemetry", action="store_true")
    args = parser.parse_args()

    ingest_session(args.season, args.round_number, args.session, args.with_telemetry)


if __name__ == "__main__":
    main()
