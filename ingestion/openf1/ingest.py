"""CLI entry point for OpenF1 ingestion.

Usage:
    uv run python -m ingestion.openf1.ingest --session-key 9159
"""

from __future__ import annotations

import argparse
import logging

import requests

from ingestion.config import load_config
from ingestion.openf1 import client
from ingestion.raw_writer import write_json

logger = logging.getLogger(__name__)

FETCHERS = {
    "laps": client.get_laps,
    "pit": client.get_pit,
    "position": client.get_position,
    "weather": client.get_weather,
    "race_control": client.get_race_control,
}


def ingest_session(session_key: int) -> None:
    """Fetch each OpenF1 endpoint independently.

    Some endpoints (e.g. `pit`) return a 404 "no results" for sessions with
    nothing to report rather than an empty list, which is a legitimate
    outcome, not a failure. Endpoints are fetched and written one at a time
    so a single missing endpoint doesn't discard the tables that already
    succeeded.
    """
    config = load_config()
    base_path = str(session_key)

    for name, fetch in FETCHERS.items():
        try:
            payload = fetch(session_key, config)
        except requests.HTTPError as error:
            logger.warning("Skipping %s for session %s: %s", name, session_key, error)
            continue

        destination = write_json(config.raw_data_root, "openf1", f"{base_path}/{name}", payload)
        logger.info("Wrote %s", destination)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest an OpenF1 session into the Raw Layer")
    parser.add_argument("--session-key", type=int, required=True)
    args = parser.parse_args()

    ingest_session(args.session_key)


if __name__ == "__main__":
    main()
