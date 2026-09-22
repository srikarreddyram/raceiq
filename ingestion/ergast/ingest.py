"""CLI entry point for Ergast (Jolpica mirror) ingestion.

Usage:
    uv run python -m ingestion.ergast.ingest --season 2023 --round 1
"""

from __future__ import annotations

import argparse
import logging

from ingestion.config import load_config
from ingestion.ergast import client
from ingestion.raw_writer import write_json

logger = logging.getLogger(__name__)


def ingest_round(season: int, round_number: int) -> None:
    config = load_config()
    base_path = f"{season}/{round_number}"

    payloads = {
        "results": client.get_race_results(season, round_number, config),
        "qualifying": client.get_qualifying_results(season, round_number, config),
        "constructor_standings": client.get_constructor_standings(season, round_number, config),
    }

    for name, payload in payloads.items():
        destination = write_json(config.raw_data_root, "ergast", f"{base_path}/{name}", payload)
        logger.info("Wrote %s", destination)


def ingest_circuits(season: int) -> None:
    config = load_config()
    payload = client.get_circuits(season, config)
    destination = write_json(config.raw_data_root, "ergast", f"{season}/circuits", payload)
    logger.info("Wrote %s", destination)


def ingest_schedule(season: int) -> None:
    """The season's full calendar, run and unrun — the race weekend planner
    plans upcoming races from it, not just the ones with results."""
    config = load_config()
    payload = client.get_season_schedule(season, config)
    destination = write_json(config.raw_data_root, "ergast", f"{season}/schedule", payload)
    logger.info("Wrote %s", destination)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest Ergast-schema data into the Raw Layer")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--round", type=int, dest="round_number", help="Omit to ingest circuits only")
    args = parser.parse_args()

    ingest_circuits(args.season)
    ingest_schedule(args.season)
    if args.round_number is not None:
        ingest_round(args.season, args.round_number)


if __name__ == "__main__":
    main()
