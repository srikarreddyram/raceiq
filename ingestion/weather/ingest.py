"""CLI entry point for weather ingestion.

Usage:
    uv run python -m ingestion.weather.ingest \\
        --circuit-id bahrain --lat 26.0325 --lon 50.5106 \\
        --start-date 2023-03-05 --end-date 2023-03-05
"""

from __future__ import annotations

import argparse
import logging

from ingestion.config import load_config
from ingestion.raw_writer import write_json
from ingestion.weather import client

logger = logging.getLogger(__name__)


def ingest_circuit_weather(
    circuit_id: str, latitude: float, longitude: float, start_date: str, end_date: str
) -> None:
    config = load_config()
    payload = client.get_historical_weather(latitude, longitude, start_date, end_date, config)
    destination = write_json(
        config.raw_data_root, "weather", f"{circuit_id}/{start_date}_{end_date}", payload
    )
    logger.info("Wrote %s", destination)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest circuit weather into the Raw Layer")
    parser.add_argument("--circuit-id", type=str, required=True)
    parser.add_argument("--lat", type=float, required=True, dest="latitude")
    parser.add_argument("--lon", type=float, required=True, dest="longitude")
    parser.add_argument("--start-date", type=str, required=True)
    parser.add_argument("--end-date", type=str, required=True)
    args = parser.parse_args()

    ingest_circuit_weather(args.circuit_id, args.latitude, args.longitude, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
