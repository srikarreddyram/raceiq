"""CLI entry point for the Bronze layer: Raw -> DuckDB.

Rebuilds every Bronze table from whatever is currently on disk under
data/raw/. Safe to re-run at any time — each loader does a full
`CREATE OR REPLACE` rather than an incremental append (see
pipelines/bronze/common.py).

Usage:
    uv run python -m pipelines.bronze.run
    uv run python -m pipelines.bronze.run --source fastf1
"""

from __future__ import annotations

import argparse
import logging

from ingestion.config import load_config
from pipelines.bronze import ergast_loader, fastf1_loader, openf1_loader, weather_loader
from pipelines.bronze.db import get_connection

logger = logging.getLogger(__name__)

LOADERS = {
    "fastf1": fastf1_loader.load_all,
    "ergast": ergast_loader.load_all,
    "openf1": openf1_loader.load_all,
    "weather": weather_loader.load_all,
}


def run(sources: list[str]) -> None:
    raw_root = load_config().raw_data_root
    con = get_connection()
    try:
        for source in sources:
            logger.info("Loading bronze tables for source=%s", source)
            LOADERS[source](raw_root, con)
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Build the Bronze layer from the Raw Layer")
    parser.add_argument("--source", choices=sorted(LOADERS), action="append", dest="sources")
    args = parser.parse_args()

    run(args.sources or sorted(LOADERS))


if __name__ == "__main__":
    main()
