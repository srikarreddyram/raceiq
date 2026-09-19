"""CLI entry point for CarProfile inference — PRD Section 8.

Reads gold.lap_features and writes gold.car_profiles. Safe to re-run;
the table is a full CREATE OR REPLACE, like every other Gold build.

Must run AFTER pipelines.gold (it reads lap_features) and BEFORE any
model training that uses profile features.

Usage:
    uv run python -m car_profiles.run
"""

from __future__ import annotations

import logging

from car_profiles.inference import characteristics
from pipelines.gold.db import get_connection

logger = logging.getLogger(__name__)


def run() -> None:
    con = get_connection()
    try:
        logger.info("Building gold.car_profiles")
        characteristics.build(con)
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
