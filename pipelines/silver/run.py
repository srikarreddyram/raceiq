"""CLI entry point for the Silver layer: Bronze -> normalised, joined tables.

Build order matters here (unlike Bronze, where sources are independent):
dimensions first (constructors, circuits, drivers, races), then laps
(which resolves ids against those dimensions), then pit_stops (which reads
back out of silver.laps). Safe to re-run anytime — every table is a full
`CREATE OR REPLACE` derived straight from Bronze.

Usage:
    uv run python -m pipelines.silver.run
"""

from __future__ import annotations

import logging

from pipelines.silver import calendar, circuits, constructors, drivers, laps, pit_stops, races, track_status, weather
from pipelines.silver.db import get_connection

logger = logging.getLogger(__name__)

BUILD_ORDER = [
    ("constructors", constructors.build),
    ("circuits", circuits.build),
    ("drivers", drivers.build),
    ("races", races.build),
    ("calendar", calendar.build),
    ("laps", laps.build),
    ("pit_stops", pit_stops.build),
    ("track_status", track_status.build),
    ("weather", weather.build),
]


def run() -> None:
    con = get_connection()
    try:
        for name, build_fn in BUILD_ORDER:
            logger.info("Building silver.%s", name)
            build_fn(con)
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
