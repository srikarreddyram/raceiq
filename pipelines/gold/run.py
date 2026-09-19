"""CLI entry point for the Gold layer.

Build order matters: lap_features first (everything else reads from it),
then driver_history and circuit_history (both read gold.lap_features +
silver.races), then race_features (assembles all three). Safe to re-run
anytime — every table is a full `CREATE OR REPLACE`.

Usage:
    uv run python -m pipelines.gold.run
"""

from __future__ import annotations

import logging

from car_profiles.inference import characteristics as car_profiles
from pipelines.gold import circuit_history, driver_history, lap_features, race_features
from pipelines.gold.db import get_connection

logger = logging.getLogger(__name__)

# car_profiles lives under car_profiles/ to match the PRD's repository
# layout (Section 18), but it writes a Gold table and sits squarely in the
# Gold dependency chain — it reads lap_features and race_features reads it
# — so it's orchestrated here. Keeping it out of this list would let
# `pipelines.gold.run` finish "successfully" while race_features silently
# joined against a stale profile table.
BUILD_ORDER = [
    ("lap_features", lap_features.build),
    ("driver_history", driver_history.build),
    ("circuit_history", circuit_history.build),
    ("car_profiles", car_profiles.build),
    ("race_features", race_features.build),
]


def run() -> None:
    con = get_connection()
    try:
        for name, build_fn in BUILD_ORDER:
            logger.info("Building gold.%s", name)
            build_fn(con)
    finally:
        con.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
