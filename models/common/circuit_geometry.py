"""Circuit geometry as model features — gold.circuit_geometry, built by
track_maps/ (`uv run python -m track_maps.run`).

One lookup shared by training (prepare_dataset) and inference
(strategy_engine/oracles.py), so the two cannot disagree about which
columns exist or how a circuit with no map is represented (NaN — missing,
which LightGBM handles natively — never 0).

Only the columns that survived track_maps/evaluate_lift.py are here.
Excluded from every model: sector average speeds (lap length over sector
speed is essentially the reference race's lap time — a leak into lap-time
targets), pit_lane_delta and tyre_stress_index (averaged over all
history, including the rows being predicted), DRS length (null by
regulation era, not a property of the circuit) and track_evolution_rate
(null).

The speed-derived columns come from one reference lap per circuit, often
from 2025 or 2026. They are constant per circuit across every season, so
they describe the circuit rather than any particular race — but the
caveat is recorded here because it's the first thing to revisit if
reference laps are ever re-chosen.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.db import get_connection

LAYOUT_COLUMNS = [
    "lap_length_m",
    "corner_count",
    "avg_corner_radius_m",
    "elevation_range_m",
    "elevation_variance",
]
SPEED_DERIVED_COLUMNS = [
    "slow_corner_count",
    "medium_corner_count",
    "fast_corner_count",
    "slow_corner_pct",
    "total_braking_distance_m",
    "downforce_demand_index",
]
CIRCUIT_GEOMETRY_COLUMNS = LAYOUT_COLUMNS + SPEED_DERIVED_COLUMNS


@lru_cache(maxsize=1)
def _geometry_table() -> pd.DataFrame:
    con = get_connection()
    try:
        return (
            con.execute(f"SELECT circuit_id, {', '.join(CIRCUIT_GEOMETRY_COLUMNS)} FROM gold.circuit_geometry")
            .df()
            .set_index("circuit_id")
            .astype(float)
        )
    finally:
        con.close()


def add_circuit_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the geometry columns by circuit_id. Keeps row order, index
    and circuit_id's dtype exactly as they were."""
    df = df.copy()
    table = _geometry_table()
    key = df["circuit_id"].astype(str)
    for column in CIRCUIT_GEOMETRY_COLUMNS:
        df[column] = key.map(table[column]).astype(float)
    return df


def circuit_geometry(circuit_id: str) -> dict[str, float]:
    table = _geometry_table()
    if circuit_id not in table.index:
        return {column: np.nan for column in CIRCUIT_GEOMETRY_COLUMNS}
    return table.loc[circuit_id].to_dict()
