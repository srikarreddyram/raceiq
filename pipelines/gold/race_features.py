"""Gold Race Features — the final assembled feature store table.

This is PRD Section 9's Gold Layer contract made literal: one row per lap
per driver, with `lap_features` (same-race, zero-leakage) joined to
`driver_history` and `circuit_history` (cross-race, as-of-race-date
aggregates). This is the table model training and the serving layer are
meant to read from — nothing upstream of this should be queried directly
by a model.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_SQL = """
CREATE OR REPLACE TABLE gold.race_features AS
SELECT
    lf.*,
    dh.prior_races_count AS driver_prior_races_count,
    dh.avg_pace_delta AS driver_avg_pace_delta,
    dh.consistency_score AS driver_consistency_score,
    dh.wet_weather_rating AS driver_wet_weather_rating,
    dh.overtaking_score AS driver_overtaking_score,
    ch.prior_races_at_circuit,
    ch.circuit_baseline_track_temp,
    ch.condition_delta,
    ch.historical_sc_rate
FROM gold.lap_features lf
LEFT JOIN gold.driver_history dh ON dh.driver_id = lf.driver_id AND dh.race_id = lf.race_id
LEFT JOIN gold.circuit_history ch ON ch.race_id = lf.race_id
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.race_features").fetchone()[0]
    logger.info("gold.race_features: %s rows", count)
