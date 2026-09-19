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
    ch.historical_sc_rate,
    ch.historical_dnf_rate,
    -- CarProfile characteristics (PRD Section 8), joined on (race, team).
    -- Every column is inferred strictly from that team's PRIOR races in
    -- the same season (see car_profiles/inference/characteristics.py), so
    -- joining them onto this race's own rows introduces no leakage.
    -- LEFT JOIN, not INNER: a team's first race of a season has no profile
    -- yet, and dropping those rows would quietly delete every season
    -- opener from the training set.
    cp.races_observed AS car_profile_races_observed,
    cp.tyre_warmup_rate AS car_tyre_warmup_rate,
    cp.cold_tyre_pace_loss AS car_cold_tyre_pace_loss,
    cp.downforce_proxy AS car_downforce_proxy,
    cp.degradation_vs_field AS car_degradation_vs_field,
    cp.degradation_rate_soft AS car_degradation_rate_soft,
    cp.degradation_rate_medium AS car_degradation_rate_medium,
    cp.degradation_rate_hard AS car_degradation_rate_hard,
    cp.safety_car_restart_pace AS car_safety_car_restart_pace,
    cp.undercut_vulnerability AS car_undercut_vulnerability,
    cp.tyre_temp_sensitivity AS car_tyre_temp_sensitivity,
    cp.aero_sensitivity AS car_aero_sensitivity
FROM gold.lap_features lf
LEFT JOIN gold.driver_history dh ON dh.driver_id = lf.driver_id AND dh.race_id = lf.race_id
LEFT JOIN gold.circuit_history ch ON ch.race_id = lf.race_id
LEFT JOIN gold.car_profiles cp ON cp.race_id = lf.race_id AND cp.team_id = lf.team_id
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.race_features").fetchone()[0]
    logger.info("gold.race_features: %s rows", count)
