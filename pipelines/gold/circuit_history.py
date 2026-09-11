"""Gold Circuit History — PRD Section 10.5 (`condition_delta`) and
Section 11.4's `historical_sc_rate_this_circuit`.

Both fields use the same expanding-window discipline: an average strictly
over races before the current one's date at the same circuit, never
including the current race itself. This is the leakage rule the PRD's
temporal train/test-split requirement demands applied at feature-
computation time, not just at model-split time — if a race's own weather
or safety-car outcome were folded into its own baseline, the feature
would partly encode its own answer.

Grain: one row per race_id (not per lap) — every lap in a race shares the
same circuit-history baseline, since the baseline doesn't change within a
race.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_SQL = """
CREATE OR REPLACE TABLE gold.circuit_history AS
WITH race_conditions AS (
    SELECT
        r.race_id,
        r.circuit_id,
        r.date,
        AVG(lf.track_temp) AS race_avg_track_temp,
        AVG(lf.air_temp) AS race_avg_air_temp,
        MAX(lf.safety_car_active::INT) AS race_had_safety_car
    FROM silver.races r
    JOIN gold.lap_features lf ON lf.race_id = r.race_id
    GROUP BY r.race_id, r.circuit_id, r.date
),
with_baseline AS (
    SELECT
        this.race_id,
        this.circuit_id,
        this.race_avg_track_temp,
        AVG(prior.race_avg_track_temp) AS circuit_baseline_track_temp,
        AVG(prior.race_avg_air_temp) AS circuit_baseline_air_temp,
        AVG(prior.race_had_safety_car) AS historical_sc_rate,
        COUNT(prior.race_id) AS prior_races_at_circuit
    FROM race_conditions this
    LEFT JOIN race_conditions prior
        ON prior.circuit_id = this.circuit_id AND prior.date < this.date
    GROUP BY this.race_id, this.circuit_id, this.race_avg_track_temp
)
SELECT
    race_id,
    circuit_id,
    prior_races_at_circuit,
    circuit_baseline_track_temp,
    circuit_baseline_air_temp,
    historical_sc_rate,
    (race_avg_track_temp - circuit_baseline_track_temp) AS condition_delta
FROM with_baseline
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.circuit_history").fetchone()[0]
    logger.info("gold.circuit_history: %s rows", count)
