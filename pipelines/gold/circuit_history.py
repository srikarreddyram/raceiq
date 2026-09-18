"""Gold Circuit History — PRD Section 10.5 (`condition_delta`),
Section 11.4's `historical_sc_rate_this_circuit`, and `historical_dnf_rate`
(not PRD-named, added to ground the strategy engine's rival-attrition
modeling — see simulation/monte_carlo.py's docstring on why a simulated
race previously had no chance of a rival retiring).

All three fields use the same expanding-window discipline: an average
strictly over races before the current one's date at the same circuit,
never including the current race itself. This is the leakage rule the
PRD's temporal train/test-split requirement demands applied at feature-
computation time, not just at model-split time — if a race's own weather,
safety-car, or retirement outcome were folded into its own baseline, the
feature would partly encode its own answer.

`historical_dnf_rate` classifies a result as a retirement unless its
Ergast `status` is "Finished", "Lapped", or a "+N Lap(s)" variant (all
three mean the driver completed the race distance, just not on the lead
lap) — anything else (Accident, Engine, Collision, Retired,
Disqualified, ...) counts as a retirement. "Did not start" is excluded
from the denominator entirely: a driver who never started was never at
risk of retiring mid-race, which is the specific event this feature
grounds.

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
race_attrition AS (
    SELECT
        r.race_id,
        AVG(
            CASE WHEN res.status = 'Finished' OR res.status = 'Lapped' OR res.status LIKE '+%' THEN 0 ELSE 1 END
        ) AS race_dnf_rate
    FROM silver.races r
    JOIN bronze.ergast_results res ON res.season = r.season AND res.round = r.round
    WHERE res.status != 'Did not start'
    GROUP BY r.race_id
),
with_baseline AS (
    SELECT
        this.race_id,
        this.circuit_id,
        this.race_avg_track_temp,
        AVG(prior.race_avg_track_temp) AS circuit_baseline_track_temp,
        AVG(prior.race_avg_air_temp) AS circuit_baseline_air_temp,
        AVG(prior.race_had_safety_car) AS historical_sc_rate,
        AVG(prior_attrition.race_dnf_rate) AS historical_dnf_rate,
        COUNT(prior.race_id) AS prior_races_at_circuit
    FROM race_conditions this
    LEFT JOIN race_conditions prior
        ON prior.circuit_id = this.circuit_id AND prior.date < this.date
    LEFT JOIN race_attrition prior_attrition ON prior_attrition.race_id = prior.race_id
    GROUP BY this.race_id, this.circuit_id, this.race_avg_track_temp
)
SELECT
    race_id,
    circuit_id,
    prior_races_at_circuit,
    circuit_baseline_track_temp,
    circuit_baseline_air_temp,
    historical_sc_rate,
    historical_dnf_rate,
    (race_avg_track_temp - circuit_baseline_track_temp) AS condition_delta
FROM with_baseline
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.circuit_history").fetchone()[0]
    logger.info("gold.circuit_history: %s rows", count)
