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

Unlike `historical_sc_rate`, this isn't a plain all-time expanding average
per circuit: F1's mechanical reliability resets with each major
regulation cycle rather than drifting gradually — year 1 of an all-new
chassis/power-unit ruleset (2022's ground-effect regs, 2026's new
chassis-and-power-unit rules) reliably brings a wave of teething
mechanical failures that a mature, several-years-in season under the same
stable rules (2025 was year 4 of the 2022 ruleset) doesn't have. Blending
pre- and post-regulation-change seasons into one all-time average would
systematically misjudge both directions at once — overstating risk in a
mature season, understating it in a fresh one — which is exactly the
correction this project's own PRD author flagged from real F1 knowledge,
not something derivable from the data alone. `era` below buckets the
seasons this project's data spans (2018-) into the three real regulation
packages actually involved: 2018-2021 (final years of the 2017-spec
wide-body aero rules), 2022-2025 (the ground-effect ruleset), 2026+ (the
new chassis/power-unit ruleset). `historical_dnf_rate` prefers the
circuit's own expanding average *within the current era*, and only falls
back to a coarser (but still era-matched) estimate when that's too sparse
to trust — see the SQL's COALESCE chain. A brand-new era's early races
will legitimately have thin era-specific history by construction; that's
an honest limit of an empirical approach, not a bug to paper over with a
guessed number.

Grain: one row per race_id (not per lap) — every lap in a race shares the
same circuit-history baseline, since the baseline doesn't change within a
race.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_REGULATION_ERA_SQL = "CASE WHEN r.season <= 2021 THEN 0 WHEN r.season <= 2025 THEN 1 ELSE 2 END"

_SQL = f"""
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
),
race_attrition AS (
    SELECT
        r.race_id,
        r.circuit_id,
        r.date,
        {_REGULATION_ERA_SQL} AS era,
        AVG(
            CASE WHEN res.status = 'Finished' OR res.status = 'Lapped' OR res.status LIKE '+%' THEN 0 ELSE 1 END
        ) AS race_dnf_rate
    FROM silver.races r
    JOIN bronze.ergast_results res ON res.season = r.season AND res.round = r.round
    WHERE res.status != 'Did not start'
    GROUP BY r.race_id, r.circuit_id, r.date, r.season
),
this_race_era AS (
    SELECT r.race_id, r.circuit_id, r.date, {_REGULATION_ERA_SQL} AS era
    FROM silver.races r
),
-- Three fallback levels, each restricted to THIS race's own regulation
-- era (never blending across an era boundary): the circuit's own prior
-- races in this era, then any circuit's prior races in this era, then —
-- only if this era has no prior races at all for any circuit yet, e.g.
-- the very first race(s) of a brand-new ruleset — every prior race
-- regardless of era, as the least-bad information available at that
-- point. See module docstring on why era-matching takes priority over
-- circuit-matching here (the opposite priority from historical_sc_rate).
dnf_rates AS (
    SELECT
        this.race_id,
        AVG(prior.race_dnf_rate) FILTER (WHERE prior.circuit_id = this.circuit_id AND prior.era = this.era)
            AS circuit_era_dnf_rate,
        AVG(prior.race_dnf_rate) FILTER (WHERE prior.era = this.era) AS era_wide_dnf_rate,
        AVG(prior.race_dnf_rate) AS all_time_dnf_rate
    FROM this_race_era this
    LEFT JOIN race_attrition prior ON prior.date < this.date
    GROUP BY this.race_id
)
SELECT
    wb.race_id,
    wb.circuit_id,
    wb.prior_races_at_circuit,
    wb.circuit_baseline_track_temp,
    wb.circuit_baseline_air_temp,
    wb.historical_sc_rate,
    COALESCE(dr.circuit_era_dnf_rate, dr.era_wide_dnf_rate, dr.all_time_dnf_rate) AS historical_dnf_rate,
    (wb.race_avg_track_temp - wb.circuit_baseline_track_temp) AS condition_delta
FROM with_baseline wb
JOIN dnf_rates dr ON dr.race_id = wb.race_id
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.circuit_history").fetchone()[0]
    logger.info("gold.circuit_history: %s rows", count)
