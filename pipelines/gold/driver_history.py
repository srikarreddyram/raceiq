"""Gold Driver History — PRD Section 10.1 (Driver Features), partial.

Every aggregate here is computed *as of* each race — using only races with
an earlier date for that driver, never the target race itself or anything
after it — for the same reason as `circuit_history.py`: folding the
target race into its own historical baseline would leak the answer into
the feature.

Built: `avg_pace_delta` (circuit-specific — only meaningful compared
within the same track), `overtaking_score` (positions gained, grid minus
finish, from Ergast results), `consistency_score` (stddev of pace delta
in "clean air" laps only — not in traffic, not a pit lap), and
`wet_weather_rating` (wet-lap pace delta minus dry-lap pace delta).

Two PRD-named fields are deliberately NOT built here:

- `qualifying_delta` needs parsed qualifying lap times in seconds; Silver
  only carries Ergast's raw Q1/Q2/Q3 time strings (`bronze.ergast_qualifying`)
  and no `silver.qualifying` table exists yet to normalise them.
- `aggression_score` needs incident data with fault attribution, which
  isn't ingested anywhere — `bronze.openf1_race_control` has free-text
  messages that could eventually be mined for this, but that's a real
  parsing project of its own, not a quick addition here.

`overtaking_score` is also a simplification of the PRD's wording
("at this circuit type") to "globally, any circuit" — a circuit-type
taxonomy doesn't exist yet (it depends on `track_maps/`, not built), and
most driver/circuit pairs don't have enough prior races for a
circuit-scoped version to be meaningful anyway.

`tyre_conservation_index` (degradation relative to teammate) is also
deferred — it needs same-race teammate matching by team_id and compound,
which deserves its own careful pass rather than being bolted onto this
query.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_SQL = """
CREATE OR REPLACE TABLE gold.driver_history AS
WITH lap_hist AS (
    SELECT
        lf.driver_id,
        lf.race_id,
        r.circuit_id,
        r.date,
        lf.pace_delta_this_lap,
        lf.rainfall_flag,
        lf.traffic_flag,
        lf.is_pit_lap
    FROM gold.lap_features lf
    JOIN silver.races r ON r.race_id = lf.race_id
),
lap_targets AS (
    SELECT DISTINCT driver_id, race_id, circuit_id, date FROM lap_hist
),
pace_and_consistency AS (
    SELECT
        t.driver_id,
        t.race_id,
        AVG(CASE WHEN h.circuit_id = t.circuit_id THEN h.pace_delta_this_lap END) AS avg_pace_delta,
        STDDEV_SAMP(CASE WHEN NOT h.traffic_flag AND NOT h.is_pit_lap THEN h.pace_delta_this_lap END)
            AS consistency_score,
        AVG(CASE WHEN h.rainfall_flag THEN h.pace_delta_this_lap END)
            - AVG(CASE WHEN NOT h.rainfall_flag THEN h.pace_delta_this_lap END) AS wet_weather_rating,
        COUNT(DISTINCT h.race_id) AS prior_races_count
    FROM lap_targets t
    LEFT JOIN lap_hist h ON h.driver_id = t.driver_id AND h.date < t.date
    GROUP BY t.driver_id, t.race_id
),
results_hist AS (
    SELECT
        res.driver_id,
        res.season,
        res.round,
        r.date,
        (res.grid - res.position) AS places_gained
    FROM bronze.ergast_results res
    JOIN silver.races r ON r.season = res.season AND r.round = res.round
    WHERE res.grid IS NOT NULL AND res.position IS NOT NULL
),
results_targets AS (
    SELECT DISTINCT driver_id, season, round, date FROM results_hist
),
overtaking AS (
    SELECT
        t.driver_id,
        t.season,
        t.round,
        AVG(h.places_gained) AS overtaking_score
    FROM results_targets t
    LEFT JOIN results_hist h ON h.driver_id = t.driver_id AND h.date < t.date
    GROUP BY t.driver_id, t.season, t.round
)
SELECT
    pc.driver_id,
    pc.race_id,
    pc.prior_races_count,
    pc.avg_pace_delta,
    pc.consistency_score,
    pc.wet_weather_rating,
    ot.overtaking_score
FROM pace_and_consistency pc
LEFT JOIN silver.races r ON r.race_id = pc.race_id
LEFT JOIN overtaking ot
    ON ot.driver_id = pc.driver_id AND ot.season = r.season AND ot.round = r.round
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.driver_history").fetchone()[0]
    logger.info("gold.driver_history: %s rows", count)
