"""Gold Lap Features — PRD Section 10.6 (Race State), 10.5 (Weather, partial),
10.2 (Tyre, partial).

This is the base fact table at the PRD's stated Gold grain: one row per lap
per driver. Every feature here is computable in real time from information
that exists at or before the current lap in the *same* race — nothing here
looks at a future lap or a future race, so this table alone introduces no
temporal leakage. Cross-race historical features (driver form, circuit
baselines) live separately in `driver_history.py` / `circuit_history.py`,
which need their own leakage discipline (as-of-race-date, prior races
only) and shouldn't be mixed into this same-race logic.

Two PRD-named fields are deliberately NOT built here:

- `rain_probability_next_10_laps` would require looking at laps *after*
  the current one within the same race. Since our weather source is
  actual observed history, not a real forecast feed, computing this from
  future laps would mean training on the literal answer — exactly the
  leakage the PRD's "zero data leakage" success criterion guards against.
  It's left out until a genuine forecast source is wired in.
- `predicted_remaining_life` (tyre) and the team-operating-window fields
  are model/CarProfile outputs, not engineered inputs — they don't belong
  in the feature store that feeds those models.

Two window-function features worth knowing how they work:

- `degradation_rate`: an *expanding* linear regression slope of lap time
  against tyre age, computed with DuckDB's `regr_slope` as a window
  aggregate restricted to `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT
  ROW` within each (race, driver, stint) — i.e. "the degradation trend
  using only laps completed so far this stint," never future laps. It's
  NULL for a stint's first lap or two (not enough points to fit a slope)
  and doesn't exclude Safety Car / VSC laps from the fit, which can skew
  it — a known simplification, not a correctness bug.
- `grip_estimate`: lap time minus that stint's first lap time (via
  `FIRST_VALUE` over the same expanding frame) — pace lost relative to a
  fresh-tyre baseline, PRD's definition verbatim.

`gap_to_car_ahead`/`gap_to_car_behind` are derived from `gap_to_leader`
(already computed in Silver) rather than re-deriving cumulative time,
since the leader term cancels out in the subtraction — see the SQL below.

`safety_car_active`/`yellow_active`/`vsc_active`/`red_flag_active` are
derived from `track_status_code` — FastF1's own per-lap summary string,
which can hold multiple digits (e.g. "24" means both Yellow(2) and
SafetyCar(4) occurred at some point during that lap) — rather than from
an as-of join against the race-wide TrackStatus *event* stream
(silver.track_status). An earlier version used that event-stream join,
which locates each status change against the shared lap-boundary timeline
(lap_timeline.py's median-across-drivers approximation) — precise for
"which lap was this event near," but imprecise by about a lap right at a
flag transition. That imprecision was invisible until a Lap Time model
trained on it showed the exact signature: several drivers jumping ~35-45s
at the identical lap number, with the flag column still reading "Clear."
The per-lap summary string doesn't have that boundary-approximation
problem — it's FastF1's own record of what happened *during* the lap that
already finished, so using it is strictly current/past information, not a
leakage risk.

`field_avg_lap_time_seconds`, `degradation_rate`, and `stint_first_lap_time`
all exclude red-flagged laps (`is_red_flag_lap`, computed early enough in
the CTE chain to gate these), not just pit laps. This was found training
a second lap-time model (an LSTM, models/lap_time_sequence/) whose
validation loss stayed inexplicably ~10x worse than test loss throughout
training: a 2024 race's red flag left every driver's lap 1 recorded at
~2,500 seconds (FastF1's lap time spans the full session-clock stoppage,
not real pace), and because `degradation_rate` is an *expanding* window
regression, that one lap didn't just corrupt its own row — it corrupted
every later lap's fitted degradation rate for that whole stint.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

_SQL = """
CREATE OR REPLACE TABLE gold.lap_features AS
WITH race_totals AS (
    SELECT race_id, MAX(lap_number) AS race_total_laps
    FROM silver.laps
    GROUP BY race_id
),
rival_joined AS (
    SELECT
        l.*,
        LAG(l.driver_id) OVER w AS rival_driver_id,
        LAG(l.team_id) OVER w AS rival_team_id,
        LAG(l.compound) OVER w AS rival_compound,
        LAG(l.tyre_age) OVER w AS rival_tyre_age,
        LAG(l.gap_to_leader) OVER w AS ahead_gap_to_leader,
        LEAD(l.gap_to_leader) OVER w AS behind_gap_to_leader,
        -- Computed here (rather than only in the final SELECT, where the
        -- equivalent red_flag_active column also lives) so it can gate the
        -- field-average and degradation-rate windows below. A red-flagged
        -- lap's recorded time spans the full session-clock stoppage, not
        -- real pace (one 2024 race shows every driver's lap 1 at ~2,500
        -- seconds) — left unfiltered, one such lap would corrupt not just
        -- its own row but *every later lap's* degradation_rate in that
        -- stint, since that window is an expanding regression over all
        -- laps so far.
        COALESCE(l.track_status_code LIKE '%5%', FALSE) AS is_red_flag_lap
    FROM silver.laps l
    WINDOW w AS (PARTITION BY l.race_id, l.lap_number ORDER BY l.position)
),
with_field_avg AS (
    SELECT
        rj.*,
        AVG(lap_time_seconds) FILTER (WHERE NOT is_pit_lap AND NOT is_red_flag_lap)
            OVER (PARTITION BY race_id, lap_number) AS field_avg_lap_time_seconds
    FROM rival_joined rj
),
with_tyre AS (
    SELECT
        wfa.*,
        regr_slope(lap_time_seconds, tyre_age)
            FILTER (WHERE NOT is_pit_lap AND NOT is_red_flag_lap) OVER stint_so_far
            AS degradation_rate,
        FIRST_VALUE(CASE WHEN is_pit_lap OR is_red_flag_lap THEN NULL ELSE lap_time_seconds END IGNORE NULLS)
            OVER stint_so_far AS stint_first_lap_time
    FROM with_field_avg wfa
    WINDOW stint_so_far AS (
        PARTITION BY race_id, driver_id, stint_number ORDER BY lap_number
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    )
)
SELECT
    wt.lap_id,
    wt.race_id,
    wt.driver_id,
    wt.team_id,
    wt.lap_number,
    rt.race_total_laps - wt.lap_number AS laps_remaining,
    wt.position AS current_position,
    wt.gap_to_leader,
    (wt.gap_to_leader - wt.ahead_gap_to_leader) AS gap_to_car_ahead,
    (wt.behind_gap_to_leader - wt.gap_to_leader) AS gap_to_car_behind,
    COALESCE((wt.gap_to_leader - wt.ahead_gap_to_leader) < 1.0, FALSE) AS traffic_flag,
    wt.rival_driver_id,
    wt.rival_team_id,
    wt.rival_compound,
    wt.rival_tyre_age,
    wt.track_status_code,
    COALESCE(wt.track_status_code LIKE '%4%', FALSE) AS safety_car_active,
    COALESCE(wt.track_status_code LIKE '%2%', FALSE) AS yellow_active,
    COALESCE(wt.track_status_code LIKE '%6%' OR wt.track_status_code LIKE '%7%', FALSE) AS vsc_active,
    wt.is_red_flag_lap AS red_flag_active,
    wthr.air_temp,
    wthr.track_temp,
    wthr.humidity,
    wthr.wind_speed,
    wthr.wind_direction,
    COALESCE(wthr.rainfall, FALSE) AS rainfall_flag,
    wt.compound,
    wt.tyre_age,
    wt.stint_number,
    wt.is_pit_lap,
    wt.pit_stop_duration,
    wt.lap_time_seconds,
    wt.field_avg_lap_time_seconds,
    (wt.lap_time_seconds - wt.field_avg_lap_time_seconds) AS pace_delta_this_lap,
    wt.degradation_rate,
    (wt.lap_time_seconds - wt.stint_first_lap_time) AS grip_estimate
FROM with_tyre wt
LEFT JOIN race_totals rt ON rt.race_id = wt.race_id
ASOF LEFT JOIN silver.weather wthr
    ON wt.race_id = wthr.race_id AND wt.lap_number >= wthr.lap_number
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.lap_features").fetchone()[0]
    logger.info("gold.lap_features: %s rows", count)
