"""Layer 1 of PRD Section 8's CarProfile — the data-inferred team car
characteristics, computed from the Gold layer.

PRD 8.1 describes a profile "per team per season, updated after every
race". Taken literally as one row per (team, season) computed over that
whole season, the profile would be unusable as a model feature: a race in
round 3 would be described by characteristics partly inferred from round
18. So the grain here is one row per (race_id, team_id), holding what was
knowable about that car *before* that race — the same expanding-window
discipline gold/driver_history.py and gold/circuit_history.py already
use, for the same leakage reason.

The window is restricted to the same season, deliberately: a CarProfile
describes a specific car, and next season's car is a different object.
This is the same reasoning that made historical_dnf_rate
regulation-era-aware in circuit_history.py — carrying last year's chassis
characteristics into this year's profile would describe a car that no
longer exists. The cost is that round 1 of any season has no profile at
all, and early rounds have a thin one; that's honest, and the
`races_observed` column is there so a consumer can weigh it.

Layer 2 of PRD 8.1 (the manually seeded engineering priors —
`downforce_philosophy`, `tyre_operating_window`, `setup_sensitivity`,
`known_weaknesses`) is NOT here. Those come from technical journalism
about each car, which this project doesn't ingest; inventing plausible
values would produce a table that looks complete and quietly feeds
fiction into the strategy engine. The columns are absent rather than
null-filled so their absence is visible.

Two characteristics deviate from the PRD's literal definition, both noted
at their SQL:
- `downforce_proxy` assumes sector 2 is the high-downforce sector, which
  PRD 8.1 states as a general rule. It isn't true at every circuit, so
  this is a field-relative approximation, not a measurement.
- `aero_sensitivity` is computed across races rather than within one:
  wind varies far more between race weekends than within a session in
  this data (the whole dataset spans 0-8.5 units), so a within-race
  windy-vs-still split would mostly measure noise.
"""

from __future__ import annotations

import logging

import duckdb

logger = logging.getLogger(__name__)

# Below this many prior races in the season, an inferred characteristic is
# averaging so few observations that it's closer to noise than signal. The
# profile row is still emitted (with races_observed set) rather than
# suppressed — consumers decide.
MIN_RACES_FOR_CONFIDENCE = 3

_SQL = """
CREATE OR REPLACE TABLE gold.car_profiles AS
WITH green AS (
    -- Every pace measurement below is restricted to representative racing
    -- laps. A pit lap includes pit-lane transit, and a lap under safety
    -- car, VSC or red flag is a track-condition artefact, not car
    -- characteristic — averaging those in would describe the race, not the
    -- car.
    SELECT
        lf.*,
        r.season,
        r.date,
        (NOT lf.is_pit_lap
         AND NOT lf.safety_car_active
         AND NOT lf.vsc_active
         AND NOT lf.red_flag_active) AS is_green_lap,
        ROW_NUMBER() OVER (
            PARTITION BY lf.race_id, lf.driver_id, lf.stint_number ORDER BY lf.lap_number
        ) AS lap_in_stint
    FROM gold.lap_features lf
    JOIN silver.races r ON r.race_id = lf.race_id
),
field_sectors AS (
    -- Sector pace is only comparable within the same race and lap, so the
    -- field baseline is per (race, lap) exactly like field_avg_lap_time is.
    SELECT
        race_id,
        lap_number,
        AVG(sector2_time_seconds) FILTER (WHERE is_green_lap) AS field_s2,
        AVG(sector1_time_seconds + sector3_time_seconds) FILTER (WHERE is_green_lap) AS field_s13
    FROM green
    GROUP BY race_id, lap_number
),
restarts AS (
    -- The lap a car takes immediately after a safety car period ends.
    SELECT
        race_id,
        driver_id,
        lap_number,
        pace_delta_this_lap
    FROM (
        SELECT
            race_id,
            driver_id,
            lap_number,
            pace_delta_this_lap,
            safety_car_active,
            LAG(safety_car_active) OVER (PARTITION BY race_id, driver_id ORDER BY lap_number) AS was_sc
        FROM green
        WHERE NOT is_pit_lap AND NOT red_flag_active
    )
    WHERE was_sc AND NOT safety_car_active
),
pit_cycles AS (
    -- Track position around a team's own pit cycle: where the car was when
    -- it committed to the stop, versus where it actually came out three
    -- laps later. PRD 8.1 defines undercut_vulnerability as position loss
    -- "when rivals pit first", which needs per-rival pit timing this
    -- warehouse doesn't carry. Net position lost across a car's own pit
    -- cycle is the closest thing the data supports: a car that is
    -- routinely jumped while stopping is, in effect, the vulnerable one.
    SELECT
        race_id,
        team_id,
        position_after - position_at_stop AS positions_lost
    FROM (
        SELECT
            race_id,
            team_id,
            current_position AS position_at_stop,
            LEAD(current_position, 3) OVER (
                PARTITION BY race_id, driver_id ORDER BY lap_number
            ) AS position_after,
            is_pit_lap
        FROM green
    )
    WHERE is_pit_lap AND position_after IS NOT NULL
),
race_team AS (
    -- One row per (race, team): this race's raw observation of each
    -- characteristic. Averaging these across prior races is what makes a
    -- profile.
    SELECT
        g.race_id,
        g.season,
        g.date,
        g.team_id,
        AVG(g.track_temp) AS race_track_temp,
        AVG(g.wind_speed) AS race_wind_speed,

        -- Degradation steepness, per compound and overall, measured
        -- relative to nothing here — the field comparison happens in the
        -- profile stage via race_field below.
        AVG(g.degradation_rate) FILTER (WHERE g.is_green_lap) AS team_degradation_rate,
        AVG(g.degradation_rate) FILTER (WHERE g.is_green_lap AND g.compound = 'SOFT') AS degradation_rate_soft,
        AVG(g.degradation_rate) FILTER (WHERE g.is_green_lap AND g.compound = 'MEDIUM') AS degradation_rate_medium,
        AVG(g.degradation_rate) FILTER (WHERE g.is_green_lap AND g.compound = 'HARD') AS degradation_rate_hard,

        -- Warmup: how much slower the first three laps of a stint are than
        -- laps 4-10, once the tyre is in its window. A car that warms tyres
        -- slowly pays more here.
        AVG(g.pace_delta_this_lap) FILTER (WHERE g.is_green_lap AND g.tyre_age <= 3)
            - AVG(g.pace_delta_this_lap) FILTER (WHERE g.is_green_lap AND g.tyre_age BETWEEN 4 AND 10)
            AS tyre_warmup_rate,

        -- Cold-tyre loss: the out-lap itself against the lap after it,
        -- which is the sharpest version of the same effect.
        AVG(g.pace_delta_this_lap) FILTER (WHERE g.lap_in_stint = 1 AND NOT g.red_flag_active)
            - AVG(g.pace_delta_this_lap) FILTER (WHERE g.lap_in_stint = 2 AND NOT g.red_flag_active)
            AS cold_tyre_pace_loss,

        -- Downforce proxy: relative pace in the (assumed) high-downforce
        -- sector versus the low-downforce sectors. Negative = comparatively
        -- stronger in S2 = more downforce-biased than the field.
        AVG(g.sector2_time_seconds - fs.field_s2) FILTER (WHERE g.is_green_lap)
            - AVG((g.sector1_time_seconds + g.sector3_time_seconds) - fs.field_s13) FILTER (WHERE g.is_green_lap)
            AS downforce_proxy,

        -- Lap-to-lap consistency in this race, paired with the race's wind
        -- in the profile stage to give aero_sensitivity.
        STDDEV_SAMP(g.pace_delta_this_lap) FILTER (WHERE g.is_green_lap) AS pace_variability,

        COUNT(*) FILTER (WHERE g.is_green_lap) AS green_laps
    FROM green g
    LEFT JOIN field_sectors fs ON fs.race_id = g.race_id AND fs.lap_number = g.lap_number
    GROUP BY g.race_id, g.season, g.date, g.team_id
),
race_field AS (
    -- The field's own degradation in each race, so a team's degradation can
    -- be expressed as a delta rather than an absolute that mostly encodes
    -- how abrasive the circuit was.
    SELECT race_id, AVG(team_degradation_rate) AS field_degradation_rate
    FROM race_team
    GROUP BY race_id
),
race_team_restart AS (
    SELECT
        g.race_id,
        g.team_id,
        AVG(rs.pace_delta_this_lap) - AVG(g.pace_delta_this_lap) FILTER (WHERE g.is_green_lap)
            AS safety_car_restart_pace
    FROM green g
    LEFT JOIN restarts rs
        ON rs.race_id = g.race_id AND rs.driver_id = g.driver_id AND rs.lap_number = g.lap_number
    GROUP BY g.race_id, g.team_id
),
race_team_undercut AS (
    SELECT race_id, team_id, AVG(positions_lost) AS undercut_vulnerability
    FROM pit_cycles
    GROUP BY race_id, team_id
),
observations AS (
    SELECT
        rt.*,
        rt.team_degradation_rate - rf.field_degradation_rate AS degradation_vs_field,
        rtr.safety_car_restart_pace,
        rtu.undercut_vulnerability
    FROM race_team rt
    LEFT JOIN race_field rf ON rf.race_id = rt.race_id
    LEFT JOIN race_team_restart rtr ON rtr.race_id = rt.race_id AND rtr.team_id = rt.team_id
    LEFT JOIN race_team_undercut rtu ON rtu.race_id = rt.race_id AND rtu.team_id = rt.team_id
)
SELECT
    this.race_id,
    this.season,
    this.team_id,
    COUNT(prior.race_id) AS races_observed,

    -- Every column below averages ONLY over `prior` — races of the same
    -- season strictly before this one. `this` supplies the key and nothing
    -- else, which is what keeps this table safe to join onto a model's
    -- feature row for the very race it describes.
    AVG(prior.tyre_warmup_rate) AS tyre_warmup_rate,
    AVG(prior.cold_tyre_pace_loss) AS cold_tyre_pace_loss,
    AVG(prior.downforce_proxy) AS downforce_proxy,
    AVG(prior.degradation_vs_field) AS degradation_vs_field,
    AVG(prior.degradation_rate_soft) AS degradation_rate_soft,
    AVG(prior.degradation_rate_medium) AS degradation_rate_medium,
    AVG(prior.degradation_rate_hard) AS degradation_rate_hard,
    AVG(prior.safety_car_restart_pace) AS safety_car_restart_pace,
    AVG(prior.undercut_vulnerability) AS undercut_vulnerability,

    -- PRD 8.1 defines this one as a correlation: does this car's
    -- degradation, relative to the field, get worse as the track gets
    -- hotter? Needs several prior races before it means anything, hence
    -- races_observed sitting next to it.
    --
    -- Written out as covariance over the product of standard deviations
    -- rather than calling CORR(): DuckDB's CORR raises rather than
    -- returning NULL when either input has zero variance, which happens
    -- constantly here — a team's first prior race gives one observation,
    -- and a back-to-back pair of races can genuinely share a track temp.
    -- NULLIF turns those into a null profile value, which is the honest
    -- answer ("not enough spread to say"), not an aborted build.
    COVAR_SAMP(prior.degradation_vs_field, prior.race_track_temp)
        / NULLIF(STDDEV_SAMP(prior.degradation_vs_field) * STDDEV_SAMP(prior.race_track_temp), 0)
        AS tyre_temp_sensitivity,
    COVAR_SAMP(prior.pace_variability, prior.race_wind_speed)
        / NULLIF(STDDEV_SAMP(prior.pace_variability) * STDDEV_SAMP(prior.race_wind_speed), 0)
        AS aero_sensitivity
FROM observations this
LEFT JOIN observations prior
    ON prior.team_id = this.team_id
    AND prior.season = this.season
    AND prior.date < this.date
GROUP BY this.race_id, this.season, this.team_id
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SQL)
    count = con.execute("SELECT COUNT(*) FROM gold.car_profiles").fetchone()[0]
    usable = con.execute(
        f"SELECT COUNT(*) FROM gold.car_profiles WHERE races_observed >= {MIN_RACES_FOR_CONFIDENCE}"
    ).fetchone()[0]
    logger.info("gold.car_profiles: %s rows (%s with >=%s prior races)", count, usable, MIN_RACES_FOR_CONFIDENCE)
