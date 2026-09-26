"""The rest of the grid, as it looks BEFORE the race starts.

strategy_engine/field.py builds a rival snapshot from live race state —
everyone's current gap and their pace over the last five laps. Reusing it
for a pre-race plan is wrong in a way that quietly destroys the plan:
called at lap 1, every gap is near zero and every "recent pace delta" is
first-lap start chaos, and the simulation then extrapolates that across
the whole race. Measured on Bahrain 2025 that put Norris 249 seconds
ahead of the field and Bearman 125 behind. With rivals scattered over
±250s, no tyre strategy could possibly change a finishing position, so
every candidate scored identically and the planner reported a three-way
tie between starting compounds as though it were a recommendation.

So a pre-race field needs two different inputs, both stable and both
knowable on Saturday night:

  track position   the grid, converted to a time gap using the median
                   early-race gap actually observed at each running
                   position across this project's whole history (P2 sits
                   1.5s back, P10 11.4s, P20 23.2s). Measured, not assumed.

  pace             each driver's season-to-date average pace delta, over
                   races strictly before this one, sharpened by this
                   weekend's qualifying gap once there is one
                   (race_plan/weekend_pace.py). Season form is the stable
                   thing a strategist reasons from; qualifying is what
                   they know about this track by Saturday night; and
                   restricting form to prior races keeps a pre-race plan
                   from reading results it couldn't have had.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from models.common.db import get_connection
from race_plan.weekend_pace import qualifying_gaps, weekend_pace
from strategy_engine.field import RivalTrend
from strategy_engine.tyre_baselines import StopPatterns, historical_stop_patterns

# Fallback for a grid slot with no measured history (deep grids in older
# seasons are thin). Roughly the per-position spacing the table shows.
_GAP_PER_POSITION_FALLBACK = 1.2


@lru_cache(maxsize=None)
def _starting_gap_by_position() -> dict[int, float]:
    """Median gap to the leader at lap 3, by running position.

    Lap 3 rather than lap 1: the first two laps still contain the start
    and its concertina effect, and what's wanted here is the settled
    running order a strategy has to work against.
    """
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT CAST(current_position AS INT) AS position,
                   MEDIAN(gap_to_leader) AS median_gap
            FROM gold.race_features
            WHERE lap_number = 3
                AND current_position IS NOT NULL
                AND gap_to_leader IS NOT NULL
            GROUP BY position
            """
        ).df()
    finally:
        con.close()
    return {int(row.position): float(row.median_gap) for row in rows.itertuples()}


def starting_gap_for_position(position: int) -> float:
    table = _starting_gap_by_position()
    if position in table:
        return table[position]
    return (position - 1) * _GAP_PER_POSITION_FALLBACK


@lru_cache(maxsize=None)
def _season_form(season: int, before_date: str) -> dict[str, float]:
    """Average green-flag pace delta per driver, over races earlier in the
    same season only.

    Same-season only for the same reason car_profiles/ is: the car is what
    sets pace, and last season's car is a different one.
    """
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT rf.driver_id, AVG(rf.pace_delta_this_lap) AS pace_delta
            FROM gold.race_features rf
            JOIN silver.races r ON r.race_id = rf.race_id
            WHERE r.season = ?
                AND r.date < ?
                AND NOT rf.is_pit_lap
                AND NOT rf.safety_car_active
                AND NOT rf.vsc_active
                AND NOT rf.red_flag_active
            GROUP BY rf.driver_id
            """,
            [season, before_date],
        ).df()
    finally:
        con.close()
    return {row.driver_id: float(row.pace_delta) for row in rows.itertuples()}


def build_pre_race_field(race_id: str, exclude_driver_id: str) -> list[RivalTrend]:
    season, rnd = (int(part) for part in race_id.split("_"))

    con = get_connection()
    try:
        race_date = con.execute("SELECT date FROM silver.races WHERE race_id = ?", [race_id]).fetchone()
        entries = con.execute(
            """
            SELECT driver_id, constructor_id, grid
            FROM bronze.ergast_results
            WHERE season = ? AND round = ? AND driver_id != ?
            """,
            [season, rnd, exclude_driver_id],
        ).df()
        # Whatever tyre each rival actually started on isn't knowable before
        # the race either; the simulation only uses a rival's compound to
        # judge whether they'll need a stop, so the field's most common
        # starting compound is the honest stand-in.
        common_start = con.execute(
            """
            SELECT compound FROM gold.race_features
            WHERE race_id = ? AND lap_number = 1 AND compound IS NOT NULL
            GROUP BY compound ORDER BY COUNT(*) DESC LIMIT 1
            """,
            [race_id],
        ).fetchone()
    finally:
        con.close()

    if entries.empty:
        return []

    form = _season_form(season, str(race_date[0])) if race_date else {}
    quali = qualifying_gaps(race_id)
    default_compound = common_start[0] if common_start else "MEDIUM"

    rivals = []
    for row in entries.itertuples():
        grid = int(row.grid) if row.grid and int(row.grid) > 0 else 20
        rivals.append(
            RivalTrend(
                driver_id=row.driver_id,
                team_id=row.constructor_id,
                gap_to_leader=starting_gap_for_position(grid),
                # No prior races this season (opening round, or a debutant)
                # means no form to lean on — field-average pace is the
                # neutral assumption rather than a guess in either direction.
                recent_pace_delta=weekend_pace(_form_or_zero(form, row.driver_id), quali.get(row.driver_id)),
                compound=default_compound,
                tyre_age=0.0,
            )
        )
    return rivals


def _form_or_zero(form: dict[str, float], driver_id: str) -> float:
    value = form.get(driver_id, 0.0)
    return 0.0 if value is None or pd.isna(value) else float(value)


def driver_race_pace(race_id: str, driver_id: str) -> float:
    """One driver's expected race pace (race_plan/weekend_pace.py), from
    the same season form and qualifying the rivals' comes from.

    Used as the planner's pace anchor so both sides of the comparison come
    from one estimator. Field-average pace (0.0) stands in for form when
    there are no prior races this season to learn from — the neutral
    assumption rather than a flattering or pessimistic guess.
    """
    season, _ = (int(part) for part in race_id.split("_"))
    con = get_connection()
    try:
        race_date = con.execute("SELECT date FROM silver.races WHERE race_id = ?", [race_id]).fetchone()
    finally:
        con.close()
    form = _season_form(season, str(race_date[0])) if race_date else {}
    return weekend_pace(_form_or_zero(form, driver_id), qualifying_gaps(race_id).get(driver_id))


def stop_patterns_for_race(race_id: str) -> StopPatterns | None:
    """The real stop patterns the pre-race field is drawn from (see
    tyre_baselines.historical_stop_patterns), from races before this one —
    run or unrun, so it reads the calendar rather than the results."""
    con = get_connection()
    try:
        row = con.execute(
            """
            SELECT circuit_id, date FROM silver.calendar WHERE race_id = ?
            UNION ALL
            SELECT circuit_id, date FROM silver.races WHERE race_id = ?
            LIMIT 1
            """,
            [race_id, race_id],
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None
    return historical_stop_patterns(row[0], str(row[1]))
