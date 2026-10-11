"""What a pit stop costs at each circuit, in seconds — the time the
simulation charges a car on the lap it stops.

Measured as the stop's real cost: the in-lap plus the out-lap, minus two
of the same car's normal laps around them (laps 2-4 before the in-lap and
2-4 after the out-lap, green flag, no other stop). That is the time lost
against staying out, which is what a strategy trades.

It used to be the time from pit-lane entry to exit
(silver.pit_stops.stop_duration_seconds). That isn't the cost: a car that
stays out still has to cover that stretch of track, which takes several
seconds. Over 3,652 green-flag stops the real cost is 22.7 s at the
median circuit and the pit-lane time was 24.0 — two seconds too much per
stop on average, and far more at some tracks (Silverstone 20.1 against
29.9, Spa 18.6 against 23.8, Monaco 21.1 against 25.5). Every second of
that made an extra stop look worse than it is, which is the direction the
planner's stop counts were wrong in (about 1.2 stops a car against 1.6).
PIT_LOSS_FROM_LAP_TIMES = False restores the old figure, for the
validations to compare against.

`exclude_race_id` exists for race_plan/: a plan made before the start
can't know what stops cost in the race it's planning, and at a new
circuit that race is the only data — excluded, a new circuit falls
through to the cross-circuit median, which is the honest position to be
in before anyone has raced there.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from models.common.db import get_connection

PIT_LOSS_FROM_LAP_TIMES = True
_PLAUSIBLE_LOSS_RANGE = (5.0, 60.0)  # outside it: damage, a penalty served, a slow stop with a problem
_REFERENCE_LAPS = (2, 3, 4)  # laps before the in-lap / after the out-lap that set "normal" pace
_MIN_REFERENCE_LAPS = 4
_FALLBACK_SECONDS = 22.7  # the measured cross-circuit median, for a circuit with no stops on record


@lru_cache(maxsize=1)
def _stop_costs() -> pd.DataFrame:
    """One row per green-flag stop: circuit, race, and its real cost."""
    con = get_connection()
    try:
        laps = con.execute(
            """
            SELECT lf.race_id, r.circuit_id, lf.driver_id, lf.lap_number, lf.lap_time_seconds AS t, lf.stint_number,
                   lf.is_pit_lap AS pit,
                   (lf.safety_car_active OR lf.vsc_active OR lf.red_flag_active OR COALESCE(lf.yellow_active, FALSE)) AS caution
            FROM gold.lap_features lf JOIN silver.races r ON r.race_id = lf.race_id
            ORDER BY lf.race_id, lf.driver_id, lf.lap_number
            """
        ).df()
    finally:
        con.close()
    laps["pit"] = laps["pit"].fillna(False).astype(bool)
    laps["caution"] = laps["caution"].fillna(False).astype(bool)
    car = laps.groupby(["race_id", "driver_id"], sort=False)
    # An in-lap is the last lap of a stint (stints change only on a tyre
    # change, pipelines/silver/laps.py). It and the out-lap must be green.
    in_lap = (car["stint_number"].shift(-1) > laps["stint_number"]) & ~laps["caution"]
    out_t = car["t"].shift(-1)
    out_green = ~car["caution"].shift(-1, fill_value=True).astype(bool)
    # "Normal" pace: the car's green, non-pit laps 2-4 before the in-lap
    # and 2-4 after the out-lap.
    normal = laps["t"].where(~laps["pit"] & ~laps["caution"])
    refs = [car_shift for k in _REFERENCE_LAPS for car_shift in (normal.groupby([laps["race_id"], laps["driver_id"]]).shift(k),
                                                                 normal.groupby([laps["race_id"], laps["driver_id"]]).shift(-(k + 1)))]
    ref = pd.concat(refs, axis=1)
    cost = laps["t"] + out_t - 2 * ref.median(axis=1)
    keep = (
        in_lap & out_green & laps["t"].notna() & out_t.notna()
        & (ref.notna().sum(axis=1) >= _MIN_REFERENCE_LAPS) & cost.between(*_PLAUSIBLE_LOSS_RANGE)
    )
    return pd.DataFrame({"race_id": laps["race_id"], "circuit_id": laps["circuit_id"], "cost": cost})[keep].reset_index(drop=True)


@lru_cache(maxsize=None)
def _circuit_pit_loss(exclude_race_id: str | None = None) -> dict[str, float]:
    if PIT_LOSS_FROM_LAP_TIMES:
        stops = _stop_costs()
        if exclude_race_id:
            stops = stops[stops["race_id"] != exclude_race_id]
        return stops.groupby("circuit_id")["cost"].median().to_dict()
    con = get_connection()
    try:
        rows = con.execute(
            f"""
            SELECT r.circuit_id, AVG(ps.stop_duration_seconds) AS avg_loss
            FROM silver.pit_stops ps
            JOIN silver.races r ON r.race_id = ps.race_id
            WHERE ps.stop_duration_seconds BETWEEN 15 AND 60
                {"AND ps.race_id != ?" if exclude_race_id else ""}
            GROUP BY r.circuit_id
            """,
            [exclude_race_id] if exclude_race_id else [],
        ).df()
    finally:
        con.close()
    return {row.circuit_id: row.avg_loss for row in rows.itertuples()}


def typical_pit_loss_seconds(circuit_id: str, exclude_race_id: str | None = None) -> float:
    """What a stop costs at this circuit (module docstring)."""
    return float(_circuit_pit_loss(exclude_race_id).get(circuit_id, _FALLBACK_SECONDS))
