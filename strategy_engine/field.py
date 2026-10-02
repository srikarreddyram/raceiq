"""The rest of the field, as a set of simple pace trends to project
forward — PRD Section 12.2's `rival_states`, extended beyond "the car
directly ahead" (all `RaceState` itself carries) to every other car still
running, since ranking a simulated strategy against the field needs
everyone, not just one rival.

Each rival's future pace is projected as their *current* trend continuing
(`recent_pace_delta`, i.e. their `pace_delta_this_lap` at the snapshot
lap) rather than by simulating their own strategic decisions — modeling
every rival's own tyre choices and pit timing would mean running this
same engine recursively for 19 other cars, each needing their own
CarProfile-equivalent context. That's out of scope here; this projects
"how is this car trending right now," which is a reasonable approximation
over the short-to-medium term a strategy call actually needs, and is
honest about not capturing a rival's own future pit stops.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.db import get_connection

# What a car's pace will be for the rest of the race, estimated mid-race.
#
# This was the car's mean pace delta over its last five laps — any five
# laps, safety-car ones included. Measured against what cars then did
# (green-flag pace over the remaining laps, every dry race, at 40%
# distance) that is the worst estimate available:
#
#                                   2025     2026    (MAE, s/lap)
#     last 5 laps (was in use)      0.567    0.609
#     last 5 green laps             0.413    0.435
#     race so far, green laps       0.322    0.360
#     pre-race weekend estimate     0.336    0.441   (form + qualifying)
#     race so far + weekend, fitted 0.289    0.352
#
# Five laps is mostly traffic, tyre state and a leader managing a gap:
# cars running P1-5 came out 0.3 s/lap slower than they went on to be, so
# the simulation had leaders falling back through the pit stops. The
# estimate is now a blend of the car's green-flag pace over the whole race
# so far and the pre-race weekend estimate (race_plan/weekend_pace.py).
# The weights were fitted on 2018-2024 and move with race distance — the
# more of the race has been run, the more it says:
#
#     race fraction    0.25   0.40   0.55   0.70
#     race so far      0.44   0.53   0.59   0.63
#     weekend          0.50   0.42   0.36   0.32
#
# which the two lines below reproduce. Both earlier regulation eras give
# the same weights (2018-21: 0.54 / 0.40, 2022-24: 0.53 / 0.43 pooled).
# They sum to about 0.94: any estimate overstates how far a car is from
# the field. REST_OF_RACE_PACE = False restores the five-lap figure, for
# the validation to compare against.
REST_OF_RACE_PACE = True
MIN_GREEN_LAPS = 3  # fewer green laps so far than this and the race says nothing yet
MAX_PLAUSIBLE_PACE_DELTA = 5.0  # s/lap; beyond it a lap is damage or a spin, not pace
WET_COMPOUNDS = ("INTERMEDIATE", "WET")


def _blend_weights(race_fraction: float) -> tuple[float, float]:
    f = float(np.clip(race_fraction, 0.1, 0.9))
    return 0.34 + 0.42 * f, 0.60 - 0.40 * f


@lru_cache(maxsize=256)
def rest_of_race_pace(race_id: str, lap_number: int) -> dict[str, float]:
    """Every driver's expected pace delta for the rest of the race, as it
    looks at the end of `lap_number` — see REST_OF_RACE_PACE."""
    # Imported here: race_plan.field imports this module.
    from race_plan.field import _form_or_zero, _season_form
    from race_plan.weekend_pace import qualifying_gaps, weekend_pace

    con = get_connection()
    try:
        laps = con.execute(
            """
            SELECT driver_id, lap_number, pace_delta_this_lap AS pace, compound,
                   (NOT is_pit_lap AND NOT safety_car_active AND NOT vsc_active AND NOT red_flag_active
                    AND NOT COALESCE(yellow_active, FALSE) AND lap_number >= 2
                    AND pace_delta_this_lap IS NOT NULL AND ABS(pace_delta_this_lap) < ?) AS usable
            FROM gold.race_features
            WHERE race_id = ? AND lap_number <= ?
            """,
            [MAX_PLAUSIBLE_PACE_DELTA, race_id, lap_number],
        ).df()
        meta = con.execute(
            """
            SELECT r.date, (SELECT MAX(lap_number) FROM gold.race_features WHERE race_id = r.race_id) AS total_laps
            FROM silver.races r WHERE r.race_id = ?
            """,
            [race_id],
        ).fetchone()
        drivers = con.execute("SELECT DISTINCT driver_id FROM gold.race_features WHERE race_id = ?", [race_id]).df()
    finally:
        con.close()
    if meta is None:
        return {}

    form = _season_form(int(race_id.split("_")[0]), str(meta[0]))
    quali = qualifying_gaps(race_id)
    w_race, w_weekend = _blend_weights(lap_number / meta[1] if meta[1] else 0.4)
    # Only laps on the kind of tyre the car is on now: pace on intermediates
    # says nothing about pace on slicks once the track has dried.
    laps["wet_tyres"] = laps["compound"].isin(WET_COMPOUNDS)
    now_wet = laps.sort_values("lap_number").groupby("driver_id")["wet_tyres"].last()
    same_tyres = laps[laps["usable"] & (laps["wet_tyres"] == laps["driver_id"].map(now_wet))]
    so_far = same_tyres.groupby("driver_id")["pace"].agg(["mean", "size"])
    race_pace = {driver_id: float(r["mean"]) for driver_id, r in so_far.iterrows() if r["size"] >= MIN_GREEN_LAPS}
    out = {}
    for driver_id in drivers["driver_id"]:
        weekend = weekend_pace(_form_or_zero(form, driver_id), quali.get(driver_id))
        if driver_id in race_pace:
            out[driver_id] = w_race * race_pace[driver_id] + w_weekend * weekend
        else:
            out[driver_id] = weekend  # nothing green to go on yet: the pre-race estimate as it stands
    return out


@dataclass
class RivalTrend:
    driver_id: str
    team_id: str
    gap_to_leader: float
    # Expected pace for the rest of the race, s/lap against the field
    # (negative is faster) — see REST_OF_RACE_PACE. The name is from when
    # this was the last five laps' pace.
    recent_pace_delta: float
    compound: str
    tyre_age: float  # at the snapshot lap — used to decide whether this rival still owes a future pit stop


_TREND_WINDOW_LAPS = 5


def build_field_snapshot_from_gold(race_id: str, lap_number: int, exclude_driver_id: str) -> list[RivalTrend]:
    """For testing/demo against a real historical race — pulls every other
    classified driver's actual state at that lap. A live API caller would
    instead supply this list directly from telemetry, matching the PRD's
    `rival_states` input.

    `recent_pace_delta` averages `pace_delta_this_lap` over the last few
    laps rather than reading a single lap — a single lap's pace delta is
    noisy (+/-1-2s is normal lap-to-lap variation), and this gets
    extrapolated across dozens of remaining laps in the simulation, so
    feeding it un-smoothed would massively amplify one noisy sample into
    an absurd projected pace advantage or deficit.
    """
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT
                driver_id,
                team_id,
                LAST(gap_to_leader ORDER BY lap_number) AS gap_to_leader,
                AVG(pace_delta_this_lap) AS recent_pace_delta,
                LAST(compound ORDER BY lap_number) AS compound,
                LAST(tyre_age ORDER BY lap_number) AS tyre_age
            FROM gold.race_features
            WHERE race_id = ? AND driver_id != ?
                AND lap_number BETWEEN ? AND ? AND NOT is_pit_lap
            GROUP BY driver_id, team_id
            """,
            [race_id, exclude_driver_id, lap_number - _TREND_WINDOW_LAPS + 1, lap_number],
        ).df()
    finally:
        con.close()

    blended = rest_of_race_pace(race_id, lap_number) if REST_OF_RACE_PACE else {}
    return [
        RivalTrend(
            driver_id=r.driver_id,
            team_id=r.team_id,
            gap_to_leader=r.gap_to_leader,
            recent_pace_delta=blended.get(r.driver_id, r.recent_pace_delta if pd.notna(r.recent_pace_delta) else 0.0),
            compound=r.compound,
            tyre_age=float(r.tyre_age),
        )
        for r in rows.itertuples()
    ]


def driver_recent_pace(race_id: str, lap_number: int, driver_id: str) -> float | None:
    """One driver's expected pace for the rest of the race, measured
    exactly as every rival's `recent_pace_delta` is (REST_OF_RACE_PACE).
    The engine anchors our own car's pace to this so both sides of the
    comparison come from one estimator (see
    engine.candidate_pace_overrides)."""
    if REST_OF_RACE_PACE:
        blended = rest_of_race_pace(race_id, lap_number)
        if driver_id in blended:
            return blended[driver_id]
    con = get_connection()
    try:
        row = con.execute(
            """
            SELECT AVG(pace_delta_this_lap)
            FROM gold.race_features
            WHERE race_id = ? AND driver_id = ? AND lap_number BETWEEN ? AND ? AND NOT is_pit_lap
            """,
            [race_id, driver_id, lap_number - _TREND_WINDOW_LAPS + 1, lap_number],
        ).fetchone()
    finally:
        con.close()
    return None if row is None or row[0] is None or pd.isna(row[0]) else float(row[0])
