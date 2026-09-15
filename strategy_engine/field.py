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

import pandas as pd

from models.common.db import get_connection


@dataclass
class RivalTrend:
    driver_id: str
    team_id: str
    gap_to_leader: float
    recent_pace_delta: float  # their pace_delta_this_lap at the snapshot — negative is faster than the field


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
                AVG(pace_delta_this_lap) AS recent_pace_delta
            FROM gold.race_features
            WHERE race_id = ? AND driver_id != ?
                AND lap_number BETWEEN ? AND ? AND NOT is_pit_lap
            GROUP BY driver_id, team_id
            """,
            [race_id, exclude_driver_id, lap_number - _TREND_WINDOW_LAPS + 1, lap_number],
        ).df()
    finally:
        con.close()

    return [
        RivalTrend(
            driver_id=r.driver_id,
            team_id=r.team_id,
            gap_to_leader=r.gap_to_leader,
            recent_pace_delta=r.recent_pace_delta if pd.notna(r.recent_pace_delta) else 0.0,
        )
        for r in rows.itertuples()
    ]
