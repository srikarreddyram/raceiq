"""The race weekend planner's API-facing pieces and the car speed profile."""

from __future__ import annotations

import pytest

from car_profiles.speed_profile import speed_profile
from models.common.db import get_connection


@pytest.fixture(scope="module")
def con():
    c = get_connection()
    yield c
    c.close()


def test_speed_profile_2025_reads_like_the_season(con):
    p = speed_profile(con, 2025)
    teams = {t["team_id"]: t for t in p["teams"]}
    # McLaren was the fastest car of 2025 without being quick in a straight
    # line — its time came in the corners, and the reading should say so.
    assert p["teams"][0]["team_id"] == "mclaren"
    assert teams["mclaren"]["top_speed_delta_kph"] < teams["williams"]["top_speed_delta_kph"]
    assert "corners" in teams["mclaren"]["reading"]


def test_speed_profile_flags_2026_regulations(con):
    reading = speed_profile(con, 2026)["teams"][0]["reading"]
    assert "2026 cars" in reading and "no DRS" in reading


def test_speed_profile_has_no_data_before_2023(con):
    with pytest.raises(LookupError):
        speed_profile(con, 2021)


def test_planner_grid_override_changes_only_the_start():
    from race_plan.plan import _starting_state

    real, _ = _starting_state("2026_14", "leclerc")
    what_if, _ = _starting_state("2026_14", "leclerc", grid_override=12)
    assert what_if.current_position == 12
    assert what_if.gap_to_leader > real.gap_to_leader
    assert what_if.circuit_id == real.circuit_id


def test_standings_are_race_points_ordered(con):
    from serving.api.routers.planner import get_standings

    s = get_standings(season=2025, con=con)
    points = [d.points for d in s.drivers]
    assert points == sorted(points, reverse=True)
    assert s.drivers[0].name and " " in s.drivers[0].name  # a name, not a driver_id
