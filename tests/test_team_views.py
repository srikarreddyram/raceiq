"""Car Profile and Tyre views' backends — car_profiles/season_profile.py,
car_profiles/degradation_curves.py and models/tyre_degradation/trace.py.
Integration tests against the real warehouse, like the rest of tests/."""

from __future__ import annotations

import pytest

from car_profiles.degradation_curves import season_model, team_tyre_report
from car_profiles.season_profile import season_profile
from models.common.db import get_connection
from models.tyre_degradation.trace import remaining_life_trace


@pytest.mark.parametrize("season", range(2018, 2026))
def test_fuel_effect_is_measured_not_assumed(season):
    # Lap times fall ~0.05 s/lap as fuel burns and the track rubbers in —
    # the figure usually quoted for F1. A coefficient far from that means
    # the regression is confounding fuel with tyre wear.
    b = season_model(season)["fuel_track_seconds_per_lap"]
    assert -0.07 < b < -0.03


@pytest.mark.parametrize("season", [2022, 2023, 2024, 2025])
def test_softs_wear_faster_than_hards_once_fuel_is_removed(season):
    wear = season_model(season)["field_wear_seconds_per_lap"]
    assert wear["SOFT"] > wear["HARD"] > 0


def test_tyre_report_shape():
    report = team_tyre_report("mclaren", 2025)
    assert [c["compound"] for c in report["compounds"]] == ["SOFT", "MEDIUM", "HARD"]
    medium = report["compounds"][1]
    lo, hi = medium["wear_ci95"]
    assert lo <= medium["wear_s_per_lap"] <= hi
    assert all(p["laps"] >= 15 for p in medium["curve"])


def test_season_profile_intervals_and_counts():
    con = get_connection()
    try:
        profile = season_profile(con, "mclaren", 2025)
        races = con.execute(
            "SELECT COUNT(*) FROM gold.car_race_observations WHERE team_id = 'mclaren' AND season = 2025"
        ).fetchone()[0]
    finally:
        con.close()
    assert profile["races_observed"] == races
    assert profile["seeded_priors"] is None  # PRD 8.1 Layer 2 isn't ingested; never invented
    for c in profile["characteristics"]:
        assert c["n"] <= races
        assert len(c["trajectory"]) == races
        if c["ci95"] is not None:
            assert c["ci95"][0] <= c["value"] <= c["ci95"][1]
        if c["kind"] == "correlation" and c["value"] is not None:
            assert -1 <= c["value"] <= 1


def test_remaining_life_trace_uses_promoted_model():
    trace = remaining_life_trace("mclaren", 2025)
    assert trace["race_id"] == "2025_24"
    assert trace["laps"]
    assert all(lap["predicted_remaining"] >= -1 for lap in trace["laps"])
