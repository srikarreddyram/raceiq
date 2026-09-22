"""Driver View backend — driver_profiles/profile.py, against the real warehouse."""

from __future__ import annotations

import pytest

from driver_profiles.profile import driver_profile, era_of, is_classified
from models.common.db import get_connection


@pytest.fixture(scope="module")
def con():
    c = get_connection()
    yield c
    c.close()


def test_classification_comes_from_status_not_position():
    # Ergast gives retirements a finishing-order position (a lap-3 DNF reads
    # P22); only the status says whether the driver was classified.
    assert is_classified("Finished") and is_classified("+1 Lap") and is_classified("Lapped")
    assert not is_classified("Retired") and not is_classified("Disqualified") and not is_classified(None)


def test_eras():
    assert [era_of(s) for s in (2018, 2021, 2022, 2025, 2026)] == ["2018-21", "2018-21", "2022-25", "2022-25", "2026+"]


def test_verstappen_2023_matches_the_record(con):
    # 19 wins from 22 races is the real 2023 record; race points exclude sprints.
    s = driver_profile(con, "max_verstappen", 2023)["summary"]
    assert s["races"] == 22 and s["wins"] == 19
    assert s["median_teammate_gap_s"] < 0  # faster than Perez


def test_retirements_excluded_from_average_finish(con):
    p = driver_profile(con, "leclerc", 2026)
    finishes = [r["position"] for r in p["races"] if r["classified"]]
    assert p["summary"]["not_classified"] == sum(not r["classified"] for r in p["races"])
    assert p["summary"]["avg_finish"] == pytest.approx(sum(finishes) / len(finishes), abs=0.05)


def test_teammate_gaps_are_antisymmetric(con):
    # Leclerc's gap to Hamilton on a shared race is Hamilton's gap to him, negated.
    lec = {r["race_id"]: r["teammate_gap_s"] for r in driver_profile(con, "leclerc", 2025)["races"]}
    ham = {r["race_id"]: r["teammate_gap_s"] for r in driver_profile(con, "hamilton", 2025)["races"]}
    shared = [k for k in lec if lec[k] is not None and ham.get(k) is not None]
    assert shared
    for k in shared:
        assert lec[k] == pytest.approx(-ham[k], abs=0.002)


def test_wet_split_is_null_without_enough_wet_laps(con):
    p = driver_profile(con, "leclerc", 2026)
    for split in p["wet_dry"]:
        if split["wet_laps"] < p["min_wet_laps"]:
            assert split["wet_teammate_gap_s"] is None
