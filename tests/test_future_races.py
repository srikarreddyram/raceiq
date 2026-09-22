"""Planning races that haven't happened — race_plan/future.py.

"Upcoming" is read from the calendar at test time, and the Open-Meteo
forecast is stubbed out, so these neither go stale as the season runs nor
touch the network.
"""

from __future__ import annotations

import pytest

import race_plan.future as future
from models.common.db import get_connection


@pytest.fixture(scope="module")
def upcoming():
    con = get_connection()
    try:
        row = con.execute(
            "SELECT race_id, circuit_id FROM silver.calendar WHERE NOT has_results ORDER BY season, round LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if row is None:
        pytest.skip("no upcoming round on the calendar")
    return row


@pytest.fixture(autouse=True)
def no_forecast(monkeypatch):
    monkeypatch.setattr(future, "_forecast", lambda *a, **k: None)


def test_calendar_holds_unrun_rounds(upcoming):
    con = get_connection()
    try:
        assert not future.has_race_data(con, upcoming[0])
        assert future.calendar_entry(con, upcoming[0])["circuit_id"] == upcoming[1]
    finally:
        con.close()


def test_circuit_priors_for_a_known_and_a_new_circuit():
    con = get_connection()
    try:
        baku = future.circuit_priors(con, "baku", 2026)
        new = future.circuit_priors(con, "nowhere_ring", 2026)
    finally:
        con.close()
    assert baku["prior_races"] >= 5 and baku["laps_known"] and baku["total_laps"] == 51
    assert 0 <= baku["sc_rate"] <= 1
    # A first race somewhere: no invented history, a stated assumption instead.
    assert new["prior_races"] == 0 and new["sc_rate"] is None and not new["laps_known"]
    assert new["dnf_rate"] is not None  # the era-wide retirement rate stands in


def test_conditions_typical_then_overridden():
    priors = {"prior_races": 3, "air_temp": 24.0, "track_temp": 36.0, "track_minus_air": 12.0, "rain_share": 0.34}
    entry = {"date": "2099-06-01", "time_utc": "13:00:00Z", "latitude": 0.0, "longitude": 0.0}
    typical = future.race_conditions(entry, priors)
    assert typical.source == "typical" and typical.track_temp == 36.0 and not typical.rain_expected
    manual = future.race_conditions(entry, priors, track_temp=50.0, rain=True)
    assert manual.source == "override" and manual.track_temp == 50.0 and manual.rain_expected


def test_grid_slots_fill_around_our_driver():
    order = {"a": 1, "b": 2, "c": 3, "d": 4}
    slots = future._grid_slots(order, ["a", "b", "c", "d"], "c", 1)
    assert slots == {"c": 1, "a": 2, "b": 3, "d": 4}


def test_an_upcoming_race_can_be_planned(upcoming):
    from race_plan.plan import build_race_plan

    con = get_connection()
    try:
        leader = con.execute(
            "SELECT driver_id FROM bronze.ergast_results WHERE season = ? GROUP BY driver_id ORDER BY SUM(points) DESC LIMIT 1",
            [int(upcoming[0].split("_")[0])],
        ).fetchone()[0]
    finally:
        con.close()
    plan = build_race_plan(upcoming[0], leader, n_simulations=300)
    assert plan.is_future and plan.grid_is_expected
    assert plan.conditions_source == "typical"  # forecast stubbed out
    assert plan.stops and all(s.window_open <= s.nominal_lap <= s.window_close for s in plan.stops)
    what_if = build_race_plan(upcoming[0], leader, n_simulations=300, grid_override=15)
    assert what_if.grid_position == 15 and not what_if.grid_is_expected
