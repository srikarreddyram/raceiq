"""The pre-race field: real stop patterns, the weekend pace estimate, and
the typical-strategy number the planner leads with."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import strategy_engine.simulation.monte_carlo as mc
from race_plan.weekend_pace import FORM_WEIGHT, FORM_WEIGHT_WITHOUT_QUALI, QUALI_CLIP_SECONDS, QUALI_WEIGHT, weekend_pace
from strategy_engine.field import RivalTrend
from strategy_engine.state import RaceState
from strategy_engine.tyre_baselines import StopPatterns, historical_stop_patterns
from strategy_engine.tyre_pace import field_plan_cost


def _state(**kw) -> RaceState:
    base = dict(
        circuit_id="hungaroring", team_id="t", driver_id="d", current_lap=0, race_total_laps=70,
        compound="MEDIUM", tyre_age=0.0, stint_number=1, current_position=10.0, gap_to_leader=10.0,
        gap_to_car_ahead=1.0, gap_to_car_behind=1.0, lap_time_seconds=80.0, field_avg_lap_time_seconds=80.0,
        degradation_rate=0.03, grip_estimate=0.0, air_temp=25.0, track_temp=35.0, humidity=50.0,
        wind_speed=1.0, wind_direction=0, rainfall_flag=False, driver_avg_pace_delta=0.0,
        driver_consistency_score=1.0, driver_overtaking_score=0.0, condition_delta=0.0,
        historical_sc_rate=0.3, historical_dnf_rate=0.1, historical_overtaking_rate=0.17,
        circuit_baseline_track_temp=35.0, car_profile={}, rival_ahead=None,
    )
    base.update(kw)
    return RaceState(**base)


def test_weekend_pace_uses_qualifying_when_it_has_run():
    assert weekend_pace(0.5, None) == pytest.approx(FORM_WEIGHT_WITHOUT_QUALI * 0.5)
    assert weekend_pace(0.5, 0.3) == pytest.approx(FORM_WEIGHT * 0.5 + QUALI_WEIGHT * 0.3)
    # A car that out-qualified its form is expected quicker than form alone says.
    assert weekend_pace(0.5, -0.5) < weekend_pace(0.5, None)
    assert QUALI_CLIP_SECONDS > 0


def test_sampled_stops_spread_across_the_field_and_skip_the_last_lap():
    patterns = StopPatterns(
        fractions=((0.3,), (0.25, 0.6), (0.99,)),  # 0.99 of 70 laps lands on the final lap
        compounds=(("MEDIUM", "HARD"), ("MEDIUM", "HARD", "HARD"), ("MEDIUM", "HARD")),
        source="test",
    )
    state = _state()
    out = mc.sample_rival_pit_laps(state, patterns, 500, 19, np.random.default_rng(1))
    assert out.shape == (500, 70, 19)
    assert not out[:, -1, :].any()  # no stop on the last lap
    stops_per_car = out.sum(axis=1)
    assert set(np.unique(stops_per_car)) <= {0, 1, 2}
    # Different cars get different patterns within one simulation.
    assert (stops_per_car.std(axis=1) > 0).mean() > 0.9


def test_field_plan_cost_is_the_patterns_average_cost():
    wear = {"SOFT": 0.06, "MEDIUM": 0.04, "HARD": 0.03}
    one = StopPatterns(fractions=((0.5,),), compounds=(("MEDIUM", "HARD"),), source="t")
    two = StopPatterns(fractions=((0.33, 0.66),), compounds=(("MEDIUM", "HARD", "HARD"),), source="t")
    # Fresher tyres on average cost less per lap.
    assert field_plan_cost(two, 60, wear) < field_plan_cost(one, 60, wear)


def test_rivals_get_the_safety_car_pit_discount_too():
    state = _state(race_total_laps=10)
    rivals = [RivalTrend("r", "t", 11.0, 0.0, "MEDIUM", 0.0)]
    rng = np.random.default_rng(0)
    shared = mc.build_shared_context(state, rivals, 4, rng, pre_race=True)
    shared = replace(
        shared,
        sc_occurs=np.ones_like(shared.sc_occurs),
        rival_pit=np.zeros((4, 10, 1), dtype=bool),
    )
    shared.rival_pit[:, 3, 0] = True
    f = mc._field(state, rivals, shared)
    assert f.pit[:, 3, 1] == pytest.approx(f.pit_loss_s * mc.SC_PIT_DISCOUNT)


def test_stop_patterns_leave_out_red_flag_races_and_keep_to_the_era():
    patterns = historical_stop_patterns("hungaroring", "2025-08-01")
    if patterns is None:
        pytest.skip("no Hungary history in this warehouse")
    assert len(patterns.fractions) >= 20
    assert "regulation era" in patterns.source or "every era" in patterns.source
    # Every car in a dry, red-flag-free race stops at least once.
    assert all(len(f) >= 1 for f in patterns.fractions)
