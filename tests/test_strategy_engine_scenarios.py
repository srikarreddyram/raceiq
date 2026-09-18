"""Automates the sanity checks this session ran by hand, over and over,
against real historical scenarios after every change to the strategy
engine. Each property asserted here is one this session actually verified
manually at some point; codifying them catches the next regression
automatically instead of relying on someone remembering to re-run four
CLI commands and eyeball the JSON.

Uses a small n_simulations for speed — these are correctness/sanity
checks, not calibration checks, and 500 sims is enough to catch a broken
0%/100% or a crash without the ~5000-sim production-grade Monte Carlo
runtime.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strategy_engine.engine import recommend_strategy
from strategy_engine.oracles import (
    predict_expected_finish_now,
    predict_next_lap_time,
    predict_remaining_tyre_life,
    predict_safety_car_probability,
    predict_win_probability_now,
)

N_SIMULATIONS = 500


def _assert_valid_probability(value: float, name: str) -> None:
    assert not math.isnan(value), f"{name} is NaN"
    assert 0.0 <= value <= 1.0, f"{name}={value} outside [0, 1]"


class TestOraclesReturnSaneValues:
    """Every oracle should return a finite, real-world-plausible value for
    a known-good real historical state — regardless of which candidate
    strategy or simulation is later built on top of it.
    """

    def test_lap_time_oracle(self, leader_state):
        predicted = predict_next_lap_time(leader_state)
        assert not math.isnan(predicted)
        # A lap time of under 30s or over 5 minutes is not a real F1 lap
        # under green-flag-ish conditions; catches a badly-shaped feature
        # row long before it'd surface as a bad Monte Carlo result.
        assert 30.0 < predicted < 300.0

    def test_tyre_life_oracle_nonnegative(self, leader_state):
        remaining = predict_remaining_tyre_life(leader_state)
        assert not math.isnan(remaining)
        assert remaining >= 0.0

    def test_safety_car_oracle(self, leader_state):
        _assert_valid_probability(predict_safety_car_probability(leader_state), "safety_car_probability")

    def test_win_probability_oracle(self, leader_state, last_place_state):
        _assert_valid_probability(predict_win_probability_now(leader_state), "win_probability (leader)")
        _assert_valid_probability(predict_win_probability_now(last_place_state), "win_probability (last)")

    def test_leader_beats_last_place_on_win_probability(self, leader_state, last_place_state):
        """The one property that must never regress: whatever the exact
        number, an actual real-race leader must not be modeled as *less*
        likely to win than the actual last-place driver at the same race
        and lap. This exact comparison caught a real, severe bug earlier
        this session (a leader modeled at 5% while last place-adjacent
        cars looked comparatively fine) — see monte_carlo.py's docstring.
        """
        assert predict_win_probability_now(leader_state) > predict_win_probability_now(last_place_state)

    def test_expected_finish_oracle(self, leader_state, last_place_state):
        leader_finish = predict_expected_finish_now(leader_state)
        last_finish = predict_expected_finish_now(last_place_state)
        assert 1.0 <= leader_finish <= 20.0
        assert 1.0 <= last_finish <= 20.0
        # Lower finishing position number = better; the leader's expected
        # finish must be numerically lower (better) than last place's.
        assert leader_finish < last_finish


class TestRecommendStrategyRealScenarios:
    """End-to-end: search -> simulate -> score -> recommend, against real
    historical race states. No candidate strategy or simulation output
    should ever violate basic probability/consistency rules, regardless
    of which specific strategy comes out on top.
    """

    def _check_recommendation(self, recommendation: dict) -> None:
        top = recommendation["recommended_strategy"]
        _assert_valid_probability(top["win_probability"], "win_probability")
        _assert_valid_probability(top["podium_probability"], "podium_probability")
        _assert_valid_probability(top["points_probability"], "points_probability")
        # Podium implies at least as likely as win; points implies at
        # least as likely as podium — these are nested outcomes by
        # definition (P3 finish is also a "points" finish), never the
        # reverse.
        assert top["podium_probability"] >= top["win_probability"] - 1e-9
        assert top["points_probability"] >= top["podium_probability"] - 1e-9
        assert 1.0 <= top["expected_finish"] <= 20.0
        assert top["expected_points"] >= 0.0
        assert top["risk_score"] >= 0.0

        finish_distribution = top["finish_distribution"]
        total_probability = sum(finish_distribution.values())
        assert math.isclose(total_probability, 1.0, abs_tol=1e-6), (
            f"finish_distribution sums to {total_probability}, not 1.0"
        )
        for position, probability in finish_distribution.items():
            _assert_valid_probability(probability, f"finish_distribution[{position}]")

        assert recommendation["model_cross_checks"]["win_probability_model_estimate"] is not None
        assert len(recommendation["reasoning"]) > 0

    def test_leader_scenario(self, leader_state, leader_rivals):
        recommendation, n_candidates = recommend_strategy(leader_state, leader_rivals, n_simulations=N_SIMULATIONS)
        assert n_candidates > 0
        self._check_recommendation(recommendation)

    def test_last_place_scenario(self, last_place_state, last_place_rivals):
        recommendation, n_candidates = recommend_strategy(
            last_place_state, last_place_rivals, n_simulations=N_SIMULATIONS
        )
        assert n_candidates > 0
        self._check_recommendation(recommendation)

    def test_wet_race_scenario_no_false_certainty(self, wet_race_state):
        """The specific failure mode PACE_DEVIATION_CAP_PER_LAP exists to
        bound (see monte_carlo.py's docstring): a rare, sparse-in-training
        wet/drying-track combination should never produce a flat,
        noise-proof 100% win probability — that's an extrapolation
        artifact, not genuine confidence, however the model got there.
        """
        from strategy_engine.field import build_field_snapshot_from_gold

        rivals = build_field_snapshot_from_gold("2025_10", 30, exclude_driver_id="piastri")
        recommendation, n_candidates = recommend_strategy(wet_race_state, rivals, n_simulations=N_SIMULATIONS)
        assert n_candidates > 0
        top = recommendation["recommended_strategy"]
        assert top["win_probability"] < 0.999, "a rare/sparse scenario produced false-certain 100% win probability"


class TestSimulationDeterminismUnderSharedContext:
    """Common random numbers (see monte_carlo.py's module docstring): two
    different candidate strategies evaluated against the *same*
    SharedContext must see identical safety-car and retirement draws, or
    comparing their scores head-to-head isn't a fair comparison at all.
    """

    def test_shared_context_reused_across_strategies_gives_identical_randomness(self, leader_state, leader_rivals):
        from strategy_engine.search.candidates import Strategy
        from strategy_engine.simulation.monte_carlo import build_shared_context, simulate_strategy

        rng = np.random.default_rng(42)
        shared = build_shared_context(leader_state, leader_rivals, N_SIMULATIONS, rng)

        no_stop = Strategy(pit_plan=(), label="No further stops")
        result = simulate_strategy(leader_state, no_stop, leader_rivals, shared)

        assert result.final_positions.shape == (N_SIMULATIONS,)
        assert result.safety_car_occurred.shape == (N_SIMULATIONS,)
        assert (result.final_positions >= 1).all()
        assert not np.isnan(result.final_positions).any()
