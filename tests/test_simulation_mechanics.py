"""The lap-by-lap race (monte_carlo round 7) and the pace model feeding it
(strategy_engine/tyre_pace.py) — each behaviour here is one the real-outcome
validations depend on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategy_engine.simulation.monte_carlo as mc
from strategy_engine.field import RivalTrend
from strategy_engine.tyre_baselines import rival_stop_laps, typical_max_stint_length


def _race(start, pace, n_laps=30, delta=0.6, pit=None):
    n_sims, n_cars = 1, len(start)
    noise = np.zeros((n_sims, n_laps, n_cars))
    pit_loss = np.zeros((n_sims, n_laps, n_cars)) if pit is None else pit
    return mc.run_race(
        np.array(start, dtype=float),
        np.array([pace], dtype=float),
        noise,
        pit_loss,
        np.full((n_sims, n_cars), n_laps),
        np.zeros((n_sims, n_laps), dtype=bool),
        delta,
    )[0]


def test_a_slightly_faster_car_is_held_behind():
    # 0.3 s/lap quicker, but the circuit needs 0.6 s/lap to get past: it
    # finishes stuck behind, not 9 s up the road as a free-air projection says.
    times = _race(start=[0.0, 0.8], pace=[0.0, -0.3])
    assert times[1] > times[0]


def test_a_much_faster_car_gets_past():
    times = _race(start=[0.0, 0.8], pace=[0.0, -1.5])
    assert times[1] < times[0]


def test_following_in_dirty_air_costs_time():
    free = _race(start=[0.0, 5.0], pace=[0.0, 0.0])
    close = _race(start=[0.0, 0.8], pace=[0.0, 0.0])
    # Behind in clean air the gap holds; within a second it grows by the
    # measured dirty-air loss on every lap it stays there.
    assert free[1] - free[0] == pytest.approx(5.0)
    assert close[1] - close[0] > 0.8


def test_a_car_in_the_pits_drops_back_freely():
    pit = np.zeros((1, 30, 2))
    pit[0, 5, 0] = 22.0  # the leader stops on lap 6
    times = _race(start=[0.0, 0.8], pace=[0.0, 0.0], pit=pit)
    assert times[1] < times[0]


def test_retired_rivals_never_finish_ahead():
    n_laps = 20
    times = mc.run_race(
        np.array([5.0, 0.0]),
        np.zeros((1, 2)),
        np.zeros((1, n_laps, 2)),
        np.zeros((1, n_laps, 2)),
        np.array([[n_laps, 4]]),
        np.zeros((1, n_laps), dtype=bool),
        0.6,
    )[0]
    assert np.isinf(times[1]) and np.isfinite(times[0])


def test_passing_is_harder_where_history_says_so():
    assert mc.passing_delta_seconds(0.073) > mc.passing_delta_seconds(0.17) > mc.passing_delta_seconds(0.377)


def test_pace_errors_follow_the_measured_distribution():
    rng = np.random.default_rng(0)
    draws = mc.sample_pace_errors(rng, (200_000,), pre_race=True)
    table = np.array(mc.PACE_ERROR_PERCENTILES_PRE_RACE)
    assert draws.min() >= table[0] - 1e-9 and draws.max() <= table[-1] + 1e-9
    assert np.median(draws) == pytest.approx(table[49], abs=0.02)


def test_safety_cars_run_their_measured_length():
    # A caution that starts must last SC_PERIOD_LAPS laps, not one.
    onsets = np.zeros((1, 20), dtype=bool)
    onsets[0, 3] = True
    running = np.cumsum(onsets, axis=1)
    running[:, mc.SC_PERIOD_LAPS :] -= running[:, : -mc.SC_PERIOD_LAPS].copy()
    assert (running > 0).sum() == mc.SC_PERIOD_LAPS


def test_rivals_are_charged_every_stop_their_stints_need():
    # Tyres that run out immediately with a long race left need more than
    # one stop — the old rule charged at most one.
    stint = typical_max_stint_length("bahrain", "HARD")
    stops = rival_stop_laps("bahrain", "HARD", tyre_age=stint, n_laps=int(stint * 2.5))
    assert len(stops) >= 2
    assert rival_stop_laps("bahrain", "HARD", tyre_age=0, n_laps=5) == []


def test_our_car_and_an_identical_rival_get_identical_pace():
    # One estimator for both sides: same recent pace, same tyres, same plan
    # -> same projected pace. The asymmetry this replaced let a P10 car win
    # a third of simulations.
    from strategy_engine.engine import candidate_pace_overrides
    from strategy_engine.search.candidates import Strategy
    from strategy_engine.tyre_pace import tyre_adjusted_rivals
    from models.common.data import load_race_features
    from strategy_engine.state import RaceState

    df = load_race_features()
    rows = df[df.race_id == "2025_4"]  # conftest's verified scenario: Bahrain 2025, lap 20
    state = RaceState.from_gold_row(
        rows[(rows.lap_number == 20) & (rows.driver_id == "piastri")].iloc[0],
        race_total_laps=int(rows.lap_number.max()),
    )
    twin = RivalTrend("twin", state.team_id, state.gap_to_leader, -0.5, state.compound, state.tyre_age)
    (rival_pace,) = [r.recent_pace_delta for r in tyre_adjusted_rivals([twin], state, 2025)]
    stops = rival_stop_laps(state.circuit_id, state.compound, state.tyre_age, state.race_total_laps - state.current_lap)
    twin_plan = Strategy(
        pit_plan=tuple((state.current_lap + 1 + s, "HARD") for s in stops), label="twin's plan"
    )
    (our_pace,) = candidate_pace_overrides(state, [twin_plan], -0.5, 2025)
    assert our_pace == pytest.approx(rival_pace, abs=0.02)


def test_validation_counts_each_stop_once():
    # is_pit_lap flags the in-lap AND the out-lap; the validation must
    # charge one stop, not two.
    from strategy_engine.validate_in_race import _actual_plan

    laps = pd.DataFrame(
        {
            "driver_id": ["x"] * 6,
            "lap_number": [10, 11, 12, 13, 14, 15],
            "stint_number": [1, 1, 1, 2, 2, 2],
            "compound": ["SOFT", "SOFT", "SOFT", "HARD", "HARD", "HARD"],
            "is_pit_lap": [False, False, True, True, False, False],
        }
    )
    assert _actual_plan(laps, "x", snapshot_lap=10) == ((12, "HARD"),)
