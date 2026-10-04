"""Round 8 of the race simulation: the pit-lane fix and per-circuit tyre
wear, which are in use, and tyres tracked lap by lap and covers, which
were measured and switched off but must still do what they say."""

from __future__ import annotations

import numpy as np
import pytest

import strategy_engine.simulation.monte_carlo as mc
from car_profiles.degradation_curves import circuit_wear_factor
from strategy_engine import tyre_pace

N_LAPS = 30
PIT_LOSS = 20.0


def _wear(rate: float, n_cars: int) -> np.ndarray:
    return np.full((1, n_cars, mc.MAX_STINTS), rate)


def _race(start, pit, wear_rate=None, delta=1.3):
    n_cars = len(start)
    return mc.run_race(
        np.array(start, dtype=float),
        np.zeros((1, n_cars)),
        np.zeros((1, N_LAPS, n_cars)),
        pit,
        np.full((1, n_cars), N_LAPS),
        np.zeros((1, N_LAPS), dtype=bool),
        delta,
        tyre_wear=None if wear_rate is None else _wear(wear_rate, n_cars),
        start_age=np.zeros(n_cars),
    )[0]


def test_a_cars_race_time_carries_its_tyre_ages():
    # Two cars a minute apart, so neither affects the other. One stops.
    pit = np.zeros((1, N_LAPS, 2))
    pit[0, 9, 1] = PIT_LOSS
    times = _race([0.0, 60.0], pit, wear_rate=0.1)
    cost = mc.scheduled_tyre_cost(pit, _wear(0.1, 2), np.zeros(2))[0] * N_LAPS
    assert times[0] == pytest.approx(cost[0])
    assert times[1] == pytest.approx(60.0 + PIT_LOSS + cost[1])
    # The stop bought fresher tyres: less tyre cost than running to the flag.
    assert cost[1] < cost[0]


def test_an_undercut_works_only_when_tyres_wear():
    # B starts 2 s behind A and stops two laps earlier.
    pit = np.zeros((1, N_LAPS, 2))
    pit[0, 11, 0] = PIT_LOSS  # A
    pit[0, 9, 1] = PIT_LOSS  # B
    flat = _race([0.0, 2.0], pit)
    assert flat[1] > flat[0]  # one constant pace per car: no undercut
    worn = _race([0.0, 2.0], pit, wear_rate=0.15)
    assert worn[1] < worn[0]  # two laps on fresh tyres against old ones


def _cover_case(draw: float):
    # Road order: car 2 leads, car 1 is 1 s behind it, car 0 (ours) 1 s further back.
    n_cars = 3
    pit = np.zeros((1, N_LAPS, n_cars), dtype=np.float32)
    pit[0, 5, 1] = PIT_LOSS  # car 1 stops this lap
    pit[0, 8, 2] = PIT_LOSS  # car 2's stop is due in three laps
    pit[0, 8, 0] = PIT_LOSS  # so is ours
    order = np.array([[2, 1, 0]])
    covers = mc.Covers(
        draws=np.full((1, N_LAPS, n_cars), draw, dtype=np.float32),
        can_react=np.array([False, True, True]),
        pit_loss_s=PIT_LOSS,
    )
    mc._apply_covers(
        5, order, np.array([[0.0, 1.0, 2.0]]), np.ones((1, n_cars), dtype=bool), pit, np.zeros((1, N_LAPS), dtype=bool), covers
    )
    return pit


def test_a_rival_covers_a_stop_behind_it_and_our_car_keeps_its_plan():
    pit = _cover_case(draw=0.0)
    assert pit[0, 6, 2] == PIT_LOSS and pit[0, 8, 2] == 0.0  # pulled forward to the next lap
    assert pit[0, 8, 0] == PIT_LOSS and pit[0, 6, 0] == 0.0  # ours stays where the plan put it


def test_no_cover_when_the_draw_says_no():
    pit = _cover_case(draw=0.99)
    assert pit[0, 8, 2] == PIT_LOSS and pit[0, 6, 2] == 0.0


def test_the_production_model_keeps_only_what_validated():
    # Built, measured, and switched off — see monte_carlo.PIT_LANE_BLOCKS.
    model = mc.race_model(2025)
    assert model["covers"] is False and model["wear"] is None
    assert mc.PIT_LANE_BLOCKS is False and tyre_pace.CIRCUIT_WEAR is True


def test_a_car_in_the_pit_lane_holds_nobody_up():
    # B is 19 s behind A when A stops; A rejoins 0.7 s behind B. B hasn't
    # "passed" A on track and needs no passing margin to stay ahead.
    pit = np.zeros((1, N_LAPS, 2))
    pit[0, 5, 0] = PIT_LOSS
    times = _race([0.0, 19.3], pit)
    assert times[1] < times[0]


def test_circuit_wear_factor_is_neutral_without_history_and_bounded_with_it():
    assert circuit_wear_factor("no_such_circuit", "2025-06-01") == 1.0
    factor = circuit_wear_factor("bahrain", "2025-04-10")
    assert 0.25 <= factor <= 3.0


def test_bahrain_is_harder_on_tyres_than_jeddah():
    bahrain, jeddah = circuit_wear_factor("bahrain", "2025-04-10"), circuit_wear_factor("jeddah", "2025-04-20")
    if bahrain == 1.0 or jeddah == 1.0:
        pytest.skip("no wear history for these circuits in this warehouse")
    assert bahrain > 1.0 > jeddah


def test_race_wear_scales_the_seasons_rates_by_the_circuit():
    season = tyre_pace.wear_rates(2025)
    race = tyre_pace.history_wear("2025_4")  # Bahrain; race_wear also blends in practice (tests/test_long_runs.py)
    ratios = {c: race[c] / season[c] for c in season if season[c] > 0}
    assert len(set(round(r, 6) for r in ratios.values())) == 1  # one factor for every compound
