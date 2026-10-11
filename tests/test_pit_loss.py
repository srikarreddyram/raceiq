"""What a pit stop costs (strategy_engine/pit_loss.py)."""

from __future__ import annotations

import strategy_engine.pit_loss as pit_loss


def test_a_stop_costs_less_than_the_time_in_the_pit_lane():
    stops = pit_loss._stop_costs()
    assert len(stops) > 1000
    assert 18 < stops["cost"].median() < 27  # measured 22.7 s
    # Silverstone: 29.9 s in the pit lane, about 20 s against staying out.
    assert pit_loss.typical_pit_loss_seconds("silverstone") < 24


def test_a_new_circuit_gets_the_cross_circuit_median():
    assert pit_loss.typical_pit_loss_seconds("nowhere") == pit_loss._FALLBACK_SECONDS


def test_a_plan_can_leave_out_the_race_it_is_planning():
    stops = pit_loss._stop_costs()
    only = stops["circuit_id"].value_counts()
    single = only[only > 0].index[0]
    # Excluding a race changes (or empties) that circuit's figure, never errors.
    race = stops[stops["circuit_id"] == single]["race_id"].iloc[0]
    assert isinstance(pit_loss.typical_pit_loss_seconds(single, exclude_race_id=race), float)
