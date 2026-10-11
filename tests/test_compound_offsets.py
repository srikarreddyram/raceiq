"""Compound pace offsets (car_profiles.degradation_curves.compound_offsets)
and how they reach the tyre cost (strategy_engine.tyre_pace)."""

from __future__ import annotations

import pytest

import strategy_engine.tyre_pace as tp
from car_profiles.degradation_curves import compound_offsets, regulation_era


def test_eras():
    assert [regulation_era(s) for s in (2018, 2021, 2022, 2025, 2026)] == ["2018-21", "2018-21", "2022-25", "2022-25", "2026"]


def test_softs_were_quicker_than_mediums_in_the_2022_cars():
    o = compound_offsets(2024)
    assert o["MEDIUM"] == 0.0
    assert -0.8 < o["SOFT"] < -0.1  # measured -0.41 (90%: -0.47 to -0.30)


def test_the_2018_21_estimate_isnt_used():
    assert compound_offsets(2019) == {"SOFT": 0.0, "MEDIUM": 0.0, "HARD": 0.0}


def test_offsets_survive_the_circuit_and_practice_scaling():
    wear = tp.race_wear("2025_14")
    assert isinstance(wear, tp.TyreRates)
    assert wear.offsets == compound_offsets(2025)
    # a new soft costs its offset; a medium at age 10 costs its wear
    assert tp.tyre_cost("SOFT", 0, wear) == pytest.approx(wear.offsets["SOFT"])
    assert tp.tyre_cost("MEDIUM", 10, wear) == pytest.approx(10 * wear["MEDIUM"])


def test_plain_wear_dicts_still_work():
    assert tp.tyre_cost("HARD", 5, {"HARD": 0.02}) == pytest.approx(0.1)


def test_a_thin_practice_sample_doesnt_set_the_field_s_wear():
    from race_plan.long_runs import MIN_FIELD_DEGRADATION_LAPS, field_degradation, practice_laps

    laps = practice_laps("2026_17")
    if laps.empty:
        pytest.skip("Singapore 2026 practice isn't in this warehouse")
    hard_push = int((laps["push"] & (laps["compound"] == "HARD")).sum())
    if hard_push >= MIN_FIELD_DEGRADATION_LAPS:
        pytest.skip("more practice has been ingested since")
    assert field_degradation("2026_17")["HARD"] is None
