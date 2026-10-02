"""A stint changes only when the tyres do (pipelines/silver/laps.py)."""

from __future__ import annotations

import pandas as pd

from pipelines.silver.laps import _real_stint_numbers


def _laps(rows):
    df = pd.DataFrame(rows, columns=["lap_number", "stint_number", "compound", "tyre_age_laps"])
    df["race_id"], df["driver"] = "2025_10", "VER"
    df["stint_number"] = df["stint_number"].astype("Int64")
    return df


def test_pit_lane_passages_on_the_same_tyres_are_not_new_stints():
    # A real stop after lap 2, then two laps through the pit lane behind
    # the safety car on the same hards.
    laps = _laps(
        [
            (1, 1, "MEDIUM", 1.0),
            (2, 1, "MEDIUM", 2.0),
            (3, 2, "HARD", 1.0),
            (4, 2, "HARD", 2.0),
            (5, 3, "HARD", 3.0),
            (6, 4, "HARD", 4.0),
        ]
    )
    assert _real_stint_numbers(laps).tolist() == [1, 1, 2, 2, 2, 2]


def test_a_stop_onto_the_same_compound_is_still_a_stop():
    laps = _laps([(1, 1, "HARD", 20.0), (2, 2, "HARD", 1.0), (3, 2, "HARD", 2.0)])
    assert _real_stint_numbers(laps).tolist() == [1, 2, 2]


def test_an_unknown_tyre_age_keeps_the_recorded_change():
    laps = _laps([(1, 1, "HARD", 5.0), (2, 2, "HARD", None), (3, 2, "HARD", None)])
    assert _real_stint_numbers(laps).tolist() == [1, 2, 2]


def test_each_car_counts_its_own_stints():
    a = _laps([(1, 1, "SOFT", 1.0), (2, 2, "HARD", 1.0)])
    b = _laps([(1, 1, "SOFT", 1.0), (2, 1, "SOFT", 2.0)])
    b["driver"] = "NOR"
    both = pd.concat([a, b], ignore_index=True)
    assert _real_stint_numbers(both).tolist() == [1, 2, 1, 1]
