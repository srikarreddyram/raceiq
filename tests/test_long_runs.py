"""Practice long runs (race_plan/long_runs.py) and how they reach the plan."""

from __future__ import annotations

import pytest

import strategy_engine.tyre_pace as tyre_pace
from race_plan import long_runs


def _sepang():
    df = long_runs.summary("2026_16", "FP2")
    if df.empty:
        pytest.skip("Sepang 2026 practice isn't in this warehouse")
    return df.set_index("code")


def test_push_laps_reproduce_a_published_long_run_analysis():
    # FDataAnalysis's FP2 long-run breakdown of the 2026 race at Sepang.
    df = _sepang()
    for code, laps, mean in [("ANT", 11, 104.480), ("LEC", 7, 104.800), ("VER", 13, 104.808), ("STR", 5, 106.953), ("COL", 20, 105.580)]:
        assert df.loc[code, "push_laps"] == laps
        assert df.loc[code, "mean_lap"] == pytest.approx(mean, abs=0.001)
    assert df["gap"].min() == 0.0 and df.index[0] == "ANT"


def test_degradation_tells_friday_story():
    # Ferrari off on softs, McLaren barely: the story the published analysis told.
    df = _sepang()
    assert df.loc["LEC", "deg_soft"] > df.loc["NOR", "deg_soft"]
    assert df.loc["HAM", "deg_soft"] > df.loc["PIA", "deg_soft"]


def test_the_plan_blends_practice_into_tyre_wear():
    history = tyre_pace.history_wear("2026_16")
    practice = long_runs.field_degradation("2026_16")
    used = tyre_pace.race_wear("2026_16")
    for compound, value in practice.items():
        if value is None:
            assert used[compound] == pytest.approx(history[compound])  # nothing to blend: history alone
        else:
            expected = max(0.0, tyre_pace.PRACTICE_WEIGHT * value + tyre_pace.HISTORY_WEIGHT * history[compound])
            assert used[compound] == pytest.approx(expected)


def test_a_weekend_without_practice_uses_history():
    assert all(v is None for v in long_runs.field_degradation("2018_1").values())
    assert tyre_pace.race_wear("2018_1") == tyre_pace.history_wear("2018_1")
