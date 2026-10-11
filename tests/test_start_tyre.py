"""How the planner picks the tyre to start on (race_plan.plan.choose_start_tyre)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from race_plan.plan import START_TIE_TOLERANCE, choose_start_tyre
from strategy_engine.tyre_baselines import StopPatterns


def _results(**scores):
    return {c: (SimpleNamespace(strategy_score=s), []) for c, s in scores.items()}


PATTERNS = StopPatterns(
    fractions=((0.4,),) * 10,
    compounds=(("MEDIUM", "HARD"),) * 7 + (("SOFT", "HARD"),) * 2 + (("HARD", "MEDIUM"),),
    source="test",
)


def test_a_clear_winner_in_the_simulation_is_the_plan():
    start, basis, share = choose_start_tyre(_results(HARD=0.60, MEDIUM=0.50, SOFT=0.40), PATTERNS)
    assert (start, basis, share) == ("HARD", "simulation", None)


def test_a_tie_goes_to_the_most_common_start_here():
    top = 0.50
    within = top * (1 - START_TIE_TOLERANCE / 2)
    start, basis, share = choose_start_tyre(_results(HARD=top, MEDIUM=within, SOFT=0.30), PATTERNS)
    assert start == "MEDIUM" and basis == "most common start here"
    assert share == pytest.approx(0.7)


def test_only_tied_tyres_are_eligible():
    # Soft is the second most common start but out of the tie: not picked.
    start, _, _ = choose_start_tyre(_results(HARD=0.50, SOFT=0.20, MEDIUM=0.30), PATTERNS)
    assert start == "HARD"


def test_no_history_leaves_the_simulation_s_pick():
    start, basis, _ = choose_start_tyre(_results(HARD=0.50, MEDIUM=0.499), None)
    assert (start, basis) == ("HARD", "simulation")
