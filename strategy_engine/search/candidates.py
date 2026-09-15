"""Strategy candidate enumeration — PRD Section 12.3.

A strategy is a sequence of `(pit_lap, compound)` pairs from the current
lap to race end. This bounds the search to 0, 1, or 2 additional stops
from the current state — real strategies beyond 2 further stops from any
mid-race point are rare enough in practice that enumerating them isn't
worth the combinatorial blow-up, and PRD 12.3's own filters (tyre life
feasibility, compound rules) would reject almost all of them anyway.

Feasibility filtering uses the Tyre Degradation model as PRD 12.3
specifies ("filtered by tyre life feasibility"), but only *here*, at
search time, to bound candidate stint lengths — not inside the Monte
Carlo simulation loop itself (see simulation/monte_carlo.py's docstring
for why: a candidate's pit-lap/compound sequence is fixed by definition,
so tyre state along it is deterministic, and re-predicting remaining life
lap-by-lap inside the simulation would just re-derive the same feasibility
bound at a much higher cost for 5,000 simulations).

Two PRD 12.3 filters aren't implemented: the Monaco/Singapore-style
"track position penalty" and the `downforce_circuit_match` penalty both
depend on circuit geometry and team car profiles (`track_maps/`,
`car_profiles/`, neither built) — scoring, not just search, would need
those, so they're deferred to when those subsystems exist rather than
approximated with a guess here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations, product

from strategy_engine.oracles import predict_remaining_tyre_life
from strategy_engine.state import RaceState
from strategy_engine.tyre_baselines import typical_degradation_rate

FEASIBILITY_TOLERANCE = 0.15  # the tyre model's own measured MAE is several laps; don't over-reject on a point estimate


@dataclass(frozen=True)
class Strategy:
    pit_plan: tuple[tuple[int, str], ...]  # ((pit_lap, new_compound), ...), in lap order
    label: str


def _stint_feasible(state: RaceState, compound: str, stint_length: int, is_current_stint: bool) -> bool:
    if is_current_stint:
        remaining_life = predict_remaining_tyre_life(state)
    else:
        fresh = replace(
            state,
            compound=compound,
            tyre_age=1.0,
            stint_number=state.stint_number + 1,
            degradation_rate=typical_degradation_rate(state.circuit_id, compound),
            grip_estimate=0.0,
        )
        remaining_life = predict_remaining_tyre_life(fresh)
    return stint_length <= remaining_life * (1 + FEASIBILITY_TOLERANCE)


def _uses_two_compounds(state: RaceState, plan: tuple[tuple[int, str], ...]) -> bool:
    compounds_used = set(state.compounds_used_this_race) | {c for _, c in plan}
    return len(compounds_used) >= 2


def generate_candidates(state: RaceState, lap_step: int = 2, max_pit_laps_per_stop: int = 12) -> list[Strategy]:
    candidates: list[Strategy] = []
    remaining_laps = list(range(state.current_lap + 1, state.race_total_laps + 1, lap_step))
    compounds = state.available_compounds

    # 0-stop: finish on the current tyres.
    stint_length = state.race_total_laps - state.current_lap
    if _uses_two_compounds(state, ()) and _stint_feasible(state, state.compound, stint_length, True):
        candidates.append(Strategy(pit_plan=(), label="No further stops"))

    # 1-stop.
    for pit_lap in remaining_laps:
        first_stint = pit_lap - state.current_lap
        if not _stint_feasible(state, state.compound, first_stint, True):
            continue
        for compound in compounds:
            second_stint = state.race_total_laps - pit_lap
            plan = ((pit_lap, compound),)
            if not _uses_two_compounds(state, plan):
                continue
            if not _stint_feasible(state, compound, second_stint, False):
                continue
            candidates.append(Strategy(pit_plan=plan, label=f"Pit lap {pit_lap} -> {compound}"))

    # 2-stop: cap the pit-lap grid searched per stop to keep candidate count sane.
    sparse_laps = remaining_laps[::2][:max_pit_laps_per_stop]
    for pit_lap_1, pit_lap_2 in combinations(sparse_laps, 2):
        first_stint = pit_lap_1 - state.current_lap
        if not _stint_feasible(state, state.compound, first_stint, True):
            continue
        middle_stint = pit_lap_2 - pit_lap_1
        final_stint = state.race_total_laps - pit_lap_2
        for compound_1, compound_2 in product(compounds, compounds):
            plan = ((pit_lap_1, compound_1), (pit_lap_2, compound_2))
            if not _uses_two_compounds(state, plan):
                continue
            if not _stint_feasible(state, compound_1, middle_stint, False):
                continue
            if not _stint_feasible(state, compound_2, final_stint, False):
                continue
            candidates.append(
                Strategy(
                    pit_plan=plan,
                    label=f"Pit lap {pit_lap_1} -> {compound_1}, lap {pit_lap_2} -> {compound_2}",
                )
            )

    return candidates
