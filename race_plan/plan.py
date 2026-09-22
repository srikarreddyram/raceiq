"""A pre-race weekend plan, rather than a mid-race pit call.

strategy_engine/ answers "it's lap 20, what now?". This answers the
question a strategist actually has on Saturday night: which compounds do
we start on and run, roughly when do we stop, and how long do we hold out
for a safety car before giving up and pitting anyway.

Three things that differ from the in-race engine:

1. **The starting compound is a decision, not a given.** The in-race
   engine takes the tyre you're already on as fixed. Here every legal
   starting compound is planned out separately and the plans are compared
   against each other, which is what makes this a compound
   *recommendation* rather than a pit-timing one.

2. **Grid position is the starting track position.** Taken from Ergast's
   `grid` rather than lap 1's classified position, because a plan made
   before the race can't know what happened at turn 1 — and because
   strategy from P10 genuinely differs from strategy from P1, which is
   the point of feeding it in.

3. **Stops are windows, not laps.** A single "pit on lap 27" is false
   precision: the simulation usually can't separate lap 26 from lap 28 in
   any meaningful way. The window here is the span of pit laps across
   every candidate whose score lands within a tolerance of the best one —
   so its width is an honest read of how sharp the call actually is. A
   one-lap window means the timing really matters; an eight-lap window
   means it doesn't, and the strategist should spend their attention
   elsewhere.

Tyre-set allocation across the weekend lives in
race_plan/tyre_allocation.py and is printed alongside this plan: what the
race needs is reserved first, then qualifying, and the remainder is the
practice budget.

Usage:
    uv run python -m race_plan.plan --race-id 2026_7 --driver leclerc
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import pandas as pd

from models.common.data import load_race_features
from models.common.db import get_connection
from race_plan.field import starting_gap_for_position
from race_plan.vsc_threshold import WaitWindow, compute_wait_window
from strategy_engine.scoring.score import StrategyScore, score_strategy
from strategy_engine.search.candidates import generate_candidates
from strategy_engine.tyre_pace import plan_cost, wear_rates
from strategy_engine.simulation.monte_carlo import (
    _deterministic_tyre_plan,
    build_shared_context,
    simulate_strategies,
)
from strategy_engine.state import DRY_COMPOUNDS, RaceState
from strategy_engine.tyre_baselines import typical_max_stint_length

# How far below the best strategy's score a candidate can land and still
# count as "the same call". Sets the width of every reported pit window,
# so it's the one number that decides whether this tool sounds more
# certain than it is.
WINDOW_SCORE_TOLERANCE = 0.05

# Wet compounds are a response to conditions on the day, not something a
# dry-weather plan should pre-commit to.
PLANNABLE_STARTING_COMPOUNDS = DRY_COMPOUNDS

SCREEN_SIMULATIONS = 150
PLAN_FINALISTS = 24


@dataclass
class PlannedStop:
    stop_number: int
    compound: str
    window_open: int
    window_close: int
    nominal_lap: int
    wait: WaitWindow | None = None

    @property
    def window_width(self) -> int:
        return self.window_close - self.window_open


@dataclass
class RacePlan:
    race_id: str
    driver_id: str
    team_id: str
    circuit_id: str
    grid_position: int
    total_laps: int
    track_temp: float
    air_temp: float
    rain_expected: bool
    historical_sc_rate: float | None
    starting_compound: str
    stops: list[PlannedStop]
    expected_finish: float
    win_probability: float
    points_probability: float
    expected_points: float
    considered: list[dict] = field(default_factory=list)
    cold_start_circuit: bool = False
    # Planning an unrun race (race_plan/future.py): where the conditions
    # came from, and whether the grid slot was chosen or expected.
    is_future: bool = False
    conditions_source: str = "measured"
    conditions_note: str = "Measured during the race."
    rain_probability: float | None = None
    grid_is_expected: bool = False
    laps_known: bool = True
    prior_races_at_circuit: int | None = None
    historical_overtaking_rate: float | None = None
    circuit_baseline_track_temp: float | None = None

    @property
    def compound_choice_is_decisive(self) -> bool:
        """True only if the best starting compound actually beats the next
        one by more than simulation noise. Three compounds separated by
        0.002 of score is a tie, and printing the top one as a
        recommendation would be inventing a decision the model didn't make.
        """
        if len(self.considered) < 2:
            return True
        best, second = self.considered[0]["strategy_score"], self.considered[1]["strategy_score"]
        return (best - second) > 0.02 * max(best, 1e-9)

    @property
    def compound_sequence(self) -> list[str]:
        return [self.starting_compound] + [stop.compound for stop in self.stops]


def _grid_position(race_id: str, driver_id: str) -> int | None:
    season, rnd = race_id.split("_")
    con = get_connection()
    try:
        row = con.execute(
            "SELECT grid FROM bronze.ergast_results WHERE season = ? AND round = ? AND driver_id = ?",
            [int(season), int(rnd), driver_id],
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        return None
    grid = int(row[0])
    # Ergast records a pit-lane start as grid 0. That's a real starting
    # position, just not one that sorts with the others — treated as last.
    return grid if grid > 0 else 20


@lru_cache(maxsize=1)
def _cached_race_features() -> pd.DataFrame:
    """One Gold load per process.

    `load_race_features()` pulls ~205k rows, and anything that builds many
    plans in a loop — the overtaking calibration sweep, a batch of drivers —
    was paying that cost per call. Cached here rather than inside
    models/common/data.py, which other callers expect to return fresh data.
    """
    return load_race_features()


def _starting_state(race_id: str, driver_id: str, grid_override: int | None = None) -> tuple[RaceState, pd.DataFrame]:
    df = _cached_race_features()
    race_rows = df[df.race_id == race_id]
    if race_rows.empty:
        raise ValueError(f"No Gold rows for race_id={race_id!r}")

    driver_rows = race_rows[race_rows.driver_id == driver_id]
    if driver_rows.empty:
        raise ValueError(f"{driver_id!r} has no laps in {race_id!r}")

    total_laps = int(race_rows.lap_number.max())
    first_lap = driver_rows.sort_values("lap_number").iloc[0]
    state = RaceState.from_gold_row(first_lap, race_total_laps=total_laps)

    # Lap 1 has no field average: FastF1 records no lap time for the
    # standing-start lap, so AVG() over that lap is undefined. Left as NaN
    # it poisons the pace calculation, and a NaN gap silently scores this
    # driver P1 in every simulation (see monte_carlo's guard). The race's
    # own median green-flag field average is the pace *scale* of this
    # circuit — how long a lap takes here — which a plan needs and which
    # no amount of pre-race care can invent for a first-time venue.
    if state.field_avg_lap_time_seconds is None or pd.isna(state.field_avg_lap_time_seconds):
        fallback = race_rows.loc[~race_rows.is_pit_lap, "field_avg_lap_time_seconds"].median()
        state = replace(state, field_avg_lap_time_seconds=float(fallback))

    # `grid_override` is the "what if we start P8?" question a strategist asks
    # before qualifying is known.
    grid = grid_override if grid_override is not None else _grid_position(race_id, driver_id)
    if grid is not None:
        # The plan is made before the start, so track position is the grid
        # slot — a pre-race plan that inherited lap 1's measured gaps would
        # be quietly using the result of a start it's supposed to be
        # planning for.
        #
        # The gap comes from the same grid-slot table the rivals' does
        # (race_plan/field.py). It used to be 0.0 for our driver at every
        # grid slot while rivals got their slot's measured gap — so a P16
        # starter was placed level with the leader, the sticky-track-position
        # rule counted no rival as having started ahead, and the grid was
        # worth almost nothing in every plan.
        state = replace(
            state,
            current_position=float(grid),
            gap_to_leader=starting_gap_for_position(grid),
            tyre_age=0.0,
            stint_number=1,
        )
    return state, race_rows


def _plan_for_starting_compound(
    state: RaceState,
    compound: str,
    rivals: list,
    n_simulations: int,
    seed: int,
    race_id: str,
    pace_anchor: float,
) -> tuple[StrategyScore, list[StrategyScore]] | None:
    """Score every candidate strategy that starts on `compound`.

    `pace_anchor` is this driver's season-to-date pace delta — the same
    quantity the rivals are projected from. Measured tyre wear decides how
    the candidates differ from each other (strategy_engine/tyre_pace.py),
    and the anchor sets the level, so our driver and the field are
    measured with one estimator rather than two.

    Two-stage, like the in-race engine: every candidate is screened at
    SCREEN_SIMULATIONS, and the best PLAN_FINALISTS are re-scored at the
    full count. The pit windows are read off those finalists, which is
    where every near-optimal candidate lives. One plan took 22 s before.
    """
    start_state = replace(
        state,
        compound=compound,
        compounds_used_this_race={compound},
        # A plan is made from before lap 1, so the whole race is ahead.
        current_lap=0,
    )
    # Empirical feasibility, not the Tyre Degradation model's — planning a
    # whole race from the grid needs stint lengths teams actually run, and
    # the model's are ~40% short (see _stint_feasible_empirical).
    candidates = generate_candidates(start_state, feasibility="empirical", exclude_race_id=race_id)
    if not candidates:
        return None

    rng = np.random.default_rng(seed)

    # The spread between candidates comes from measured tyre wear (see
    # strategy_engine/tyre_pace.py for why not the LSTM); the level comes
    # from the anchor, the same season form every rival is projected from.
    wear = wear_rates(int(race_id.split("_")[0]))
    costs = np.array([plan_cost(_deterministic_tyre_plan(start_state, c), wear) for c in candidates])
    pace = dict(zip(candidates, pace_anchor + (costs - costs.mean())))

    def score_all(strategies: list, n: int) -> list[StrategyScore]:
        # Pre-race uncertainty, not the in-race figure: nothing about any
        # car's pace today has been observed yet.
        shared = build_shared_context(start_state, rivals, n, rng, pre_race=True)
        results = simulate_strategies(start_state, strategies, rivals, shared, [pace[c] for c in strategies])
        return [score_strategy(r) for r in results]

    finalists = candidates
    if len(candidates) > PLAN_FINALISTS and n_simulations > SCREEN_SIMULATIONS:
        screened = sorted(zip(score_all(candidates, SCREEN_SIMULATIONS), candidates), key=lambda x: -x[0].strategy_score)
        finalists = [c for _, c in screened[:PLAN_FINALISTS]]
    scored = score_all(finalists, n_simulations)
    scored.sort(key=lambda s: s.strategy_score, reverse=True)
    return scored[0], scored


def _windows_from_candidates(best: StrategyScore, scored: list[StrategyScore]) -> list[PlannedStop]:
    """Turn the near-optimal candidates into one window per stop.

    Only candidates with the same compound sequence as the best plan count
    — a two-stop on hards and a two-stop on softs are different calls, and
    merging their pit laps into one span would describe a plan nobody
    would actually run.
    """
    best_compounds = [compound for _, compound in best.pit_plan]
    cutoff = best.strategy_score * (1 - WINDOW_SCORE_TOLERANCE)

    comparable = [
        s
        for s in scored
        if s.strategy_score >= cutoff and [c for _, c in s.pit_plan] == best_compounds
    ]

    stops = []
    for index, (nominal_lap, compound) in enumerate(best.pit_plan):
        laps_at_this_stop = [s.pit_plan[index][0] for s in comparable if len(s.pit_plan) > index]
        stops.append(
            PlannedStop(
                stop_number=index + 1,
                compound=compound,
                window_open=min(laps_at_this_stop) if laps_at_this_stop else nominal_lap,
                window_close=max(laps_at_this_stop) if laps_at_this_stop else nominal_lap,
                nominal_lap=nominal_lap,
            )
        )
    return stops


def build_race_plan(
    race_id: str,
    driver_id: str,
    n_simulations: int = 2000,
    seed: int = 42,
    grid_override: int | None = None,
    track_temp: float | None = None,
    rain: bool | None = None,
) -> RacePlan:
    """Plan one driver's race — run or unrun.

    A completed race starts from its own first lap and measured weather
    (with optional what-if overrides). An upcoming one starts from the
    circuit's history, a forecast or typical conditions, the expected grid
    and the current entry list (race_plan/future.py).
    """
    from race_plan.field import build_pre_race_field, driver_season_form
    from race_plan.future import future_setup, has_race_data

    con = get_connection()
    try:
        run = has_race_data(con, race_id)
    finally:
        con.close()

    future_meta: dict = {}
    if run:
        state, race_rows = _starting_state(race_id, driver_id, grid_override)
        # Pre-race field, not the in-race snapshot: see race_plan/field.py for
        # what reusing the lap-1 snapshot does to this plan.
        rivals = build_pre_race_field(race_id, exclude_driver_id=driver_id)
        # Our own season form, from the identical source the rivals use.
        pace_anchor = driver_season_form(race_id, driver_id)
        measured_track = float(race_rows.track_temp.mean())
        conditions = {
            "track_temp": track_temp if track_temp is not None else measured_track,
            "air_temp": float(race_rows.air_temp.mean()),
            "rain_expected": rain if rain is not None else bool(race_rows.rainfall_flag.max()),
            "source": "override" if (track_temp is not None or rain is not None) else "measured",
            "note": "Set by hand, over the conditions measured on the day."
            if (track_temp is not None or rain is not None)
            else "Measured during the race.",
            "rain_probability": None,
        }
        if conditions["source"] == "override":
            baseline = state.circuit_baseline_track_temp
            state = replace(
                state,
                track_temp=conditions["track_temp"],
                rainfall_flag=conditions["rain_expected"],
                condition_delta=(conditions["track_temp"] - baseline) if baseline is not None else state.condition_delta,
            )
    else:
        state, rivals, cond, priors, pace_anchor, _entry = future_setup(race_id, driver_id, grid_override, track_temp, rain)
        conditions = {
            "track_temp": cond.track_temp,
            "air_temp": cond.air_temp,
            "rain_expected": cond.rain_expected,
            "source": cond.source,
            "note": cond.note,
            "rain_probability": cond.rain_probability,
        }
        future_meta = {
            "is_future": True,
            "grid_is_expected": grid_override is None,
            "laps_known": priors["laps_known"],
            "prior_races_at_circuit": priors["prior_races"],
        }

    results = {}
    for compound in PLANNABLE_STARTING_COMPOUNDS:
        outcome = _plan_for_starting_compound(
            state, compound, rivals, n_simulations, seed, race_id, pace_anchor
        )
        if outcome is not None:
            results[compound] = outcome

    if not results:
        raise RuntimeError(f"No feasible race plan found for {driver_id} at {race_id}")

    starting_compound, (best, scored) = max(results.items(), key=lambda kv: kv[1][0].strategy_score)
    stops = _windows_from_candidates(best, scored)

    # Attach the "how long do we hold out for a caution" call to each stop.
    # The hard bound is tyre life: a caution being worth waiting for never
    # justifies running a stint past what the compound can actually do here.
    stint_start = 0
    stint_compound = starting_compound
    for stop in stops:
        latest_safe = int(
            min(
                state.race_total_laps,
                stint_start + typical_max_stint_length(state.circuit_id, stint_compound, race_id),
            )
        )
        # Bounded by the pit window as well as by tyre life: the gamble on
        # offer is "pit at the window's open, or hold to its close hoping
        # for a caution". Telling a strategist to hold past the window this
        # same plan just recommended would be incoherent.
        stop.wait = compute_wait_window(
            state,
            compound=stint_compound,
            stint_start_lap=stint_start,
            earliest_lap=max(stop.window_open, stint_start + 1),
            latest_safe_lap=max(min(latest_safe, stop.window_close), stop.window_open),
            exclude_race_id=race_id,
        )
        stint_start = stop.nominal_lap
        stint_compound = stop.compound

    considered = [
        {
            "starting_compound": compound,
            "plan": outcome[0].label,
            "strategy_score": outcome[0].strategy_score,
            "expected_finish": outcome[0].expected_finish,
            "expected_points": outcome[0].expected_points,
        }
        for compound, outcome in sorted(results.items(), key=lambda kv: -kv[1][0].strategy_score)
    ]

    return RacePlan(
        cold_start_circuit=bool(
            state.historical_sc_rate is None or pd.isna(state.historical_sc_rate)
        ),
        race_id=race_id,
        driver_id=driver_id,
        team_id=state.team_id,
        circuit_id=state.circuit_id,
        grid_position=int(state.current_position),
        total_laps=state.race_total_laps,
        track_temp=float(conditions["track_temp"]),
        air_temp=float(conditions["air_temp"]),
        rain_expected=bool(conditions["rain_expected"]),
        historical_overtaking_rate=state.historical_overtaking_rate,
        circuit_baseline_track_temp=state.circuit_baseline_track_temp,
        conditions_source=conditions["source"],
        conditions_note=conditions["note"],
        rain_probability=conditions["rain_probability"],
        **future_meta,
        historical_sc_rate=state.historical_sc_rate,
        starting_compound=starting_compound,
        stops=stops,
        expected_finish=best.expected_finish,
        win_probability=best.win_probability,
        points_probability=best.points_probability,
        expected_points=best.expected_points,
        considered=considered,
    )


def format_plan(plan: RacePlan) -> str:
    lines = []
    lines.append(f"RACE PLAN — {plan.driver_id.upper()} ({plan.team_id}) — {plan.race_id} @ {plan.circuit_id}")
    lines.append("=" * 78)
    rain = "rain expected" if plan.rain_expected else "dry"
    sc = (
        "no prior races here"
        if plan.historical_sc_rate is None or pd.isna(plan.historical_sc_rate)
        else f"{plan.historical_sc_rate * 100:.0f}% historically"
    )
    lines.append(
        f"Grid P{plan.grid_position} · {plan.total_laps} laps · track {plan.track_temp:.0f}C / "
        f"air {plan.air_temp:.0f}C · {rain} · safety car {sc}"
    )
    if plan.cold_start_circuit:
        lines.append(
            "COLD START  first race at this circuit — no safety-car rate, pit loss or stint history. "
            "Circuit inputs fall back to cross-circuit averages."
        )
    lines.append("")
    if plan.compound_choice_is_decisive:
        lines.append(f"START ON   {plan.starting_compound}")
    else:
        lines.append(
            f"START ON   {plan.starting_compound}  (NOT DECISIVE — all compounds within noise, see below)"
        )
    lines.append(f"SEQUENCE   {' -> '.join(plan.compound_sequence)}")
    lines.append("")

    for stop in plan.stops:
        lines.append(f"STOP {stop.stop_number} — fit {stop.compound}")
        width = "single lap" if stop.window_width == 0 else f"{stop.window_width + 1} laps wide"
        lines.append(f"  pit window      laps {stop.window_open}-{stop.window_close}  (target {stop.nominal_lap}, {width})")
        if stop.wait is not None:
            w = stop.wait
            if w.has_window:
                lines.append(
                    f"  caution gamble  hold to lap {w.pull_the_plug_lap}, then pit regardless "
                    f"(caution saves {w.caution_saving_seconds:.0f}s, {w.per_lap_caution_probability * 100:.1f}%/lap)"
                )
            else:
                lines.append(f"  caution gamble  none — pit on window, don't wait")
            lines.append(f"                  {w.reason}")
        lines.append("")

    lines.append(
        f"OUTCOME    P{plan.expected_finish:.1f} expected · {plan.win_probability * 100:.1f}% win · "
        f"{plan.points_probability * 100:.0f}% points · {plan.expected_points:.1f} pts"
    )
    lines.append("")
    lines.append("STARTING COMPOUNDS CONSIDERED")
    for row in plan.considered:
        marker = ">" if row["starting_compound"] == plan.starting_compound else " "
        lines.append(
            f" {marker} {row['starting_compound']:<7} score {row['strategy_score']:.3f} · "
            f"P{row['expected_finish']:.1f} · {row['expected_points']:.1f} pts · {row['plan']}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a pre-race weekend plan")
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--n-simulations", type=int, default=2000)
    args = parser.parse_args()

    plan = build_race_plan(args.race_id, args.driver, n_simulations=args.n_simulations)
    print(format_plan(plan))

    # The allocation follows the race plan rather than being decided
    # separately: what the race needs is what gets reserved first.
    from race_plan.tyre_allocation import format_allocation, recommend_tyre_allocation

    print()
    print(
        format_allocation(
            recommend_tyre_allocation(args.race_id, args.driver, plan.compound_sequence)
        )
    )


if __name__ == "__main__":
    main()
