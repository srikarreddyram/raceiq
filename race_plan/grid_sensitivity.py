"""Does the pre-race simulation value the grid the way real races do?

A sample of 2025 races (the test season) is each simulated as a whole
grid: every car from its real grid slot, on its season form before the
race, with its stops drawn from the real stop patterns at that circuit
(tyre_baselines.historical_stop_patterns) and its retirement drawn from
the circuit's DNF rate. Each car's expected finish, over the simulations
it finishes, is compared with where it really finished among the
classified finishers.

Every car is treated identically — there is no "our car" here — so this
checks the race the planner simulates, not the planner's choice of
strategy. `--planner` checks that too: for every classified starter it
runs the planner on the compound they really started on and scores the
recommended plan's expected finish, the number the Race Weekend page
shows.

History, because each version found something:

- The first check moved ONE driver around the grid and compared the
  resulting spread with the real finish-by-grid spread. Confounded: real
  finish-by-grid includes fast cars qualifying at the front.
- The second simulated each starter as "our car" on one hand-picked plan
  against a field whose stops came from tyre_baselines.rival_stop_laps.
  That rule gives every car on a compound the same stops, and before the
  race every rival is on the same compound, so the whole field pitted on
  the same laps (Hungary: 31 and 67 of 70). Our car jumped the field in
  one go and skipped a stop three laps from the end, and back-half
  starters came out 1.1-1.5 places better than they really finished. Its
  sweep of the passing threshold went the wrong way — harder passing made
  them look BETTER — which is what exposed it.

Reported:
  movement       mean |finish - grid|, simulated vs real
  by grid band   mean finish for starters in P1-5, P6-10, P11-15, P16-20
  rank corr.     Spearman correlation of simulated expected finish with
                 actual finish, per race, averaged
  MAE            mean |simulated expected finish - actual finish|
  grid slope     how the error (simulated - actual) changes per grid slot;
                 0 when the grid is weighted the way real races weight it

`--strategy-edge` asks whether the credit the simulation gives the
recommended plan over a typical strategy shows up in real results (every
2025 round; slow — it builds a full plan per starter).

`--calibrate` sweeps monte_carlo.PASSING_DELTA_BASE_SECONDS.

Usage:
    uv run python -m race_plan.grid_sensitivity [--planner | --strategy-edge | --calibrate]
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.common.data import is_classified
from models.common.db import get_connection

SAMPLE_ROUNDS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
SEASON = 2025
N_SIMULATIONS = 1000
PLANNER_SIMULATIONS = 300
BANDS = [(1, 5), (6, 10), (11, 15), (16, 20)]


def _results() -> pd.DataFrame:
    con = get_connection()
    try:
        df = con.execute(
            f"""
            SELECT season || '_' || round AS race_id, driver_id, grid, position, status
            FROM bronze.ergast_results
            WHERE season = {SEASON} AND round IN ({', '.join(map(str, SAMPLE_ROUNDS))})
            """
        ).df()
    finally:
        con.close()
    df["classified"] = df["status"].map(is_classified)
    return df


def simulate_race(race_id: str, n_simulations: int = N_SIMULATIONS, seed: int = 7) -> pd.DataFrame:
    """Every car's expected finish, from one simulation of the whole grid."""
    import strategy_engine.simulation.monte_carlo as mc
    from race_plan.field import build_pre_race_field, driver_race_pace, stop_patterns_for_race
    from race_plan.plan import _starting_state
    from strategy_engine.pit_loss import typical_pit_loss_seconds
    from strategy_engine.tyre_baselines import rival_stop_laps

    # Any car with laps in the race can seed the shared state (circuit,
    # distance, SC and DNF rates); it then races as one of the field.
    first = build_pre_race_field(race_id, exclude_driver_id="")
    for seed_car in first:
        try:
            state, _ = _starting_state(race_id, seed_car.driver_id)
            break
        except ValueError:
            continue
    else:
        return pd.DataFrame()
    state = replace(state, current_lap=0)

    cars = build_pre_race_field(race_id, exclude_driver_id=state.driver_id)
    cars = [
        replace(cars[0], driver_id=state.driver_id, gap_to_leader=state.gap_to_leader, recent_pace_delta=driver_race_pace(race_id, state.driver_id))
    ] + cars
    rng = np.random.default_rng(seed)
    patterns = stop_patterns_for_race(race_id)
    shared = mc.build_shared_context(state, cars[1:], n_simulations, rng, pre_race=True)
    n_laps, n_cars = shared.n_laps, len(cars)

    if patterns is not None:
        stops = mc.sample_rival_pit_laps(state, patterns, n_simulations, n_cars, rng)
    else:
        stops = np.zeros((n_simulations, n_laps, n_cars), dtype=bool)
        for j, car in enumerate(cars):
            stops[:, rival_stop_laps(state.circuit_id, car.compound, 0, n_laps), j] = True
    discount = np.where(shared.sc_occurs, mc.SC_PIT_DISCOUNT, 1.0)[:, :, None]
    pit = (stops * typical_pit_loss_seconds(state.circuit_id) * discount).astype(np.float32)

    pace = np.array([c.recent_pace_delta for c in cars])[None, :] + np.c_[shared.pace_error, shared.rival_pace_error]
    # Retirements for every car, the first car included — like for like with
    # a comparison against classified finishers only.
    p_dnf = state.historical_dnf_rate if state.historical_dnf_rate is not None and np.isfinite(state.historical_dnf_rate) else mc.DEFAULT_DNF_RATE
    retires = rng.random((n_simulations, n_cars)) < p_dnf
    retire_lap = np.where(retires, rng.integers(0, n_laps, size=(n_simulations, n_cars)), n_laps)

    times = mc.run_race(
        np.array([c.gap_to_leader for c in cars]),
        pace,
        shared.noise,
        pit,
        retire_lap,
        shared.sc_occurs,
        mc.passing_delta_seconds(state.historical_overtaking_rate),
    )
    finished = np.isfinite(times)
    # Position among the cars still running at the flag.
    positions = (times[:, None, :] < times[:, :, None]).sum(axis=2) + 1
    expected = np.where(finished, positions, np.nan)
    return pd.DataFrame(
        {
            "race_id": race_id,
            "driver_id": [c.driver_id for c in cars],
            "simulated": np.nanmean(expected, axis=0),
            "stops_source": patterns.source if patterns is not None else "rule",
        }
    )


def simulate() -> pd.DataFrame:
    results = _results()
    frames = [simulate_race(race_id) for race_id in results["race_id"].unique()]
    sims = pd.concat(frames, ignore_index=True)
    df = results[results["classified"] & (results["grid"] > 0)].merge(sims, on=["race_id", "driver_id"])
    return df.rename(columns={"position": "actual"})[["race_id", "driver_id", "grid", "actual", "simulated", "stops_source"]]


def simulate_planner() -> pd.DataFrame:
    """The recommended plan's expected finish, for every classified starter."""
    from race_plan.field import build_pre_race_field, driver_race_pace, stop_patterns_for_race
    from race_plan.plan import _plan_for_starting_compound, _starting_state

    results = _results()
    rows = []
    for row in results[results["classified"] & (results["grid"] > 0)].itertuples():
        try:
            state, _ = _starting_state(row.race_id, row.driver_id)
        except ValueError as exc:
            print(f"  skipped {row.race_id} {row.driver_id}: {exc}")
            continue
        outcome = _plan_for_starting_compound(
            state,
            state.compound,
            build_pre_race_field(row.race_id, exclude_driver_id=row.driver_id),
            PLANNER_SIMULATIONS,
            7,
            row.race_id,
            driver_race_pace(row.race_id, row.driver_id),
            stop_patterns_for_race(row.race_id),
        )
        if outcome is None:
            continue
        rows.append(
            {
                "race_id": row.race_id,
                "driver_id": row.driver_id,
                "grid": int(row.grid),
                "actual": int(row.position),
                "simulated": float(outcome[0].expected_finish),
            }
        )
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict:
    corr = [spearmanr(g["simulated"], g["actual"]).statistic for _, g in df.groupby("race_id") if len(g) > 3]
    err = df["simulated"] - df["actual"]
    slope = float(np.polyfit(df["grid"], err, 1)[0])
    out = {
        "starts": len(df),
        "races": df["race_id"].nunique(),
        "movement_sim": float((df["simulated"] - df["grid"]).abs().mean()),
        "movement_real": float((df["actual"] - df["grid"]).abs().mean()),
        "rank_corr": float(np.nanmean(corr)),
        "mae": float(err.abs().mean()),
        "grid_slope": slope,
    }
    print(f"starts simulated: {out['starts']} over {out['races']} races")
    print(f"mean |finish - grid|   simulated {out['movement_sim']:.2f}   real {out['movement_real']:.2f}")
    print(f"rank correlation with actual finish (per race, mean): {out['rank_corr']:.3f}")
    print(f"MAE vs actual finish: {out['mae']:.2f} places")
    print(f"error per grid slot: {slope:+.3f} places (0 = grid weighted like real races)")
    print("mean finish by grid band   simulated   real")
    for lo, hi in BANDS:
        band = df[df["grid"].between(lo, hi)]
        if len(band):
            print(f"  P{lo:>2}-{hi:<2}                    {band['simulated'].mean():6.2f}   {band['actual'].mean():6.2f}")
    return out


def strategy_edge(rounds: list[int] | None = None) -> pd.DataFrame:
    """Is the strategy credit the planner gives its plan real?

    For every classified starter: the planner's recommended plan and the
    same car on a typical strategy (monte_carlo.simulate_typical_strategy),
    against what they really did. If the plan's stop count is worth what
    the simulation says, drivers who really ran that many stops should
    have beaten the typical-strategy expectation by about that much more
    than drivers who didn't.
    """
    import race_plan.plan as planner

    global SAMPLE_ROUNDS
    saved = SAMPLE_ROUNDS
    SAMPLE_ROUNDS = rounds or list(range(1, 25))
    try:
        results = _results()
    finally:
        SAMPLE_ROUNDS = saved
    con = get_connection()
    try:
        real_stops = con.execute(
            f"""
            SELECT race_id, driver_id, COUNT(DISTINCT stint_number) - 1 AS real_stops
            FROM gold.lap_features WHERE race_id LIKE '{SEASON}_%' GROUP BY 1, 2
            """
        ).df()
    finally:
        con.close()

    rows = []
    for row in results[results["classified"] & (results["grid"] > 0)].itertuples():
        try:
            plan = planner.build_race_plan(row.race_id, row.driver_id, n_simulations=PLANNER_SIMULATIONS, seed=7)
        except (ValueError, RuntimeError) as exc:
            print(f"  skipped {row.race_id} {row.driver_id}: {exc}")
            continue
        rows.append(
            {
                "race_id": row.race_id,
                "driver_id": row.driver_id,
                "grid": int(row.grid),
                "actual": int(row.position),
                "plan_finish": plan.expected_finish,
                "typical_finish": plan.typical_expected_finish,
                "plan_stops": len(plan.stops),
            }
        )
    df = pd.DataFrame(rows).merge(real_stops, on=["race_id", "driver_id"])
    df["claimed_gain"] = df["typical_finish"] - df["plan_finish"]
    df["beat_typical"] = df["typical_finish"] - df["actual"]
    df["ran_plan_stops"] = df["real_stops"] == df["plan_stops"]

    ran, other = df[df["ran_plan_stops"]], df[~df["ran_plan_stops"]]
    real_gain = ran["beat_typical"].mean() - other["beat_typical"].mean()
    se = np.sqrt(ran["beat_typical"].var() / len(ran) + other["beat_typical"].var() / len(other))
    print(f"starts: {len(df)}  ran the plan's stop count: {len(ran)}  didn't: {len(other)}")
    print(f"typical-strategy expected finish   MAE {(df['typical_finish'] - df['actual']).abs().mean():.2f}   mean error {(df['typical_finish'] - df['actual']).mean():+.2f}")
    print(f"recommended plan expected finish   MAE {(df['plan_finish'] - df['actual']).abs().mean():.2f}   mean error {(df['plan_finish'] - df['actual']).mean():+.2f}")
    print(f"simulation's credit for the plan:  {df['claimed_gain'].mean():.2f} places")
    print(f"real edge of running its stop count: {real_gain:+.2f} +/- {se:.2f} places")
    return df


CALIBRATION_GRID = [0.6, 0.9, 1.3, 1.8]


def calibrate() -> None:
    import strategy_engine.simulation.monte_carlo as mc

    original = mc.PASSING_DELTA_BASE_SECONDS
    results = []
    try:
        for value in CALIBRATION_GRID:
            mc.PASSING_DELTA_BASE_SECONDS = value
            print(f"\n=== PASSING_DELTA_BASE_SECONDS = {value} ===")
            results.append((value, report(simulate())))
    finally:
        mc.PASSING_DELTA_BASE_SECONDS = original
    print("\nvalue   movement(sim/real)   rank corr   MAE    grid slope")
    for value, out in results:
        print(
            f"{value:5.2f}   {out['movement_sim']:5.2f} / {out['movement_real']:4.2f}        "
            f"{out['rank_corr']:.3f}     {out['mae']:.2f}   {out['grid_slope']:+.3f}"
        )


def main() -> None:
    if "--calibrate" in sys.argv:
        calibrate()
    elif "--strategy-edge" in sys.argv:
        print(f"Strategy edge — every classified {SEASON} starter\n")
        strategy_edge()
    elif "--planner" in sys.argv:
        print(f"Planner's recommended plan — {SEASON}, rounds {SAMPLE_ROUNDS}\n")
        report(simulate_planner())
    else:
        print(f"Whole-grid simulation — {SEASON}, rounds {SAMPLE_ROUNDS}\n")
        report(simulate())


if __name__ == "__main__":
    main()
