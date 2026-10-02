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

`--pit-windows` checks the planner's main output — when to stop — against
the laps real cars stopped on, next to the field's usual lap. On these
ten rounds (139 stops where plan and driver made the same number): the
plan's target lap is off by 6.7 laps, the field's usual lap by 6.9, and
knowing the race's own median lap would still leave 4.3 — real cars in
one race stop far apart. The plan is better than habit in one-stop races
(7.8 against 9.5 laps) and worse in two-stop ones (5.0 against 3.0). Its
stop count is the race's most common one in every dry round but Monaco,
where 2025's two-stop rule isn't something it knows about.

`--calibrate` sweeps monte_carlo.PASSING_DELTA_BASE_SECONDS.

Usage:
    uv run python -m race_plan.grid_sensitivity [--planner | --pit-windows | --strategy-edge | --calibrate]
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
    from strategy_engine.tyre_pace import race_wear

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
    shared = mc.build_shared_context(
        state, cars[1:], n_simulations, rng, pre_race=True, stop_patterns=patterns, **mc.race_model(None, race_wear(race_id))
    )
    # Retirements for every car, the first car included — like for like with
    # a comparison against classified finishers only. (The rest draw theirs
    # in build_shared_context, at the same rate.)
    p_dnf = state.historical_dnf_rate if state.historical_dnf_rate is not None and np.isfinite(state.historical_dnf_rate) else mc.DEFAULT_DNF_RATE
    first_retires = rng.random(n_simulations) < p_dnf
    first_retire_lap = np.where(first_retires, rng.integers(0, shared.n_laps, size=n_simulations), shared.n_laps)
    # The first car runs as a typical car too, and every car — it included —
    # responds to nearby stops: there is no "our car" here.
    times = mc.race_typical_field(
        state, cars[1:], shared, cars[0].recent_pace_delta, patterns, rng,
        everyone_reacts=True, our_retire_lap=first_retire_lap,
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


def _real_stops() -> pd.DataFrame:
    """Every car's real in-laps and the compound fitted at each stop."""
    con = get_connection()
    try:
        laps = con.execute(
            f"""
            SELECT race_id, driver_id, lap_number, stint_number, compound
            FROM gold.lap_features WHERE race_id LIKE '{SEASON}_%' AND stint_number IS NOT NULL
            ORDER BY race_id, driver_id, lap_number
            """
        ).df()
    finally:
        con.close()
    laps["next_stint"] = laps.groupby(["race_id", "driver_id"]).stint_number.shift(-1)
    laps["next_compound"] = laps.groupby(["race_id", "driver_id"]).compound.shift(-1)
    stops = laps[laps.next_stint > laps.stint_number]
    return stops.groupby(["race_id", "driver_id"]).agg(laps=("lap_number", list), compounds=("next_compound", list)).reset_index()


def pit_windows() -> pd.DataFrame:
    """Does the plan's pit window hold the lap real cars stopped on?

    Every classified starter is planned from the compound they really
    started on, so the plan and the real race begin alike. Where the plan
    has as many stops as the driver really made, each planned stop is
    compared with the real one: the target lap's error, and whether the
    real lap fell inside the window. The reference is the field's own
    habit — the median lap real cars with that many stops pitted on at
    this circuit before (tyre_baselines.historical_stop_patterns). A plan
    that can't beat that is adding nothing to "pit when everyone usually
    does".
    """
    from race_plan.field import build_pre_race_field, driver_race_pace, stop_patterns_for_race
    from race_plan.plan import _plan_for_starting_compound, _starting_state, _windows_from_candidates

    results = _results()
    real = _real_stops().set_index(["race_id", "driver_id"])
    rows = []
    for row in results[results["classified"] & (results["grid"] > 0)].itertuples():
        key = (row.race_id, row.driver_id)
        if key not in real.index:
            continue
        patterns = stop_patterns_for_race(row.race_id)
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
            patterns,
        )
        if outcome is None:
            continue
        plan_stops = _windows_from_candidates(*outcome)
        total_laps = state.race_total_laps
        real_laps = real.loc[key, "laps"]
        base = None
        if patterns is not None:
            same = [f for f in patterns.fractions if len(f) == len(real_laps)]
            if same:
                base = [float(np.median([f[k] for f in same])) * total_laps for k in range(len(real_laps))]
        for k, real_lap in enumerate(real_laps):
            stop = plan_stops[k] if len(plan_stops) == len(real_laps) else None
            rows.append(
                {
                    "race_id": row.race_id,
                    "driver_id": row.driver_id,
                    "stop": k + 1,
                    "real_stops": len(real_laps),
                    "plan_stops": len(plan_stops),
                    "real_lap": real_lap,
                    "plan_lap": stop.nominal_lap if stop else None,
                    "window_open": stop.window_open if stop else None,
                    "window_close": stop.window_close if stop else None,
                    "field_lap": base[k] if base else None,
                }
            )
    df = pd.DataFrame(rows)
    matched = df[df["plan_stops"] == df["real_stops"]].copy()
    firsts = df[df["stop"] == 1]
    print(f"starters: {firsts.shape[0]}   plan's stop count matched the real one: {(firsts.plan_stops == firsts.real_stops).mean():.0%}")
    print(f"stop counts   plan {firsts.plan_stops.value_counts().sort_index().to_dict()}   real {firsts.real_stops.value_counts().sort_index().to_dict()}")
    if matched.empty:
        return df
    matched["plan_err"] = (matched.plan_lap - matched.real_lap).abs()
    matched["field_err"] = (matched.field_lap - matched.real_lap).abs()
    matched["inside"] = matched.real_lap.between(matched.window_open, matched.window_close)
    both = matched.dropna(subset=["field_err"])
    print(f"stops compared: {len(matched)}")
    print(f"target-lap error   plan {both.plan_err.mean():.2f} laps   field's usual lap {both.field_err.mean():.2f} laps   (same {len(both)} stops)")
    print(f"real stop inside the plan's window: {matched.inside.mean():.0%}   mean window {(matched.window_close - matched.window_open + 1).mean():.1f} laps")
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
    elif "--pit-windows" in sys.argv:
        print(f"Pit windows against real stops — {SEASON}, rounds {SAMPLE_ROUNDS}\n")
        pit_windows()
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
