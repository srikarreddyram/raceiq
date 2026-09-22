"""Does the pre-race simulation value the grid the way real races do?

Every real starter of a sample of 2025 races (the test season) is
simulated from their real grid slot with their real season form, exactly
as race_plan builds a plan, and their simulated expected finish is
compared with where they actually finished.

This replaces an earlier, confounded check that moved ONE driver around
the grid and compared the resulting spread (P1 to P16: ~2.6 places)
with the real finish-by-grid spread (~10). Real finish-by-grid includes
the fact that fast cars qualify at the front; moving one car of fixed
pace doesn't, so the two numbers were never comparable. Simulating
everyone from where they really started is like for like.

Reported:
  movement       mean |finish - grid|, simulated vs real
  by grid band   mean finish for starters in P1-5, P6-10, P11-15, P16-20
  rank corr.     Spearman correlation of simulated expected finish with
                 actual finish, per race, averaged
  MAE            mean |simulated expected finish - actual finish|

`--zero-gap` reproduces the bug fixed alongside this file, where our
driver started level with the leader at every grid slot.

`--calibrate` sweeps monte_carlo.PASSING_DELTA_BASE_SECONDS and scores
each value on how well simulated finishes match real ones. It replaces
race_plan/calibrate_overtaking.py, which tuned the end-of-race passing
margin that the lap-by-lap race (monte_carlo round 7) retired.

Usage:
    uv run python -m race_plan.grid_sensitivity [--zero-gap | --calibrate]
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
N_SIMULATIONS = 300
BANDS = [(1, 5), (6, 10), (11, 15), (16, 20)]


def _starters() -> pd.DataFrame:
    con = get_connection()
    try:
        df = con.execute(
            f"""
            SELECT season || '_' || round AS race_id, driver_id, grid, position, status
            FROM bronze.ergast_results
            WHERE season = {SEASON} AND round IN ({', '.join(map(str, SAMPLE_ROUNDS))}) AND grid > 0
            """
        ).df()
    finally:
        con.close()
    df["classified"] = df["status"].map(is_classified)
    # Like for like with the simulation, which assumes our driver finishes.
    return df[df["classified"]].reset_index(drop=True)


def simulate(zero_gap: bool = False) -> pd.DataFrame:
    import strategy_engine.simulation.monte_carlo as mc
    from race_plan.field import build_pre_race_field, driver_season_form
    from race_plan.plan import _starting_state
    from strategy_engine.scoring.score import score_strategy
    from strategy_engine.search.candidates import generate_candidates
    from strategy_engine.tyre_baselines import rival_stop_laps

    rows = []
    for row in _starters().itertuples():
        try:
            state, _ = _starting_state(row.race_id, row.driver_id)
            if zero_gap:
                state = replace(state, gap_to_leader=0.0)
            rivals = build_pre_race_field(row.race_id, exclude_driver_id=row.driver_id)
            candidates = generate_candidates(state, feasibility="empirical", exclude_race_id=row.race_id)
            if not candidates:
                continue
            rng = np.random.default_rng(7)
            shared = mc.build_shared_context(state, rivals, N_SIMULATIONS, rng, pre_race=True)
            # The same number of stops every rival is charged
            # (tyre_baselines.rival_stop_laps from the same new tyres), mid
            # window — this checks the simulation, not strategy choice, so
            # our car mustn't carry one stop more or fewer than the field.
            n_laps = state.race_total_laps - state.current_lap
            target = len(rival_stop_laps(state.circuit_id, state.compound, state.tyre_age, n_laps))
            nearest = min(abs(len(c.pit_plan) - target) for c in candidates)
            pool = sorted((c for c in candidates if abs(len(c.pit_plan) - target) == nearest), key=lambda c: c.pit_plan)
            result = mc.simulate_strategy(
                state,
                pool[len(pool) // 2],
                rivals,
                shared,
                pace_deviation_override=driver_season_form(row.race_id, row.driver_id),
            )
            rows.append(
                {
                    "race_id": row.race_id,
                    "driver_id": row.driver_id,
                    "grid": int(row.grid),
                    "actual": int(row.position),
                    "simulated": float(score_strategy(result).expected_finish),
                }
            )
        except Exception as exc:  # one bad start shouldn't sink the whole check
            print(f"  skipped {row.race_id} {row.driver_id}: {exc}")
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> dict:
    corr = [spearmanr(g["simulated"], g["actual"]).statistic for _, g in df.groupby("race_id") if len(g) > 3]
    out = {
        "starts": len(df),
        "movement_sim": float((df["simulated"] - df["grid"]).abs().mean()),
        "movement_real": float((df["actual"] - df["grid"]).abs().mean()),
        "rank_corr": float(np.nanmean(corr)),
        "mae": float((df["simulated"] - df["actual"]).abs().mean()),
    }
    print(f"starts simulated: {out['starts']}")
    print(f"mean |finish - grid|   simulated {out['movement_sim']:.2f}   real {out['movement_real']:.2f}")
    print(f"rank correlation with actual finish (per race, mean): {out['rank_corr']:.3f}")
    print(f"MAE vs actual finish: {out['mae']:.2f} places")
    print("mean finish by grid band   simulated   real")
    for lo, hi in BANDS:
        band = df[df["grid"].between(lo, hi)]
        if len(band):
            print(f"  P{lo:>2}-{hi:<2}                    {band['simulated'].mean():6.2f}   {band['actual'].mean():6.2f}")
    return out


CALIBRATION_GRID = [0.2, 0.4, 0.6, 0.9, 1.3]


def calibrate() -> None:
    import strategy_engine.simulation.monte_carlo as mc

    original = mc.PASSING_DELTA_BASE_SECONDS
    results = []
    try:
        for value in CALIBRATION_GRID:
            mc.PASSING_DELTA_BASE_SECONDS = value
            print(f"\n=== PASSING_DELTA_BASE_SECONDS = {value} ===")
            out = report(simulate())
            results.append((value, out))
    finally:
        mc.PASSING_DELTA_BASE_SECONDS = original
    print("\nvalue   movement(sim/real)   rank corr   MAE")
    for value, out in results:
        print(f"{value:5.2f}   {out['movement_sim']:5.2f} / {out['movement_real']:4.2f}        {out['rank_corr']:.3f}     {out['mae']:.2f}")


def main() -> None:
    if "--calibrate" in sys.argv:
        calibrate()
        return
    zero_gap = "--zero-gap" in sys.argv
    print(f"{'BUGGY (our driver at 0.0s gap)' if zero_gap else 'FIXED (grid-slot gap)'} — {SEASON}, rounds {SAMPLE_ROUNDS}\n")
    report(simulate(zero_gap))


if __name__ == "__main__":
    main()
