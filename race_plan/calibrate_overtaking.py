"""Calibrate OVERTAKE_MARGIN_BASE_SECONDS against real grid-to-flag movement.

The simulation now requires a time advantage before one car is allowed
past another, scaled by how hard the circuit is to overtake at. The scale
of that margin is not something to guess: too small and the grid means
nothing (the problem it was added to fix), too large and nobody ever
changes position and every plan just returns the grid order back.

There is a directly observable target. Across this project's data,
classified finishers move a mean of 2.85 positions between their grid slot
and the flag. Sweeping the base margin and simulating real race starts
gives the value that reproduces that.

Two honest caveats about the target:
- Observed movement includes places gained from retirements ahead and from
  pit-cycle shuffles, not only on-track passes, so matching it exactly
  would slightly over-fit the margin. It's the right order of magnitude,
  not a precise bar.
- Only finishers are counted, on both sides, so the comparison is at least
  like-for-like.

Usage:
    uv run python -m race_plan.calibrate_overtaking
"""

from __future__ import annotations

import numpy as np

from models.common.db import get_connection

# Mean |finish - grid| across classified finishers in this project's data.
OBSERVED_MEAN_POSITION_CHANGE = 2.85

CANDIDATE_MARGINS = [2.0, 6.0, 12.0, 20.0, 30.0]


def _sample_starts(limit: int = 8) -> list[tuple[str, str]]:
    """A spread of real (race, driver) starts to calibrate against —
    deliberately across different circuits and grid slots rather than one
    race, since the margin scales per circuit.
    """
    con = get_connection()
    try:
        rows = con.execute(
            """
            SELECT res.driver_id, r.race_id
            FROM bronze.ergast_results res
            JOIN silver.races r ON r.season = res.season AND r.round = res.round
            WHERE r.season = 2025 AND res.grid > 0 AND res.position IS NOT NULL
                AND r.round IN (4, 6, 10, 14)
            ORDER BY r.round, res.grid
            """
        ).df()
    finally:
        con.close()
    # Spread across the grid rather than taking the first N (all front-runners).
    picks = rows.iloc[:: max(1, len(rows) // limit)].head(limit)
    return [(row.race_id, row.driver_id) for row in picks.itertuples()]


def evaluate_margin(margin: float, starts: list[tuple[str, str]], n_simulations: int = 400) -> float:
    """Mean |simulated finish - grid| at a given base margin."""
    import strategy_engine.simulation.monte_carlo as mc
    from race_plan.field import build_pre_race_field
    from race_plan.plan import _starting_state
    from strategy_engine.scoring.score import score_strategy
    from strategy_engine.search.candidates import generate_candidates

    original = mc.OVERTAKE_MARGIN_BASE_SECONDS
    mc.OVERTAKE_MARGIN_BASE_SECONDS = margin
    try:
        movements = []
        for race_id, driver_id in starts:
            try:
                state, _ = _starting_state(race_id, driver_id)
                rivals = build_pre_race_field(race_id, exclude_driver_id=driver_id)
                candidates = generate_candidates(state, feasibility="empirical", exclude_race_id=race_id)
                if not candidates:
                    continue
                rng = np.random.default_rng(7)
                shared = mc.build_shared_context(
                    state, rivals, n_simulations, rng, pace_uncertainty=mc.PACE_ESTIMATE_UNCERTAINTY_PRE_RACE
                )
                result = mc.simulate_strategy(state, candidates[len(candidates) // 2], rivals, shared)
                scored = score_strategy(result)
                movements.append(abs(scored.expected_finish - state.current_position))
            except Exception:
                continue
        return float(np.mean(movements)) if movements else float("nan")
    finally:
        mc.OVERTAKE_MARGIN_BASE_SECONDS = original


def main() -> None:
    starts = _sample_starts()
    print(f"calibrating against {len(starts)} real starts, target {OBSERVED_MEAN_POSITION_CHANGE:.2f} positions\n")
    print(f"{'margin (s)':>11}  {'mean |finish-grid|':>19}  {'error':>7}")
    best, best_error = None, float("inf")
    for margin in CANDIDATE_MARGINS:
        movement = evaluate_margin(margin, starts)
        error = abs(movement - OBSERVED_MEAN_POSITION_CHANGE)
        print(f"{margin:>11.1f}  {movement:>19.2f}  {error:>7.2f}")
        if error < best_error:
            best, best_error = margin, error
    print(f"\nclosest: OVERTAKE_MARGIN_BASE_SECONDS = {best}")


if __name__ == "__main__":
    main()
