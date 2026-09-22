"""Top-level entry point — ties search, simulation, scoring, and
recommendation together into the single call PRD Section 12's
`/predict/strategy/optimal` endpoint (Section 14) will eventually wrap.

Usage:
    uv run python -m strategy_engine.engine --race-id 2025_4 --lap 20 --driver piastri
"""

from __future__ import annotations

import time

import numpy as np

from strategy_engine.field import build_field_snapshot_from_gold, driver_recent_pace
from strategy_engine.scoring.score import score_strategy
from strategy_engine.search.candidates import generate_candidates
from strategy_engine.simulation.monte_carlo import _deterministic_tyre_plan, build_shared_context, simulate_strategies
from strategy_engine.state import RaceState
from strategy_engine.tyre_pace import age_neutral, plan_cost, tyre_adjusted_rivals, wear_rates


def candidate_pace_overrides(
    state: RaceState, strategies: list, pace_anchor: float | None, season: int | None = None
) -> list[float]:
    """Our car's per-lap pace for each strategy: our recent pace — measured
    exactly as every rival's is (field.driver_recent_pace) — made
    tyre-age-neutral, plus that strategy's measured tyre cost over the
    remaining laps. Rivals get the same treatment in
    tyre_pace.tyre_adjusted_rivals, so both sides come from one estimator.
    See strategy_engine/tyre_pace.py for why this replaced the LSTM's
    per-strategy pace.
    """
    if pace_anchor is None or not np.isfinite(pace_anchor):
        # No measured recent pace (the caller had none) — the snapshot lap's
        # own delta, noisy but from the same session and the same scale.
        pace_anchor = float(state.lap_time_seconds - state.field_avg_lap_time_seconds)
        if not np.isfinite(pace_anchor):
            pace_anchor = 0.0
    wear = wear_rates(season)
    level = age_neutral(pace_anchor, state.compound, state.tyre_age, wear)
    return [level + plan_cost(_deterministic_tyre_plan(state, s), wear) for s in strategies]


SCREEN_SIMULATIONS = 500
FINALISTS = 20


def recommend_strategy(
    state: RaceState,
    rivals: list,
    n_simulations: int = 5000,
    seed: int = 42,
    pace_anchor: float | None = None,
    season: int | None = None,
    exclude_race_id: str | None = None,
) -> dict:
    """Search, simulate, score, explain.

    Candidates use EMPIRICAL stint feasibility — stint lengths teams have
    actually run at this circuit — not the Tyre Degradation model's, whose
    stints run ~40% short (race_plan/ switched for the same reason). On the
    model basis Piastri at Bahrain 2025 lap 20 had no one-stop candidate at
    all, and the engine recommended a two-stop that simulates at 18% to win
    over a one-stop at 54%; the LSTM's inflated two-stop pace had been
    hiding that. `exclude_race_id` keeps a historical snapshot from reading
    the stint lengths of the race it's in.

    The empirical basis yields ~270 candidates, so the search is two-stage:
    every candidate is screened at SCREEN_SIMULATIONS, the best FINALISTS
    are re-scored at the full `n_simulations`. Both stages share their own
    random draws across candidates (common random numbers).
    """
    from strategy_engine.recommendation.reasoning import build_recommendation

    rng = np.random.default_rng(seed)
    candidates = generate_candidates(state, feasibility="empirical", exclude_race_id=exclude_race_id)
    if not candidates:
        raise RuntimeError("No feasible strategy candidates found for this race state")

    rivals = tyre_adjusted_rivals(rivals, state, season)
    overrides = dict(zip(candidates, candidate_pace_overrides(state, candidates, pace_anchor, season)))

    def score_all(strategies: list, n: int) -> list:
        shared = build_shared_context(state, rivals, n, rng)
        results = simulate_strategies(state, strategies, rivals, shared, [overrides[s] for s in strategies])
        return [score_strategy(r) for r in results]

    finalists = candidates
    if len(candidates) > FINALISTS and n_simulations > SCREEN_SIMULATIONS:
        screened = sorted(zip(score_all(candidates, SCREEN_SIMULATIONS), candidates), key=lambda x: x[0].strategy_score, reverse=True)
        finalists = [strategy for _, strategy in screened[:FINALISTS]]

    scored = score_all(finalists, n_simulations)
    scored.sort(key=lambda s: s.strategy_score, reverse=True)
    return build_recommendation(state, scored, n_simulations), len(candidates)


def _load_demo_state(race_id: str, lap_number: int, driver_id: str) -> RaceState:
    from models.common.data import load_race_features

    df = load_race_features()
    race_total_laps = int(df[df.race_id == race_id]["lap_number"].max())
    row = df[(df.race_id == race_id) & (df.lap_number == lap_number) & (df.driver_id == driver_id)]
    if row.empty:
        raise ValueError(f"No row for {driver_id} at {race_id} lap {lap_number}")
    return RaceState.from_gold_row(row.iloc[0], race_total_laps=race_total_laps)


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run the strategy engine against a real historical snapshot")
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--lap", type=int, required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--n-simulations", type=int, default=5000)
    args = parser.parse_args()

    state = _load_demo_state(args.race_id, args.lap, args.driver)
    rivals = build_field_snapshot_from_gold(args.race_id, args.lap, exclude_driver_id=args.driver)

    start = time.perf_counter()
    recommendation, n_candidates = recommend_strategy(
        state,
        rivals,
        n_simulations=args.n_simulations,
        pace_anchor=driver_recent_pace(args.race_id, args.lap, args.driver),
        season=int(args.race_id.split("_")[0]),
        exclude_race_id=args.race_id,
    )
    elapsed = time.perf_counter() - start

    print(f"Evaluated {n_candidates} candidates x {args.n_simulations} simulations in {elapsed:.2f}s")
    print(json.dumps(recommendation, indent=2, default=str))


if __name__ == "__main__":
    main()
