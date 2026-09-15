"""Top-level entry point — ties search, simulation, scoring, and
recommendation together into the single call PRD Section 12's
`/predict/strategy/optimal` endpoint (Section 14) will eventually wrap.

Usage:
    uv run python -m strategy_engine.engine --race-id 2025_4 --lap 20 --driver piastri
"""

from __future__ import annotations

import time

import numpy as np

from strategy_engine.field import build_field_snapshot_from_gold
from strategy_engine.scoring.score import score_strategy
from strategy_engine.search.candidates import generate_candidates
from strategy_engine.simulation.monte_carlo import build_shared_context, simulate_strategy
from strategy_engine.state import RaceState


def recommend_strategy(state: RaceState, rivals: list, n_simulations: int = 5000, seed: int = 42) -> dict:
    from strategy_engine.recommendation.reasoning import build_recommendation

    rng = np.random.default_rng(seed)
    candidates = generate_candidates(state)
    if not candidates:
        raise RuntimeError("No feasible strategy candidates found for this race state")

    # Built once and reused for every candidate — see monte_carlo.py's
    # docstring on why comparing strategies needs shared randomness.
    shared = build_shared_context(state, n_simulations, rng)

    scored = []
    for strategy in candidates:
        result = simulate_strategy(state, strategy, rivals, shared)
        scored.append(score_strategy(result))

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
    recommendation, n_candidates = recommend_strategy(state, rivals, n_simulations=args.n_simulations)
    elapsed = time.perf_counter() - start

    print(f"Evaluated {n_candidates} candidates x {args.n_simulations} simulations in {elapsed:.2f}s")
    print(json.dumps(recommendation, indent=2, default=str))


if __name__ == "__main__":
    main()
