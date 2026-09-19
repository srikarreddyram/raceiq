"""Strategy scoring — PRD Section 12.5/12.6's output table, computed
directly from a `SimulationResult`'s distribution of simulated finishing
positions.

One deliberate deviation from the PRD's literal scoring formula:

    Strategy Score = (Expected Points x 0.6) + (Podium Probability x 0.3) + (Win Probability x 0.1)

taken at face value, `Expected Points` (0-25) completely swamps the two
probabilities (0-1) — a strategy scoring 18 expected points would already
score 10.8 before the other two terms add at most 0.4. But the PRD's own
worked example (Section 12.6) shows `expected_points: 18` alongside
`strategy_score: 0.812`, which is only reachable if expected points is
normalised to a 0-1 scale first (dividing by the maximum, 25) before
weighting — so that's what this does. It's an interpretation of an
internally-inconsistent spec example, not a guess made up from nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategy_engine.simulation.monte_carlo import SimulationResult

POINTS_TABLE = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
MAX_POINTS = 25

SCORE_WEIGHTS = {"expected_points": 0.6, "podium_probability": 0.3, "win_probability": 0.1}


@dataclass
class StrategyScore:
    label: str
    # The machine-readable form of `label`. `label` is display text ("Pit
    # lap 21 -> HARD, lap 41 -> SOFT"); carrying the plan itself alongside
    # it means an API client wanting to re-simulate or chart a strategy
    # doesn't have to parse that sentence back into data.
    pit_plan: tuple[tuple[int, str], ...]
    win_probability: float
    podium_probability: float
    points_probability: float
    expected_finish: float
    expected_points: float
    finish_distribution: dict[int, float]
    risk_score: float
    strategy_score: float
    safety_car_encounter_rate: float


def score_strategy(result: SimulationResult, weights: dict[str, float] = SCORE_WEIGHTS) -> StrategyScore:
    positions = result.final_positions
    n = len(positions)
    points = np.array([POINTS_TABLE.get(int(p), 0) for p in positions])

    finish_distribution = {
        p: float((positions == p).sum()) / n for p in range(1, int(positions.max()) + 1)
    }
    win_probability = float((positions == 1).mean())
    podium_probability = float((positions <= 3).mean())
    points_probability = float((positions <= 10).mean())
    expected_points = float(points.mean())

    strategy_score = (
        weights["expected_points"] * (expected_points / MAX_POINTS)
        + weights["podium_probability"] * podium_probability
        + weights["win_probability"] * win_probability
    )

    return StrategyScore(
        label=result.strategy.label,
        pit_plan=result.strategy.pit_plan,
        win_probability=win_probability,
        podium_probability=podium_probability,
        points_probability=points_probability,
        expected_finish=float(positions.mean()),
        expected_points=expected_points,
        finish_distribution=finish_distribution,
        risk_score=float(positions.var()),
        strategy_score=strategy_score,
        safety_car_encounter_rate=float(result.safety_car_occurred.mean()),
    )
