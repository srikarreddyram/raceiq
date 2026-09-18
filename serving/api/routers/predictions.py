"""POST /predict/laptime, /predict/strategy/optimal, /simulate — PRD
Section 14.

Every endpoint here takes a `(race_id, lap_number, driver_id)` pointer
into real historical data rather than a fully live race state — honestly,
because that's the only race state this project actually has: there's no
live telemetry feed integrated (PRD Section 4 scopes that out), and no
race is happening right now to source a genuinely live state from. A
future live-serving version would replace `RaceState.from_gold_row` with
a request payload built from real-time telemetry; the strategy engine
underneath (search -> simulate -> score -> recommend) doesn't change.
"""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from models.common.data import load_race_features
from strategy_engine.engine import recommend_strategy
from strategy_engine.field import build_field_snapshot_from_gold
from strategy_engine.oracles import predict_next_lap_time
from strategy_engine.scoring.score import score_strategy
from strategy_engine.search.candidates import Strategy
from strategy_engine.simulation.monte_carlo import build_shared_context, simulate_strategy
from strategy_engine.state import RaceState
from serving.api.schemas import (
    LapTimePredictionRequest,
    LapTimePredictionResponse,
    ScoredStrategy,
    SimulateRequest,
    SimulateResponse,
    StrategyRecommendation,
    StrategyRequest,
)

router = APIRouter(tags=["predictions"])


def _load_state(race_id: str, lap_number: int, driver_id: str) -> RaceState:
    df = load_race_features()
    race_total_laps = df.loc[df.race_id == race_id, "lap_number"].max()
    if np.isnan(race_total_laps):
        raise HTTPException(status_code=404, detail=f"Unknown race_id: {race_id!r}")

    row = df[(df.race_id == race_id) & (df.lap_number == lap_number) & (df.driver_id == driver_id)]
    if row.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data for driver_id={driver_id!r} at race_id={race_id!r} lap {lap_number}",
        )
    return RaceState.from_gold_row(row.iloc[0], race_total_laps=int(race_total_laps))


@router.post("/predict/laptime", response_model=LapTimePredictionResponse)
def predict_laptime(request: LapTimePredictionRequest) -> LapTimePredictionResponse:
    state = _load_state(request.race_id, request.lap_number, request.driver_id)
    predicted = predict_next_lap_time(state)
    return LapTimePredictionResponse(
        driver_id=request.driver_id,
        lap_number=request.lap_number,
        predicted_next_lap_time_seconds=predicted,
        actual_current_lap_time_seconds=state.lap_time_seconds,
    )


@router.post("/predict/strategy/optimal", response_model=StrategyRecommendation)
def predict_strategy_optimal(request: StrategyRequest) -> StrategyRecommendation:
    state = _load_state(request.race_id, request.lap_number, request.driver_id)
    rivals = build_field_snapshot_from_gold(request.race_id, request.lap_number, request.driver_id)

    try:
        recommendation, n_candidates = recommend_strategy(state, rivals, n_simulations=request.n_simulations)
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return StrategyRecommendation(
        circuit_id=recommendation["circuit_id"],
        driver_id=recommendation["driver_id"],
        team_id=recommendation["team_id"],
        current_lap=recommendation["current_lap"],
        laps_remaining=recommendation["laps_remaining"],
        n_simulations=request.n_simulations,
        n_candidates_evaluated=n_candidates,
        win_probability_model_estimate=recommendation["model_cross_checks"]["win_probability_model_estimate"],
        recommended_strategy=ScoredStrategy(**recommendation["recommended_strategy"]),
        reasoning=recommendation["reasoning"],
        alternatives=[ScoredStrategy(**alt) for alt in recommendation["alternatives"]],
    )


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    state = _load_state(request.race_id, request.lap_number, request.driver_id)
    rivals = build_field_snapshot_from_gold(request.race_id, request.lap_number, request.driver_id)

    strategy = Strategy(
        pit_plan=tuple(request.pit_plan),
        label=", ".join(f"Pit lap {lap} -> {compound}" for lap, compound in request.pit_plan) or "No further stops",
    )
    rng = np.random.default_rng()
    shared = build_shared_context(state, request.n_simulations, rng)
    result = simulate_strategy(state, strategy, rivals, shared)
    scored = score_strategy(result)

    return SimulateResponse(
        action=scored.label,
        win_probability=scored.win_probability,
        podium_probability=scored.podium_probability,
        points_probability=scored.points_probability,
        expected_finish=scored.expected_finish,
        expected_points=scored.expected_points,
        risk_score=scored.risk_score,
        finish_distribution=scored.finish_distribution,
    )
