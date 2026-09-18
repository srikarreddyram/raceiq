"""Bridges the strategy engine to the LSTM lap-time model
(models/lap_time_sequence/) — the pace oracle actually usable for
multi-lap-ahead simulation, unlike the LightGBM model used elsewhere in
this project (see simulation/monte_carlo.py's docstring on why chaining
that model's own predictions across 30+ laps proved chaotic, not just
noisy).

The LSTM has no such problem by construction: it takes a candidate
strategy's *entire* deterministic tyre/pit covariate sequence (known in
advance — see search/candidates.py) and predicts every lap's time in one
forward pass, with no self-referential feedback loop to go unstable.
Both the "green flag" and "safety car" trajectories for a candidate are
produced by one batched call (batch of 2: same covariates, only the
safety-car flag differs), not a per-lap loop.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch

from models.common.data import load_race_features
from models.common.registry import load_latest_pytorch_model
from models.lap_time_sequence.data import (
    CATEGORICAL_FEATURES,
    SEQUENCE_BOOLEAN_FEATURES,
    SEQUENCE_NUMERIC_FEATURES,
    STATIC_NUMERIC_FEATURES,
    Vocab,
    build_vocabs,
)
from models.lap_time_sequence.train import EXPERIMENT_NAME, RUN_NAME
from strategy_engine.state import RaceState


@lru_cache(maxsize=None)
def _vocabs() -> dict[str, Vocab]:
    # Recomputed from Gold rather than persisted alongside the model
    # weights — the same approach as model_features.py's reference
    # categorical dtypes, for the same reason: this vocabulary is a
    # deterministic function of the training data, so recomputing it is
    # simpler and can't silently drift out of sync with a separately
    # saved artifact.
    return build_vocabs(load_race_features())


@lru_cache(maxsize=None)
def _model():
    model = load_latest_pytorch_model(EXPERIMENT_NAME, RUN_NAME)
    model.eval()
    return model


def predict_trajectory(state: RaceState, plan: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Returns (green_times, sc_times), each shape (n_laps,) — the full
    remaining-race lap-time trajectory under a candidate strategy's plan,
    predicted in a single forward pass (batch of 2: green and SC variants
    share every feature except the safety-car flag).
    """
    vocabs = _vocabs()
    model = _model()
    n_laps = len(plan)

    # NaN is truthy in Python (`float('nan') or 0.0` evaluates to `nan`, not
    # `0.0`), so a per-field `x or default` fallback silently fails to
    # catch it — the leader's gap_to_car_ahead is exactly this case (NaN,
    # not None, since there's no car ahead). LightGBM handles NaN natively
    # (oracles.py passes it straight through), but a plain neural net does
    # not: one NaN input propagates NaN through every downstream
    # computation. Building the array first and calling np.nan_to_num once
    # at the end — the same approach data.py uses for training — sidesteps
    # the truthiness trap entirely rather than trying to catch it field by
    # field.
    numeric = np.zeros((n_laps, len(SEQUENCE_NUMERIC_FEATURES)), dtype=np.float32)
    for i, lap in enumerate(plan):
        numeric[i] = [
            lap["tyre_age"],
            lap["stint_number"],
            lap["degradation_rate"],
            lap["grip_estimate"],
            state.race_total_laps - lap["lap_number"],
            state.track_temp,
            state.air_temp,
            state.humidity,
            state.wind_speed,
            state.gap_to_car_ahead,
            state.gap_to_car_behind,
            state.current_position,
            state.field_avg_lap_time_seconds,
        ]
    numeric = np.nan_to_num(numeric, nan=0.0)

    is_pit_lap = np.array([[lap["is_pit_lap"] for lap in plan]], dtype=np.float32).T
    static = np.nan_to_num(
        np.array(
            [
                state.driver_avg_pace_delta,
                state.driver_consistency_score,
                state.condition_delta,
                state.historical_sc_rate,
            ],
            dtype=np.float32,
        ),
        nan=0.0,
    )
    static_tiled = np.tile(static, (n_laps, 1))

    categorical = np.stack(
        [
            np.full(n_laps, vocabs["driver_id"].encode(state.driver_id)),
            np.full(n_laps, vocabs["team_id"].encode(state.team_id)),
            np.full(n_laps, vocabs["circuit_id"].encode(state.circuit_id)),
            np.array([vocabs["compound"].encode(lap["compound"]) for lap in plan]),
        ],
        axis=1,
    ).astype(np.int64)

    def build_features(safety_car_active: float) -> np.ndarray:
        booleans = np.concatenate(
            [
                is_pit_lap,
                np.full((n_laps, 1), safety_car_active, dtype=np.float32),
                np.zeros((n_laps, len(SEQUENCE_BOOLEAN_FEATURES) - 2), dtype=np.float32),  # yellow/vsc/red/rain
            ],
            axis=1,
        )
        return np.concatenate([numeric, booleans, static_tiled], axis=1)

    features_batch = np.stack([build_features(0.0), build_features(1.0)])  # (2, n_laps, num_features)
    categorical_batch = np.stack([categorical, categorical])

    with torch.no_grad():
        pred = model(torch.from_numpy(features_batch), torch.from_numpy(categorical_batch))

    green_times = pred[0].numpy()
    sc_times = pred[1].numpy()
    return green_times, sc_times
