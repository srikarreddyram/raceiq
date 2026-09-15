"""Wraps the three trained models the Monte Carlo simulation calls every
simulated lap: Lap Time (the "time oracle", PRD 12.4 step 2), Tyre
Degradation (the stint-feasibility constraint), and Safety Car Probability
(the stochastic event sampler, PRD 12.4 step 1).

Each function builds a single-row DataFrame shaped exactly like that
model's training features from a `RaceState`, using `model_features.py`
to get the categorical encoding right, then predicts. Any feature a model
expects that a `RaceState` doesn't carry (because it's per-race-level,
like `season`, or genuinely absent, like the deferred PRD fields already
documented in each model's train.py) is filled with a neutral default
(0/False) rather than crashing — a simulated race is synthetic input by
definition, so it will never perfectly match the shape of a real training
row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.common.registry import load_latest_model
from models.lap_time.train import FEATURE_COLUMNS as LAP_TIME_FEATURES
from models.safety_car.train import FEATURE_COLUMNS as SAFETY_CAR_FEATURES
from models.tyre_degradation.train import FEATURE_COLUMNS as TYRE_FEATURES
from strategy_engine.model_features import apply_reference_categoricals
from strategy_engine.state import RaceState


def _row_from(state: RaceState, values: dict, columns: list[str]) -> pd.DataFrame:
    row = {col: values.get(col, 0) for col in columns}
    df = pd.DataFrame([row])
    return apply_reference_categoricals(df)


def predict_next_lap_time(state: RaceState) -> float:
    model = load_latest_model("lap_time_prediction")
    values = {
        "lap_time_seconds": state.lap_time_seconds,
        "field_avg_lap_time_seconds": state.field_avg_lap_time_seconds,
        "pace_delta_this_lap": state.lap_time_seconds - state.field_avg_lap_time_seconds,
        "laps_remaining": state.laps_remaining,
        "gap_to_car_ahead": state.gap_to_car_ahead,
        "gap_to_car_behind": state.gap_to_car_behind,
        "tyre_age": state.tyre_age,
        "stint_number": state.stint_number,
        "track_temp": state.track_temp,
        "air_temp": state.air_temp,
        "humidity": state.humidity,
        "wind_speed": state.wind_speed,
        "degradation_rate": state.degradation_rate,
        "grip_estimate": state.grip_estimate,
        "driver_avg_pace_delta": state.driver_avg_pace_delta,
        "driver_consistency_score": state.driver_consistency_score,
        "condition_delta": state.condition_delta,
        "current_position": state.current_position,
        "rival_tyre_age": state.rival_ahead.tyre_age if state.rival_ahead else np.nan,
        "is_pit_lap": 0,
        "safety_car_active": 0,
        "yellow_active": 0,
        "vsc_active": 0,
        "red_flag_active": 0,
        "traffic_flag": bool(state.gap_to_car_ahead is not None and state.gap_to_car_ahead < 1.0),
        "rainfall_flag": int(state.rainfall_flag),
        "driver_id": state.driver_id,
        "team_id": state.team_id,
        "circuit_id": state.circuit_id,
        "compound": state.compound,
        "rival_driver_id": state.rival_ahead.driver_id if state.rival_ahead else None,
        "rival_team_id": state.rival_ahead.team_id if state.rival_ahead else None,
        "rival_compound": state.rival_ahead.compound if state.rival_ahead else None,
    }
    row = _row_from(state, values, LAP_TIME_FEATURES)
    return float(model.predict(row)[0])


def predict_remaining_tyre_life(state: RaceState) -> float:
    model = load_latest_model("tyre_degradation")
    values = {
        "tyre_age": state.tyre_age,
        "stint_number": state.stint_number,
        "laps_remaining": state.laps_remaining,
        "track_temp": state.track_temp,
        "air_temp": state.air_temp,
        "humidity": state.humidity,
        "degradation_rate": state.degradation_rate,
        "grip_estimate": state.grip_estimate,
        "driver_avg_pace_delta": state.driver_avg_pace_delta,
        "driver_consistency_score": state.driver_consistency_score,
        "condition_delta": state.condition_delta,
        "rainfall_flag": int(state.rainfall_flag),
        "safety_car_active": 0,
        "yellow_active": 0,
        "driver_id": state.driver_id,
        "team_id": state.team_id,
        "circuit_id": state.circuit_id,
        "compound": state.compound,
        "rival_driver_id": state.rival_ahead.driver_id if state.rival_ahead else None,
        "rival_team_id": state.rival_ahead.team_id if state.rival_ahead else None,
        "rival_compound": state.rival_ahead.compound if state.rival_ahead else None,
    }
    row = _row_from(state, values, TYRE_FEATURES)
    return max(0.0, float(model.predict(row)[0]))


def predict_safety_car_probability(state: RaceState) -> float:
    model = load_latest_model("safety_car_probability")
    values = {
        "lap_number": state.current_lap,
        "laps_remaining": state.laps_remaining,
        "closest_gap_on_track": state.gap_to_car_ahead,
        "condition_delta": state.condition_delta,
        "historical_sc_rate": state.historical_sc_rate,
        "safety_car_active": 0,
        "yellow_active": 0,
        "vsc_active": 0,
        "rainfall_flag": int(state.rainfall_flag),
        "circuit_id": state.circuit_id,
    }
    row = _row_from(state, values, SAFETY_CAR_FEATURES)
    return float(model.predict_proba(row)[0, 1])
