"""Wraps every trained model the strategy engine calls on a `RaceState`:
the three the Monte Carlo simulation calls every simulated lap — Lap Time
(the "time oracle", PRD 12.4 step 2), Tyre Degradation (the stint-
feasibility constraint), and Safety Car Probability (the stochastic event
sampler, PRD 12.4 step 1) — plus Win Probability and Final Race Position,
called once per recommendation (not per simulated lap) purely as
independent cross-checks against the simulation's own output. See each
`predict_*_now` function's docstring for why that comparison is useful.

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
from models.race_position.train import FEATURE_COLUMNS as RACE_POSITION_FEATURES
from models.safety_car.train import FEATURE_COLUMNS as SAFETY_CAR_FEATURES
from models.tyre_degradation.train import FEATURE_COLUMNS as TYRE_FEATURES
from models.win_probability.train import FEATURE_COLUMNS as WIN_PROBABILITY_FEATURES
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


def predict_win_probability_now(state: RaceState) -> float:
    """The trained Win Probability classifier's estimate for the driver's
    *actual current* race state — independent of any candidate strategy or
    simulation. Used purely as a cross-check alongside the Monte Carlo
    simulation's own win-probability output (see recommendation/reasoning.py):
    the classifier was trained on real historical (position, gap, tyre age,
    ...) snapshots and their eventual outcomes, so a large disagreement
    between "what history says a driver in this exact spot usually does" and
    "what the simulation predicts for the recommended strategy" is a useful
    signal, not something to silently reconcile.
    """
    model = load_latest_model("win_probability")
    values = {
        "current_position": state.current_position,
        "gap_to_leader": state.gap_to_leader,
        "laps_remaining": state.laps_remaining,
        "tyre_age": state.tyre_age,
        "degradation_rate": state.degradation_rate,
        "driver_avg_pace_delta": state.driver_avg_pace_delta,
        "driver_overtaking_score": state.driver_overtaking_score,
        "condition_delta": state.condition_delta,
        "safety_car_active": 0,
        "is_pit_lap": 0,
        "driver_id": state.driver_id,
        "team_id": state.team_id,
        "circuit_id": state.circuit_id,
        "compound": state.compound,
        "rival_driver_id": state.rival_ahead.driver_id if state.rival_ahead else None,
        "rival_team_id": state.rival_ahead.team_id if state.rival_ahead else None,
        "rival_compound": state.rival_ahead.compound if state.rival_ahead else None,
    }
    row = _row_from(state, values, WIN_PROBABILITY_FEATURES)
    return float(model.predict_proba(row)[0, 1])


def predict_expected_finish_now(state: RaceState) -> float:
    """The trained Final Race Position classifier's estimate for the
    driver's *actual current* race state — same cross-check role as
    `predict_win_probability_now`, on a different metric. Final Race
    Position is multiclass (one probability per finishing position), so
    "expected finish" here is the probability-weighted mean position
    (sum_p position * P(position)) rather than the single most-likely
    class — consistent with how the Monte Carlo simulation's own
    `expected_finish` is computed (an average across simulated outcomes,
    not a mode), so the two numbers are actually comparable.
    """
    model = load_latest_model("final_race_position")
    values = {
        "current_position": state.current_position,
        "gap_to_leader": state.gap_to_leader,
        "laps_remaining": state.laps_remaining,
        "tyre_age": state.tyre_age,
        "degradation_rate": state.degradation_rate,
        "driver_avg_pace_delta": state.driver_avg_pace_delta,
        "driver_overtaking_score": state.driver_overtaking_score,
        "condition_delta": state.condition_delta,
        "safety_car_active": 0,
        "is_pit_lap": 0,
        "driver_id": state.driver_id,
        "team_id": state.team_id,
        "circuit_id": state.circuit_id,
        "compound": state.compound,
        "rival_driver_id": state.rival_ahead.driver_id if state.rival_ahead else None,
        "rival_team_id": state.rival_ahead.team_id if state.rival_ahead else None,
        "rival_compound": state.rival_ahead.compound if state.rival_ahead else None,
    }
    row = _row_from(state, values, RACE_POSITION_FEATURES)
    proba = model.predict_proba(row)[0]
    return float(np.dot(model.classes_, proba))
