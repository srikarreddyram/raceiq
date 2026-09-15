"""Lap Time Prediction — PRD Section 11.1.

Task: regression, predicting `next_lap_time_seconds` from the current
lap's state. This is the "time oracle" the strategy engine's Monte Carlo
simulation will call lap-by-lap under alternative tyre strategies (PRD
Section 12.4), so its accuracy caps how trustworthy every simulated
strategy comparison downstream can be.

`fuel_load_estimate` (a PRD-named key feature) isn't built — no fuel model
exists yet — so `laps_remaining` stands in as the fuel proxy: fuel burns
off roughly linearly over a race, and laps_remaining is monotonic with it.

The current lap's own `lap_time_seconds` is deliberately included as a
feature: consecutive green-flag laps are highly autocorrelated, and using
the current lap to help predict the next one is a legitimate autoregressive
feature, not leakage — the target is the *next* lap, a genuinely unknown
value at prediction time.

Trains all three algorithms PRD Section 11.1 names (LightGBM primary,
XGBoost and CatBoost for comparison), evaluates RMSE/MAE on validation and
test, and logs everything to MLflow.

Usage:
    uv run python -m models.lap_time.train
"""

from __future__ import annotations

import catboost
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_next_lap_time_target
from models.common.tracking import configure_experiment

TARGET = "next_lap_time_seconds"

NUMERIC_FEATURES = [
    "lap_time_seconds",
    "field_avg_lap_time_seconds",
    "pace_delta_this_lap",
    "laps_remaining",
    "gap_to_car_ahead",
    "gap_to_car_behind",
    "tyre_age",
    "stint_number",
    "track_temp",
    "air_temp",
    "humidity",
    "wind_speed",
    "degradation_rate",
    "grip_estimate",
    "driver_avg_pace_delta",
    "driver_consistency_score",
    "condition_delta",
    "current_position",
    "rival_tyre_age",
]
BOOLEAN_FEATURES = [
    "is_pit_lap",
    "safety_car_active",
    "yellow_active",
    "vsc_active",
    "red_flag_active",
    "traffic_flag",
    "rainfall_flag",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_COLUMNS

# Flags used only to build the "stable regime" evaluation slice below — not
# used as model features, since they describe the *next* lap and would be
# unknown at prediction time.
_DISRUPTION_FLAGS = ["is_pit_lap", "safety_car_active", "yellow_active", "vsc_active", "red_flag_active"]


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = df.sort_values(["race_id", "driver_id", "lap_number"])
    for flag in _DISRUPTION_FLAGS:
        df[f"next_{flag}"] = df.groupby(["race_id", "driver_id"])[flag].shift(-1)
    df = add_next_lap_time_target(df)
    df = df.dropna(subset=[TARGET, "lap_time_seconds"])
    # A red-flag-affected lap's recorded time spans the full session-clock
    # stoppage, not real pace (one 2024 race shows every driver's lap 1 at
    # ~2,500 seconds) — since the target here is the *next* lap's time, a
    # row is unusable whenever the lap it's predicting into was red-flagged,
    # even if the row's own current lap looks perfectly normal.
    df = df[df["next_red_flag_active"] != 1]
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    df = apply_categorical_dtypes(df, CATEGORICAL_COLUMNS)
    return df


def stable_regime_mask(df: pd.DataFrame) -> pd.Series:
    """Rows where neither the current nor the next lap involved a pit stop,
    safety car, yellow, VSC, or red flag — the fairest apples-to-apples
    slice for comparing against published lap-time-prediction benchmarks,
    since the PRD's raw RMSE is otherwise dominated by a small number of
    laps with 30-50s swings that no current-lap feature set can foresee.
    """
    mask = pd.Series(True, index=df.index)
    for flag in _DISRUPTION_FLAGS:
        mask &= df[flag] == 0
        mask &= df[f"next_{flag}"] == 0
    return mask


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_pred = model.predict(val[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])
    metrics = {
        "val_rmse": _rmse(val[TARGET], val_pred),
        "val_mae": mean_absolute_error(val[TARGET], val_pred),
        "test_rmse": _rmse(test[TARGET], test_pred),
        "test_mae": mean_absolute_error(test[TARGET], test_pred),
    }

    stable = stable_regime_mask(test)
    metrics["test_rmse_stable_regime"] = _rmse(test.loc[stable, TARGET], test_pred[stable.to_numpy()])
    metrics["test_mae_stable_regime"] = mean_absolute_error(
        test.loc[stable, TARGET], test_pred[stable.to_numpy()]
    )
    metrics["test_n_stable_regime"] = int(stable.sum())

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.set_tag("role", "primary")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")
    return metrics


def train_xgboost(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        enable_categorical=True,
        tree_method="hist",
        early_stopping_rounds=50,
        random_state=42,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        verbose=False,
    )
    val_pred = model.predict(val[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])
    metrics = {
        "val_rmse": _rmse(val[TARGET], val_pred),
        "val_mae": mean_absolute_error(val[TARGET], val_pred),
        "test_rmse": _rmse(test[TARGET], test_pred),
        "test_mae": mean_absolute_error(test[TARGET], test_pred),
    }
    with mlflow.start_run(run_name="xgboost"):
        mlflow.set_tag("algorithm", "xgboost")
        mlflow.set_tag("role", "comparison")
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, name="model")
    return metrics


def train_catboost(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    # CatBoost wants its categorical columns as strings, not pandas `category` dtype.
    cat_idx = [FEATURE_COLUMNS.index(c) for c in CATEGORICAL_COLUMNS]

    def as_catboost(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame[FEATURE_COLUMNS].copy()
        for col in CATEGORICAL_COLUMNS:
            frame[col] = frame[col].astype(str)
        return frame

    model = catboost.CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=8,
        cat_features=cat_idx,
        random_seed=42,
        early_stopping_rounds=50,
        verbose=False,
    )
    model.fit(as_catboost(train), train[TARGET], eval_set=(as_catboost(val), val[TARGET]))
    val_pred = model.predict(as_catboost(val))
    test_pred = model.predict(as_catboost(test))
    metrics = {
        "val_rmse": _rmse(val[TARGET], val_pred),
        "val_mae": mean_absolute_error(val[TARGET], val_pred),
        "test_rmse": _rmse(test[TARGET], test_pred),
        "test_mae": mean_absolute_error(test[TARGET], test_pred),
    }
    with mlflow.start_run(run_name="catboost"):
        mlflow.set_tag("algorithm", "catboost")
        mlflow.set_tag("role", "comparison")
        mlflow.log_metrics(metrics)
        mlflow.catboost.log_model(model, name="model")
    return metrics


def main() -> None:
    configure_experiment("lap_time_prediction")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    results = {
        "lightgbm": train_lightgbm(train, val, test),
        "xgboost": train_xgboost(train, val, test),
        "catboost": train_catboost(train, val, test),
    }
    for name, metrics in results.items():
        print(f"{name}: {metrics}")


if __name__ == "__main__":
    main()
