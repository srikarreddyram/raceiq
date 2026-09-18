"""Tyre Degradation — PRD Section 11.2.

Task: regression, predicting `predicted_remaining_life_laps` — how many
more laps the current tyre set has left in its current stint. This feeds
the Pit Stop Recommendation model (can this driver extend?) and the
strategy engine's Monte Carlo simulation as a hard constraint on which
stint lengths are even feasible.

Two PRD-named key features aren't available and are substituted or
dropped, both noted where they'd otherwise be: `driver_tyre_conservation_index`
(PRD 8.1's teammate-relative degradation metric, deferred — see
pipelines/gold/driver_history.py's docstring) falls back to the driver's
own historical `driver_avg_pace_delta`/`driver_consistency_score`, and
`tyre_stress_index` (a circuit property, needs track_maps/) is dropped
entirely rather than faked.

Usage:
    uv run python -m models.tyre_degradation.train
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_remaining_tyre_life_target
from models.common.tracking import configure_experiment

TARGET = "predicted_remaining_life_laps"

NUMERIC_FEATURES = [
    "tyre_age",
    "stint_number",
    "laps_remaining",
    "track_temp",
    "air_temp",
    "humidity",
    "degradation_rate",
    "grip_estimate",
    "driver_avg_pace_delta",
    "driver_consistency_score",
    "condition_delta",
]
BOOLEAN_FEATURES = ["rainfall_flag", "safety_car_active", "yellow_active"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_COLUMNS


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = add_remaining_tyre_life_target(df)
    df = df.dropna(subset=[TARGET, "tyre_age", "compound"])
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    return apply_categorical_dtypes(df, CATEGORICAL_COLUMNS)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    configure_experiment("tyre_degradation")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        # See models/win_probability/train.py's comment: random_state
        # alone doesn't make LightGBM's multi-threaded training fully
        # reproducible — pinned here for the same reason.
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        eval_metric="rmse",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test = test.copy()
    test["pred"] = model.predict(test[FEATURE_COLUMNS])
    metrics = {
        "test_rmse": _rmse(test[TARGET], test["pred"]),
        "test_mae": mean_absolute_error(test[TARGET], test["pred"]),
    }
    # PRD success criterion is phrased "within 2 laps on average" per compound.
    per_compound_mae = test.groupby("compound", observed=True).apply(
        lambda g: mean_absolute_error(g[TARGET], g["pred"]), include_groups=False
    )
    for compound, mae in per_compound_mae.items():
        metrics[f"test_mae_{str(compound).lower()}"] = float(mae)

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
