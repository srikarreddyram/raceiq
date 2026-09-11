"""Final Race Position — PRD Section 11.5.

Task: multi-class classification, predicting a driver's final classified
position (P1-P20ish) from their state at *any* lap of the race — this is
intentionally kept at per-lap grain (unlike Safety Car Probability): the
whole point is a live "if the race ended on its current trajectory, how
does this driver finish" estimate that updates lap by lap, which is
exactly what a strategy dashboard needs, not a single pre-race prediction.

Evaluated per PRD: top-3 accuracy and mean absolute position error (MAE
between the argmax-predicted position and the actual one).

Usage:
    uv run python -m models.race_position.train
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, top_k_accuracy_score

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_race_outcome_targets
from models.common.tracking import configure_experiment

TARGET = "final_position"

NUMERIC_FEATURES = [
    "current_position",
    "gap_to_leader",
    "laps_remaining",
    "tyre_age",
    "degradation_rate",
    "driver_avg_pace_delta",
    "driver_overtaking_score",
    "condition_delta",
]
BOOLEAN_FEATURES = ["safety_car_active", "is_pit_lap"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_COLUMNS


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = add_race_outcome_targets(df)
    df = df.dropna(subset=[TARGET, "current_position"])
    df[TARGET] = df[TARGET].astype(int)
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    return apply_categorical_dtypes(df, CATEGORICAL_COLUMNS)


def main() -> None:
    configure_experiment("final_race_position")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} classes={sorted(train[TARGET].unique())}")

    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])
    test_pred = model.predict(test[FEATURE_COLUMNS])
    metrics = {
        "test_top3_accuracy": top_k_accuracy_score(
            test[TARGET], test_proba, k=3, labels=model.classes_
        ),
        "test_mean_abs_position_error": mean_absolute_error(test[TARGET], test_pred),
    }

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
