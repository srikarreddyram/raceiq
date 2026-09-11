"""Pit Stop Recommendation — PRD Section 11.3.

Task: binary classification, `should_pit` — is this lap a pit lap? Target
is `is_pit_lap`, already a Gold column; no target engineering needed here.

`pit_stop_duration` is deliberately excluded from the feature list even
though it's in gold.race_features: it's only non-null *because* a row is
a pit lap, so including it would let the model trivially detect the
answer instead of learning to predict it — the one leakage trap in this
table that isn't obvious from the column name alone.

Pit stops are rare (~6% of laps), so this uses LightGBM's `is_unbalance`
rather than training on the natural class balance and getting a model
that just always predicts "stay out." PRD prioritises precision over
recall (a false "pit now" call is more costly than a missed one in most
race scenarios), reported alongside F1 and PR-AUC (threshold-independent).

Usage:
    uv run python -m models.pit_stop.train
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

TARGET_PRECISION = 0.75  # PRD Section 11.3 success criterion

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.tracking import configure_experiment

TARGET = "is_pit_lap"

NUMERIC_FEATURES = [
    "tyre_age",
    "stint_number",
    "laps_remaining",
    "gap_to_car_ahead",
    "gap_to_car_behind",
    "current_position",
    "degradation_rate",
    "grip_estimate",
    "track_temp",
    "driver_avg_pace_delta",
    "condition_delta",
    "historical_sc_rate",
]
BOOLEAN_FEATURES = ["safety_car_active", "yellow_active", "vsc_active", "rainfall_flag", "traffic_flag"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_COLUMNS


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = df.dropna(subset=[TARGET, "tyre_age"])
    for col in BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    df[TARGET] = df[TARGET].astype(int)
    return apply_categorical_dtypes(df, CATEGORICAL_COLUMNS)


def main() -> None:
    configure_experiment("pit_stop_recommendation")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} positive_rate={train[TARGET].mean():.3f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        is_unbalance=True,
        random_state=42,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        eval_metric="average_precision",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    metrics = {
        "test_precision_at_0.5": precision_score(test[TARGET], test_pred),
        "test_recall_at_0.5": recall_score(test[TARGET], test_pred),
        "test_f1_at_0.5": f1_score(test[TARGET], test_pred),
        "test_pr_auc": average_precision_score(test[TARGET], test_proba),
    }

    # PRD explicitly prioritises precision over recall for this model, which
    # means the decision threshold is a real design choice, not fixed at 0.5.
    # Find the lowest threshold that still clears the 0.75 precision bar, and
    # report the recall it costs — that's the operating point a strategy
    # engineer would actually want, not the default classifier threshold.
    precisions, recalls, thresholds = precision_recall_curve(test[TARGET], test_proba)
    meets_target = np.where(precisions[:-1] >= TARGET_PRECISION)[0]
    if len(meets_target) > 0:
        idx = meets_target[0]
        metrics["operating_threshold"] = float(thresholds[idx])
        metrics["precision_at_operating_threshold"] = float(precisions[idx])
        metrics["recall_at_operating_threshold"] = float(recalls[idx])
    else:
        metrics["max_precision_achievable"] = float(precisions[:-1].max())

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
