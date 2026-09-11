"""Safety Car Probability — PRD Section 11.4.

Task: binary classification, `safety_car_within_N_laps` (N=5 default).

Unlike the other five models, this one is trained at (race_id, lap_number)
grain, not per-driver-per-lap: a safety car is a track-wide event, not
something any single driver's gap-to-car-ahead or tyre age explains, so
keeping ~20 duplicate driver-rows per lap would just couple the target to
driver-specific features that have no real causal bearing on it.
Driver-level signals are collapsed to one race-wide number instead —
`closest_gap_on_track` (the minimum gap-to-car-ahead across the whole
field at that lap) is the closest available proxy for "cars running close
together," which the PRD's `gap_between_cars` names without fully
specifying.

`incident_rate_this_race` (a PRD-named key feature) is dropped — no
incident data is ingested (see driver_history.py's docstring on
`aggression_score` for the same gap). `rain_probability` uses the current
`rainfall_flag` rather than a forecast, for the same reason
`rain_probability_next_10_laps` was dropped from lap_features.py: there's
no real forecast source, only observed history.

Usage:
    uv run python -m models.safety_car.train
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from models.common.data import load_race_features
from models.common.features import apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_safety_car_within_n_target
from models.common.tracking import configure_experiment

TARGET = "safety_car_within_n_laps"
CATEGORICAL_COLUMNS = ["circuit_id"]
NUMERIC_FEATURES = [
    "lap_number",
    "laps_remaining",
    "closest_gap_on_track",
    "condition_delta",
    "historical_sc_rate",
]
BOOLEAN_FEATURES = ["safety_car_active", "yellow_active", "vsc_active", "rainfall_flag"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_COLUMNS


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = add_safety_car_within_n_target(df, n=5)

    race_level = (
        df.groupby(["race_id", "lap_number"])
        .agg(
            season=("season", "first"),
            circuit_id=("circuit_id", "first"),
            laps_remaining=("laps_remaining", "first"),
            condition_delta=("condition_delta", "first"),
            historical_sc_rate=("historical_sc_rate", "first"),
            safety_car_active=("safety_car_active", "max"),
            yellow_active=("yellow_active", "max"),
            vsc_active=("vsc_active", "max"),
            rainfall_flag=("rainfall_flag", "max"),
            closest_gap_on_track=("gap_to_car_ahead", "min"),
            **{TARGET: (TARGET, "max")},
        )
        .reset_index()
    )
    for col in BOOLEAN_FEATURES:
        race_level[col] = race_level[col].astype(int)
    race_level[TARGET] = race_level[TARGET].astype(int)
    return apply_categorical_dtypes(race_level, CATEGORICAL_COLUMNS)


def main() -> None:
    configure_experiment("safety_car_probability")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} positive_rate={train[TARGET].mean():.3f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        is_unbalance=True,
        random_state=42,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        eval_metric="auc",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)
    metrics = {
        "test_auc_roc": roc_auc_score(test[TARGET], test_proba),
        "test_f1": f1_score(test[TARGET], test_pred),
        "test_pr_auc": average_precision_score(test[TARGET], test_proba),
    }

    # A large chunk of predictive power here could just be "an already-active
    # SC/yellow/VSC tends to still be active 5 laps later," not genuine
    # onset forecasting. Report AUC restricted to laps that are currently
    # green-flag as the honest measure of forecasting a *new* event.
    currently_green = (
        (test["safety_car_active"] == 0) & (test["yellow_active"] == 0) & (test["vsc_active"] == 0)
    )
    metrics["test_auc_roc_onset_only"] = roc_auc_score(
        test.loc[currently_green, TARGET], test_proba[currently_green.to_numpy()]
    )

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
