"""Win Probability — PRD Section 11.6.

Task: binary classification, `wins_race`. Same per-lap grain and rationale
as Final Race Position (models/race_position/train.py) — a live estimate
that updates lap by lap, not a single pre-race number. Shares its target
source (`add_race_outcome_targets`) with that model since `wins_race` is
literally `final_position == 1`.

Evaluated per PRD: log-loss and AUC-ROC. No numeric target is set for
this model in PRD Section 19's success criteria (only Lap Time, Tyre
Degradation, Pit Stop, and Safety Car Probability have named thresholds),
so results are reported without a pass/fail bar.

Usage:
    uv run python -m models.win_probability.train
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_race_outcome_targets
from models.common.tracking import configure_experiment

TARGET = "wins_race"

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
# historical_dnf_rate (added to strategy_engine's Monte Carlo simulation
# to model rival attrition — see simulation/monte_carlo.py) was tried here
# too, on the theory that a driver's win chances depend partly on how many
# rivals are likely to retire ahead of them. Tested honestly on held-out
# 2025 data and reverted: it made this model slightly worse, not better
# (log_loss 0.109 -> 0.121, AUC 0.964 -> 0.960, same train/test split,
# otherwise-identical run). Most likely reason: it's a single coarse
# circuit+era-level scalar with no per-lap variation and no interaction
# with laps_remaining, unlike the strategy engine's own use of it (which
# explicitly converts it to a remaining-laps probability) — too weak and
# collinear with circuit_id/season for a tree model to extract real signal
# from, so it just adds a dimension to overfit on.
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
    configure_experiment("win_probability")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} positive_rate={train[TARGET].mean():.3f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        is_unbalance=True,
        random_state=42,
        # Defensive reproducibility pin (PRD's own tracking goal — see
        # models/common/tracking.py): random_state alone doesn't fully
        # pin LightGBM's multi-threaded histogram construction, since
        # floating-point summation order can vary with thread scheduling.
        # NOTE: investigating a real, large prediction swing for an actual
        # race leader (0.61 vs 0.24 win probability across two retrains)
        # initially pointed here, but retraining with these two flags set
        # reproduced the exact same (0.24) result — ruling this out. The
        # actual cause was a categorical-vocabulary bug, now fixed in
        # models/common/features.py and strategy_engine/model_features.py.
        # Left in anyway as a legitimate best practice; every other model
        # in models/ got the same two flags for the same reason.
        deterministic=True,
        force_row_wise=True,
    )
    model.fit(
        train[FEATURE_COLUMNS],
        train[TARGET],
        eval_set=[(val[FEATURE_COLUMNS], val[TARGET])],
        eval_metric="binary_logloss",
        categorical_feature=CATEGORICAL_COLUMNS,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test_proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    metrics = {
        "test_log_loss": log_loss(test[TARGET], test_proba),
        "test_auc_roc": roc_auc_score(test[TARGET], test_proba),
    }

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.set_tag("algorithm", "lightgbm")
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
