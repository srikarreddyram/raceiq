"""Tyre Degradation, reframed as a discrete-time hazard/survival model —
an alternative to `train.py`'s direct regression, logged to the same
MLflow experiment for an honest side-by-side comparison.

The problem with `train.py`'s target (`predicted_remaining_life_laps` =
stint's final tyre age minus current tyre age): it's measuring when the
TEAM chose to pit, not when the TYRE actually stopped performing. Two
drivers on identical compounds, ages, and track conditions can have wildly
different "remaining life" purely because one team called an early
undercut and the other didn't — that's strategic confounding baked
straight into the regression target, and it's the leading suspect for why
that model's MAE (~5.7 laps) misses the PRD's <2-lap target by so much.

This is exactly the "censored data" problem survival analysis exists for:
most stints don't end because the tyre physically failed, they end
because a strategist decided to end them (voluntary censoring). The fix
isn't a better regressor on the same target — it's a different model
shape. This fits a discrete-time hazard function

    h(age) = P(pit occurs at this tyre age | survived to this tyre age)

as a per-lap binary classifier (`is_pit_lap`, identical target shape to
the Pit Stop Recommendation model), but — critically — trained on a
DELIBERATELY NARROWED feature set that excludes every tactical/strategic
signal (gap to cars ahead/behind, current position, safety car/VSC/yellow
flags, laps remaining, driver_id, team_id). Pit Stop Recommendation wants
all of that, because its job IS to predict the strategic decision. This
model wants none of it, because its job is to isolate the tyre's own
physical hazard curve from the strategic noise sitting on top of it.

From the fitted hazard curve, `predicted_remaining_life_laps` is derived
the standard discrete-survival way: simulate the hazard forward one tyre
age at a time (holding track/environmental covariates fixed at their
current-lap values — this project has no forecast model for how track
temp or grip evolves lap to lap, so holding them constant is the honest
simplifying assumption, not a hidden one), turn per-step hazards into a
survival curve S(k) = product_{i<=k}(1 - h(age+i)), and take the
expectation E[remaining life] = sum_k S(k).

Usage:
    uv run python -m models.tyre_degradation.hazard
"""

from __future__ import annotations

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error, roc_auc_score

from models.common.data import load_race_features
from models.common.features import apply_categorical_dtypes
from models.common.splits import temporal_split
from models.common.targets import add_remaining_tyre_life_target
from models.common.tracking import configure_experiment

EVENT_TARGET = "is_pit_lap"
LIFE_TARGET = "predicted_remaining_life_laps"

# Deliberately physical/environmental only — no driver_id, team_id,
# gap/position/flag columns. See module docstring for why.
HAZARD_CATEGORICAL_FEATURES = ["compound", "circuit_id"]
HAZARD_NUMERIC_FEATURES = [
    "tyre_age",
    "stint_number",
    "track_temp",
    "air_temp",
    "humidity",
    "degradation_rate",
    "grip_estimate",
    "condition_delta",
]
HAZARD_BOOLEAN_FEATURES = ["rainfall_flag"]
HAZARD_FEATURE_COLUMNS = HAZARD_NUMERIC_FEATURES + HAZARD_BOOLEAN_FEATURES + HAZARD_CATEGORICAL_FEATURES

K_MAX_LAPS = 40  # simulate the hazard forward at most this many tyre-age steps


def prepare_dataset() -> pd.DataFrame:
    df = load_race_features()
    df = add_remaining_tyre_life_target(df)
    df = df.dropna(subset=[EVENT_TARGET, LIFE_TARGET, "tyre_age", "compound"])
    df[EVENT_TARGET] = df[EVENT_TARGET].astype(int)
    for col in HAZARD_BOOLEAN_FEATURES:
        df[col] = df[col].astype(int)
    return apply_categorical_dtypes(df, HAZARD_CATEGORICAL_FEATURES)


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def expected_remaining_life(model: lgb.LGBMClassifier, df: pd.DataFrame, k_max: int = K_MAX_LAPS) -> np.ndarray:
    """Turn a fitted hazard classifier into per-row expected remaining life
    via the discrete survival expectation sum_k S(k), S(k) = prod(1-h(age+i)).
    """
    n = len(df)
    long = df.iloc[np.repeat(np.arange(n), k_max)].reset_index(drop=True)
    offsets = np.tile(np.arange(1, k_max + 1), n)
    long["tyre_age"] = df["tyre_age"].to_numpy()[np.repeat(np.arange(n), k_max)] + offsets

    hazard = model.predict_proba(long[HAZARD_FEATURE_COLUMNS])[:, 1].reshape(n, k_max)
    survival = np.cumprod(1.0 - hazard, axis=1)
    return survival.sum(axis=1)


def main() -> None:
    configure_experiment("tyre_degradation")

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} event_rate={train[EVENT_TARGET].mean():.3f}")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,  # smaller than pit_stop's 63: far fewer, purely-physical features
        is_unbalance=True,
        random_state=42,
    )
    model.fit(
        train[HAZARD_FEATURE_COLUMNS],
        train[EVENT_TARGET],
        eval_set=[(val[HAZARD_FEATURE_COLUMNS], val[EVENT_TARGET])],
        eval_metric="average_precision",
        categorical_feature=HAZARD_CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    test = test.copy()
    test_proba = model.predict_proba(test[HAZARD_FEATURE_COLUMNS])[:, 1]
    test["expected_remaining_life"] = expected_remaining_life(model, test)

    metrics = {
        # Diagnostic only — not comparable to pit_stop's precision-at-threshold,
        # since this model is deliberately blind to the tactical signals that
        # make pit timing predictable. A lower score here is expected, not a bug.
        "test_hazard_auc": roc_auc_score(test[EVENT_TARGET], test_proba),
        "test_hazard_pr_auc": average_precision_score(test[EVENT_TARGET], test_proba),
        # These ARE directly comparable to train.py's regression baseline.
        "test_rmse": _rmse(test[LIFE_TARGET], test["expected_remaining_life"]),
        "test_mae": mean_absolute_error(test[LIFE_TARGET], test["expected_remaining_life"]),
    }
    per_compound_mae = test.groupby("compound", observed=True).apply(
        lambda g: mean_absolute_error(g[LIFE_TARGET], g["expected_remaining_life"]), include_groups=False
    )
    for compound, mae in per_compound_mae.items():
        metrics[f"test_mae_{str(compound).lower()}"] = float(mae)

    with mlflow.start_run(run_name="hazard_survival"):
        mlflow.set_tag("algorithm", "lightgbm_hazard_survival")
        mlflow.log_param("k_max_laps", K_MAX_LAPS)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.lightgbm.log_model(model, name="model")

    print(metrics)


if __name__ == "__main__":
    main()
