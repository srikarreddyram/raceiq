"""Does knowing the car actually help? — PRD Section 19's success
criterion "team-specific features provide measurable lift vs.
team-agnostic baseline on held-out season".

That criterion is worth testing precisely, because "team-agnostic" can
mean two different things and they give different answers:

  AGNOSTIC   no team identity at all — team_id and rival_team_id dropped
             along with every CarProfile column. This is the PRD's literal
             baseline.
  TEAM_ID    team identity as a bare categorical, which is what these
             models already ship with today. A tree can memorise
             team-level effects from this alone.
  PROFILE    team_id plus the inferred CarProfile characteristics
             (car_profiles/inference/characteristics.py).

PROFILE vs AGNOSTIC answers the PRD. PROFILE vs TEAM_ID answers the
question that actually decides whether this subsystem ships: does
describing the car beat simply naming it? Building an eight-characteristic
inference pipeline is only justified by the second comparison, so both are
reported.

Every condition trains on identical splits with identical hyperparameters
and a fixed seed; the only thing that varies is the feature list.

Usage:
    uv run python -m car_profiles.evaluate_lift
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.common.features import CATEGORICAL_COLUMNS
from models.common.splits import temporal_split

TEAM_IDENTITY_COLUMNS = ["team_id", "rival_team_id"]

CAR_PROFILE_COLUMNS = [
    "car_tyre_warmup_rate",
    "car_cold_tyre_pace_loss",
    "car_downforce_proxy",
    "car_degradation_vs_field",
    "car_degradation_rate_soft",
    "car_degradation_rate_medium",
    "car_degradation_rate_hard",
    "car_safety_car_restart_pace",
    "car_undercut_vulnerability",
    "car_tyre_temp_sensitivity",
    "car_aero_sensitivity",
]


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _feature_sets(base_features: list[str]) -> dict[str, list[str]]:
    agnostic = [c for c in base_features if c not in TEAM_IDENTITY_COLUMNS]
    return {
        "agnostic": agnostic,
        "team_id": list(base_features),
        "profile": list(base_features) + CAR_PROFILE_COLUMNS,
    }


def _train_once(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    n_estimators: int,
) -> dict[str, float]:
    categoricals = [c for c in CATEGORICAL_COLUMNS if c in features]
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(
        train[features],
        train[target],
        eval_set=[(val[features], val[target])],
        eval_metric="rmse",
        categorical_feature=categoricals,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    pred = model.predict(test[features])
    return {"rmse": _rmse(test[target], pred), "mae": mean_absolute_error(test[target], pred)}


def evaluate_lap_time() -> pd.DataFrame:
    from models.lap_time.train import FEATURE_COLUMNS, TARGET, prepare_dataset, stable_regime_mask

    df = prepare_dataset()
    train, val, test = temporal_split(df)
    stable = stable_regime_mask(test)

    rows = []
    for name, features in _feature_sets(FEATURE_COLUMNS).items():
        categoricals = [c for c in CATEGORICAL_COLUMNS if c in features]
        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        model.fit(
            train[features],
            train[TARGET],
            eval_set=[(val[features], val[TARGET])],
            eval_metric="rmse",
            categorical_feature=categoricals,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict(test[features])
        rows.append(
            {
                "model": "lap_time",
                "condition": name,
                "n_features": len(features),
                "test_rmse": _rmse(test[TARGET], pred),
                "test_mae": mean_absolute_error(test[TARGET], pred),
                # The PRD's 0.4s target is a pace-accuracy claim, and raw
                # RMSE here is dominated by pit/SC/red-flag laps no feature
                # set can foresee — so the stable-regime slice is the
                # comparison that actually speaks to the criterion.
                "stable_rmse": _rmse(test.loc[stable, TARGET], pred[stable.to_numpy()]),
                "stable_mae": mean_absolute_error(test.loc[stable, TARGET], pred[stable.to_numpy()]),
            }
        )
    return pd.DataFrame(rows)


def evaluate_tyre_degradation() -> pd.DataFrame:
    from models.tyre_degradation.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    train, val, test = temporal_split(df)

    rows = []
    for name, features in _feature_sets(FEATURE_COLUMNS).items():
        metrics = _train_once(train, val, test, features, TARGET, n_estimators=1000)
        rows.append(
            {
                "model": "tyre_degradation",
                "condition": name,
                "n_features": len(features),
                "test_rmse": metrics["rmse"],
                "test_mae": metrics["mae"],
                "stable_rmse": np.nan,
                "stable_mae": np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_pit_stop() -> pd.DataFrame:
    """Included because `undercut_vulnerability` is aimed squarely at pit
    timing — if any classifier should benefit from a CarProfile, it's this
    one. Scored on PR-AUC rather than the regression metrics above, since
    pit laps are ~6% of rows and accuracy-style metrics are meaningless at
    that balance.
    """
    from sklearn.metrics import average_precision_score

    from models.pit_stop.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    train, val, test = temporal_split(df)

    rows = []
    for name, features in _feature_sets(FEATURE_COLUMNS).items():
        categoricals = [c for c in CATEGORICAL_COLUMNS if c in features]
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=63,
            is_unbalance=True,
            random_state=42,
            deterministic=True,
            force_row_wise=True,
            verbose=-1,
        )
        model.fit(
            train[features],
            train[TARGET],
            eval_set=[(val[features], val[TARGET])],
            eval_metric="average_precision",
            categorical_feature=categoricals,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        proba = model.predict_proba(test[features])[:, 1]
        rows.append(
            {
                "model": "pit_stop",
                "condition": name,
                "n_features": len(features),
                # Stored under the shared column names so everything prints
                # in one table; for this model "error" is 1 - PR-AUC, so
                # lower is still better and the comparison logic holds.
                "test_rmse": np.nan,
                "test_mae": 1 - average_precision_score(test[TARGET], proba),
                "stable_rmse": np.nan,
                "stable_mae": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _report(results: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    print()

    for model_name, group in results.groupby("model"):
        by_condition = group.set_index("condition")
        metric = "stable_mae" if model_name == "lap_time" else "test_mae"
        if by_condition[metric].isna().all():
            metric = "test_mae"

        agnostic = by_condition.loc["agnostic", metric]
        team_id = by_condition.loc["team_id", metric]
        profile = by_condition.loc["profile", metric]

        def delta(new: float, old: float) -> str:
            change = (new - old) / old * 100
            verdict = "better" if new < old else "WORSE"
            return f"{change:+.2f}% ({verdict})"

        print(f"=== {model_name} ({metric}) ===")
        print(f"  agnostic          : {agnostic:.4f}")
        print(f"  team_id           : {team_id:.4f}   vs agnostic {delta(team_id, agnostic)}")
        print(f"  profile           : {profile:.4f}   vs agnostic {delta(profile, agnostic)}")
        print(f"                      {' ' * 8}   vs team_id  {delta(profile, team_id)}")
        print()


def main() -> None:
    results = pd.concat(
        [evaluate_lap_time(), evaluate_tyre_degradation(), evaluate_pit_stop()], ignore_index=True
    )
    _report(results)


if __name__ == "__main__":
    main()
