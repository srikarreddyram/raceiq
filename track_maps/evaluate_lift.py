"""Does describing the circuit help? — PRD Section 19's criterion "circuit
geometry features provide measurable lift vs. circuit-ID-only baseline".

Two questions, because they have different answers for different reasons:

1. SEEN CIRCUITS — the project's standard temporal split (train 2018-23,
   validate 2024, test 2025). Nearly every 2025 circuit appears in
   training, so `circuit_id` alone lets a tree memorise each circuit's
   effect. Geometry can only win here if it helps trees share what they
   learn between similar circuits.

2. UNSEEN CIRCUITS — circuits split into K folds; each fold's circuits
   are removed from training entirely and scored on their 2025 rows. This
   is the Madring question: a venue with no history, where `circuit_id` is
   an unknown category and carries nothing. Early stopping validates on
   2024 rows of the TRAINING circuits only, so no held-out circuit
   influences the fit.

Which geometry columns, and why the rest are excluded:

  LAYOUT      lap length, corner count, mean corner radius, elevation
              range and variance. Pure X/Y/Z shape — the same whichever
              year's lap drew it.
  + SPEED     adds corner speed classes, braking distance and downforce
              demand. These come from the reference lap's speed and
              throttle traces, and 25 of 32 reference laps are from 2025
              or 2026, so they're reported separately rather than trusted
              blind.
  EXCLUDED    sector average speeds (lap length / sector speed is
              essentially the reference race's lap time — a direct leak
              into a lap-time target), pit_lane_delta and
              tyre_stress_index (averaged over all history, including the
              rows being predicted), DRS length (null by regulation era,
              not a circuit property), track_evolution_rate (null).

Result (one split, LightGBM, fixed seed; error = MAE or 1 - PR-AUC):

                      seen: geometry vs circuit_id   unseen: geometry vs blind
  lap_time (stable)            -3.6%                         -8.0%
  tyre_degradation             -3.6% (blind alone -3.3%)     +3.1%
  pit_stop                     +0.3%                         +2.3%
  safety_car                  -15.0%                        +21.7%

Only Lap Time improves in both settings, so only Lap Time got the columns
(replacing circuit_id — see models/lap_time/train.py). Safety Car is the
cautionary one: a big "lift" on seen circuits that reverses on unseen
ones, meaning the numeric geometry was acting as a circuit fingerprint to
memorise ~30 circuits' safety-car history, not describing anything that
generalises.

Usage:
    uv run python -m track_maps.evaluate_lift
"""

from __future__ import annotations

import importlib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, mean_absolute_error

from models.common.circuit_geometry import CIRCUIT_GEOMETRY_COLUMNS, LAYOUT_COLUMNS, add_circuit_geometry
from models.common.features import CATEGORICAL_COLUMNS
from models.common.splits import TEST_SEASON, TRAIN_SEASONS, VALIDATION_SEASON, temporal_split

GEOMETRY_COLUMNS = CIRCUIT_GEOMETRY_COLUMNS

N_CIRCUIT_FOLDS = 5

# (module, is_classifier). Error is MAE for regressors and 1 - PR-AUC for
# classifiers, so lower is better everywhere.
MODELS = [
    ("lap_time", False),
    ("tyre_degradation", False),
    ("pit_stop", True),
    ("safety_car", True),
]


def _load(module_name: str) -> tuple[pd.DataFrame, list[str], str, pd.Series | None]:
    module = importlib.import_module(f"models.{module_name}.train")
    df = module.prepare_dataset()
    df = add_circuit_geometry(df)
    # The baseline is the model's feature list WITHOUT geometry and WITH
    # circuit_id — the pre-track_maps model, whatever production now uses.
    base = [c for c in module.FEATURE_COLUMNS if c not in GEOMETRY_COLUMNS]
    if "circuit_id" not in base:
        base.append("circuit_id")
    return df, base, module.TARGET, module


def _conditions(base: list[str]) -> dict[str, list[str]]:
    blind = [c for c in base if c != "circuit_id"]
    return {
        "blind": blind,
        "circuit_id": list(base),
        "id+layout": list(base) + LAYOUT_COLUMNS,
        "id+geometry": list(base) + GEOMETRY_COLUMNS,
        "layout": blind + LAYOUT_COLUMNS,
        "geometry": blind + GEOMETRY_COLUMNS,
    }


def _fit_score(train, val, test, features, target, classifier, stable=None) -> float:
    params = dict(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    categoricals = [c for c in CATEGORICAL_COLUMNS if c in features]
    if classifier:
        model = lgb.LGBMClassifier(objective="binary", is_unbalance=True, **params)
        metric = "average_precision"
    else:
        model = lgb.LGBMRegressor(objective="regression", **params)
        metric = "l1"
    model.fit(
        train[features],
        train[target],
        eval_set=[(val[features], val[target])],
        eval_metric=metric,
        categorical_feature=categoricals,
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    if classifier:
        return 1 - average_precision_score(test[target], model.predict_proba(test[features])[:, 1])
    pred = model.predict(test[features])
    if stable is not None:
        mask = stable.to_numpy()
        return mean_absolute_error(test.loc[mask, target], pred[mask])
    return mean_absolute_error(test[target], pred)


def evaluate_model(name: str, classifier: bool) -> list[dict]:
    df, base, target, module = _load(name)
    stable_fn = getattr(module, "stable_regime_mask", None)
    rows = []

    # 1. Seen circuits.
    train, val, test = temporal_split(df)
    for condition, features in _conditions(base).items():
        stable = stable_fn(test) if stable_fn else None
        rows.append(
            {"model": name, "setting": "seen", "condition": condition,
             "error": _fit_score(train, val, test, features, target, classifier, stable)}
        )

    # 2. Unseen circuits. circuit_id conditions are dropped: for a circuit
    # absent from training the ID is an unknown category and adds nothing.
    test_circuits = sorted(test["circuit_id"].astype(str).unique())
    rng = np.random.default_rng(42)
    folds = np.array_split(rng.permutation(test_circuits), N_CIRCUIT_FOLDS)
    unseen = {k: v for k, v in _conditions(base).items() if "circuit_id" not in k and not k.startswith("id+")}
    per_condition: dict[str, list[tuple[float, int]]] = {k: [] for k in unseen}
    for fold in folds:
        held = df["circuit_id"].astype(str).isin(set(fold))
        f_train = df[df["season"].isin(TRAIN_SEASONS) & ~held]
        f_val = df[(df["season"] == VALIDATION_SEASON) & ~held]
        f_test = df[(df["season"] == TEST_SEASON) & held]
        if f_test.empty or (classifier and f_test[target].nunique() < 2):
            continue
        stable = stable_fn(f_test) if stable_fn else None
        for condition, features in unseen.items():
            err = _fit_score(f_train, f_val, f_test, features, target, classifier, stable)
            per_condition[condition].append((err, len(f_test)))
    for condition, results in per_condition.items():
        errors, weights = zip(*results)
        rows.append(
            {"model": name, "setting": "unseen", "condition": condition,
             "error": float(np.average(errors, weights=weights))}
        )
    return rows


def _report(results: pd.DataFrame) -> None:
    for (model, setting), group in results.groupby(["model", "setting"], sort=False):
        by = group.set_index("condition")["error"]
        baseline_name = "circuit_id" if setting == "seen" else "blind"
        baseline = by[baseline_name]
        metric = "1-PR-AUC" if model in ("pit_stop", "safety_car") else ("stable MAE" if model == "lap_time" else "MAE")
        print(f"=== {model} / {setting} circuits ({metric}, baseline = {baseline_name}) ===")
        for condition, err in by.items():
            change = (err - baseline) / baseline * 100
            tag = "" if condition == baseline_name else f"  {change:+6.2f}%"
            print(f"  {condition:12s} {err:.4f}{tag}")
        print()


def main() -> None:
    rows = []
    for name, classifier in MODELS:
        rows += evaluate_model(name, classifier)
        print(f"[done] {name}", flush=True)
    results = pd.DataFrame(rows)
    print()
    _report(results)


if __name__ == "__main__":
    main()
