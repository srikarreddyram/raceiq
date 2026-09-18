"""Model drift monitoring against the real, ongoing 2026 season — PRD
Section 15/17's "monitoring", scoped to what's actually real here.

This project has no deployed API taking live user traffic yet, so there's
no request stream to monitor latency or error rates against — that would
be inert instrumentation with nothing to instrument. What DOES exist, and
grows every week, is the real 2026 F1 season: `EXCLUDED_SEASONS`
(models/common/splits.py) keeps it out of train/val/test specifically
*because* it isn't finished, but every round that DOES finish is genuine,
never-trained-on-or-tested-on ground truth for whatever model is
currently promoted. This script re-evaluates every promoted model
against however many 2026 races have actually completed, and compares
that to the metric MLflow recorded when that model was promoted — the
same comparison a live dashboard makes against incoming traffic, just
against a real held-out season instead of live requests.

Each model's own `prepare_dataset()` (from its train.py) is reused
as-is and just filtered down to season 2026 afterward, rather than
reimplementing feature/target construction here — the whole point is
comparing against exactly the same preprocessing a retrain would use, not
a second, drifting implementation of it.

The LSTM lap-time model is deliberately not covered: its evaluation needs
the full padded-sequence/collate machinery from
models/lap_time_sequence/data.py, not a single dataframe scored row by
row, and duplicating that here just for monitoring wasn't judged worth it
yet — noted rather than silently skipped.

This is observational only, unlike retraining/run.py's pass/fail gate:
a drift warning here doesn't block anything automatically. Deciding "does
this warrant a retrain" is left to whoever reads the report, at least
until there's a real reason to automate that decision too.

Usage:
    uv run python -m monitoring.run
"""

from __future__ import annotations

import lightgbm  # noqa: F401 -- see models/common/registry.py's docstring; import order matters
import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    top_k_accuracy_score,
)

from models.common.registry import load_latest_model, promoted_run_metrics
from models.common.splits import EXCLUDED_SEASONS
from models.common.tracking import MLFLOW_DB_PATH, configure_experiment

# The metrics worth flagging per model, and how much worse than the
# promoted baseline is "worth a second look" — deliberately loose
# thresholds (this is a brand-new, ~12-race sample, noisy by nature), not
# a precise SLA.
_AUC_WARN_DROP = 0.05
_ERROR_METRIC_WARN_RATIO = 1.25  # current MAE/RMSE > 1.25x baseline


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _check_win_probability() -> dict:
    from models.win_probability.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty or df2026[TARGET].nunique() < 2:
        return {}

    model = load_latest_model("win_probability")
    proba = model.predict_proba(df2026[FEATURE_COLUMNS])[:, 1]
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "current_test_auc_roc": roc_auc_score(df2026[TARGET], proba),
        "current_test_log_loss": log_loss(df2026[TARGET], proba, labels=[0, 1]),
    }


def _check_race_position() -> dict:
    from models.race_position.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty:
        return {}

    model = load_latest_model("final_race_position")
    known_classes = set(model.classes_)
    # A real, meaningful thing to catch here, not just an edge case to
    # suppress: 2026 added two new constructors (see
    # pipelines/gold/circuit_history.py's docstring on the same season's
    # new driver/team debutants), which could plausibly mean more cars on
    # the grid than any prior season ever had — a finishing position the
    # model has literally never seen a class for.
    seen_mask = df2026[TARGET].isin(known_classes)
    n_unseen = int((~seen_mask).sum())
    df_eval = df2026[seen_mask]
    if df_eval.empty:
        return {"n_rows": len(df2026), "n_races": df2026["race_id"].nunique(), "n_unseen_position_class": n_unseen}

    proba = model.predict_proba(df_eval[FEATURE_COLUMNS])
    pred = model.predict(df_eval[FEATURE_COLUMNS])
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "n_unseen_position_class": n_unseen,
        "current_test_top3_accuracy": top_k_accuracy_score(df_eval[TARGET], proba, k=3, labels=model.classes_),
        "current_test_mean_abs_position_error": mean_absolute_error(df_eval[TARGET], pred),
    }


def _check_pit_stop() -> dict:
    from models.pit_stop.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty or df2026[TARGET].nunique() < 2:
        return {}

    model = load_latest_model("pit_stop_recommendation")
    proba = model.predict_proba(df2026[FEATURE_COLUMNS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "current_test_precision_at_0.5": precision_score(df2026[TARGET], pred, zero_division=0),
        "current_test_recall_at_0.5": recall_score(df2026[TARGET], pred, zero_division=0),
        "current_test_pr_auc": average_precision_score(df2026[TARGET], proba),
    }


def _check_safety_car() -> dict:
    from models.safety_car.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()  # already race-lap grain, not per-driver
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty or df2026[TARGET].nunique() < 2:
        return {}

    model = load_latest_model("safety_car_probability")
    proba = model.predict_proba(df2026[FEATURE_COLUMNS])[:, 1]
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "current_test_auc_roc": roc_auc_score(df2026[TARGET], proba),
        "current_test_pr_auc": average_precision_score(df2026[TARGET], proba),
    }


def _check_tyre_degradation() -> dict:
    from models.tyre_degradation.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty:
        return {}

    model = load_latest_model("tyre_degradation")
    pred = model.predict(df2026[FEATURE_COLUMNS])
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "current_test_mae": mean_absolute_error(df2026[TARGET], pred),
        "current_test_rmse": _rmse(df2026[TARGET], pred),
    }


def _check_lap_time() -> dict:
    from models.lap_time.train import FEATURE_COLUMNS, TARGET, prepare_dataset

    df = prepare_dataset()
    df2026 = df[df["season"].isin(EXCLUDED_SEASONS)]
    if df2026.empty:
        return {}

    model = load_latest_model("lap_time_prediction")
    pred = model.predict(df2026[FEATURE_COLUMNS])
    return {
        "n_rows": len(df2026),
        "n_races": df2026["race_id"].nunique(),
        "current_test_mae": mean_absolute_error(df2026[TARGET], pred),
        "current_test_rmse": _rmse(df2026[TARGET], pred),
    }


CHECKS = [
    ("win_probability", "lightgbm", _check_win_probability),
    ("final_race_position", "lightgbm", _check_race_position),
    ("pit_stop_recommendation", "lightgbm", _check_pit_stop),
    ("safety_car_probability", "lightgbm", _check_safety_car),
    ("tyre_degradation", "lightgbm", _check_tyre_degradation),
    ("lap_time_prediction", "lightgbm", _check_lap_time),
]


_LOWER_IS_BETTER_SUBSTRINGS = ("mae", "rmse", "log_loss", "error")  # "error" catches *_mean_abs_position_error


def _compare(metric_name: str, current: float, baseline: float) -> str:
    is_error_metric = any(s in metric_name for s in _LOWER_IS_BETTER_SUBSTRINGS)
    if is_error_metric:
        warn = baseline > 0 and current > baseline * _ERROR_METRIC_WARN_RATIO
    else:
        warn = current < baseline - _AUC_WARN_DROP
    return "WARN" if warn else "ok"


def run() -> dict:
    report = {}
    for experiment_name, run_name, check_fn in CHECKS:
        current_metrics = check_fn()
        if not current_metrics:
            report[experiment_name] = {"status": "skipped", "reason": "no eligible 2026 rows yet"}
            continue

        baseline_metrics = promoted_run_metrics(experiment_name, run_name)
        comparisons = {}
        for key, current_value in current_metrics.items():
            if not key.startswith("current_"):
                continue
            baseline_key = key.removeprefix("current_")
            baseline_value = baseline_metrics.get(baseline_key)
            if baseline_value is None:
                comparisons[key] = {"current": current_value, "baseline": None, "status": "no baseline"}
                continue
            comparisons[key] = {
                "current": current_value,
                "baseline": baseline_value,
                "status": _compare(key, current_value, baseline_value),
            }
        report[experiment_name] = {"n_rows": current_metrics.get("n_rows"), "metrics": comparisons}

    return report


def main() -> None:
    report = run()

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    configure_experiment("monitoring")
    with mlflow.start_run():
        any_warning = False
        for experiment_name, result in report.items():
            print(f"=== {experiment_name} ===")
            if result.get("status") == "skipped":
                print(f"  skipped: {result['reason']}")
                continue
            print(f"  n_rows={result['n_rows']}")
            for metric_key, comparison in result["metrics"].items():
                status = comparison["status"]
                if status == "WARN":
                    any_warning = True
                print(f"  {metric_key}: current={comparison['current']:.4f} baseline={comparison['baseline']} [{status}]")
                if comparison["baseline"] is not None:
                    mlflow.log_metric(f"{experiment_name}.{metric_key}", comparison["current"])

        mlflow.set_tag("any_warning", str(any_warning))

    if any_warning:
        print("\nAt least one model shows meaningful drift against its promoted baseline — worth a look.")
    else:
        print("\nNo meaningful drift detected against any promoted baseline.")


if __name__ == "__main__":
    main()
