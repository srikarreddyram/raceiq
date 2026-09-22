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

Beyond the season-wide comparison, each run also records what the Pit
Wall's Model Performance View shows (PRD Section 13.2):

- the headline metric PER RACE, so drift reads as a trend across the
  season rather than one number that can hide a recent collapse;
- input drift per feature, as the Population Stability Index of each
  numeric feature's 2026 distribution against the training seasons'
  (bins from training deciles). A model can hold its accuracy while its
  inputs move — the first warning of a regulation change.

  PSI's usual thresholds (0.1 / 0.25) assume independent rows, and these
  aren't: air temperature is one value per race repeated over a thousand
  laps, so 2026's "16,000 rows" are 14 observations. Judged that way,
  air_temp showed PSI 2.5 — "massive drift" — with 2026 temperatures
  squarely inside the historical range. So each feature is judged against
  its own noise instead: PSI of 2026 is compared with the PSI of many
  random draws of the same number of TRAINING races against the full
  training set (a cluster bootstrap). Drift is flagged only when 2026 sits
  beyond the 95th percentile of those draws;
- the Lap Time model's per-lap 2026 predictions, for the view's
  prediction-vs-actual overlay.

All of it is written to the warehouse's `monitoring` schema, which the API
reads. Evaluating six models means loading six datasets, which is a
job, not a request.

Usage:
    uv run python -m monitoring.run
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import lightgbm  # noqa: F401 -- see models/common/registry.py's docstring; import order matters
import mlflow
import numpy as np
import pandas as pd
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
from models.common.splits import EXCLUDED_SEASONS, TRAIN_SEASONS
from models.common.tracking import MLFLOW_DB_PATH, configure_experiment

# The metrics worth flagging per model, and how much worse than the
# promoted baseline is "worth a second look" — deliberately loose
# thresholds (this is a brand-new, ~12-race sample, noisy by nature), not
# a precise SLA.
_AUC_WARN_DROP = 0.05
_ERROR_METRIC_WARN_RATIO = 1.25  # current MAE/RMSE > 1.25x baseline

PSI_BINS = 10
PSI_NULL_DRAWS = 200
PSI_FLOOR = 0.1  # never flag below this, however quiet a feature's noise is


@dataclass(frozen=True)
class ModelSpec:
    experiment: str
    module: str
    kind: str  # "binary", "multiclass" or "regression"
    # The metric charted race by race — the same name the promoted run
    # logged, so the chart can draw that baseline beside it.
    per_race_metric: str
    label: str


SPECS = [
    ModelSpec("win_probability", "models.win_probability.train", "binary", "auc_roc", "Win Probability"),
    ModelSpec("final_race_position", "models.race_position.train", "multiclass", "mean_abs_position_error", "Final Race Position"),
    ModelSpec("pit_stop_recommendation", "models.pit_stop.train", "binary", "pr_auc", "Pit Stop"),
    ModelSpec("safety_car_probability", "models.safety_car.train", "binary", "auc_roc", "Safety Car"),
    ModelSpec("tyre_degradation", "models.tyre_degradation.train", "regression", "mae", "Tyre Degradation"),
    ModelSpec("lap_time_prediction", "models.lap_time.train", "regression", "mae_stable_regime", "Lap Time"),
]

_LOWER_IS_BETTER_SUBSTRINGS = ("mae", "rmse", "log_loss", "error")  # "error" catches *_mean_abs_position_error


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def is_lower_better(metric_name: str) -> bool:
    return any(s in metric_name for s in _LOWER_IS_BETTER_SUBSTRINGS)


def _binary_metrics(experiment: str, y, proba) -> dict:
    if experiment == "win_probability":
        return {"auc_roc": roc_auc_score(y, proba), "log_loss": log_loss(y, proba, labels=[0, 1])}
    if experiment == "pit_stop_recommendation":
        pred = (proba >= 0.5).astype(int)
        return {
            "precision_at_0.5": precision_score(y, pred, zero_division=0),
            "recall_at_0.5": recall_score(y, pred, zero_division=0),
            "pr_auc": average_precision_score(y, proba),
        }
    return {"auc_roc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba)}


def _per_race_value(spec: ModelSpec, y: pd.Series, score: np.ndarray, stable: np.ndarray | None) -> float | None:
    """The per-race metric, or None where it's undefined for that race —
    an AUC needs both classes, and most races have no safety car at all."""
    if spec.kind == "binary":
        if y.nunique() < 2:
            return None
        return float(roc_auc_score(y, score) if spec.per_race_metric == "auc_roc" else average_precision_score(y, score))
    if stable is not None:
        if not stable.any():
            return None
        return float(mean_absolute_error(y[stable], score[stable]))
    return float(mean_absolute_error(y, score))


def _psi_from(edges: np.ndarray, expected: np.ndarray, values: np.ndarray) -> float:
    actual = np.clip(np.histogram(values, edges)[0] / len(values), 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def feature_drift(train: pd.DataFrame, current: pd.DataFrame, feature: str, seed: int = 0) -> dict | None:
    """PSI of `feature` in 2026, and the 95th percentile of PSI for a random
    set of the same number of training races — see the module docstring."""
    t = train[["race_id", feature]].dropna()
    c = current[feature].dropna().to_numpy(dtype=float)
    if len(t) < 100 or len(c) < 50 or t[feature].nunique() < 2:
        return None
    values = t[feature].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, PSI_BINS + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.clip(np.histogram(values, edges)[0] / len(values), 1e-4, None)

    observed = _psi_from(edges, expected, c)
    k = current["race_id"].nunique()
    by_race = {r: g.to_numpy(dtype=float) for r, g in t.groupby("race_id")[feature]}
    races = np.array(list(by_race))
    rng = np.random.default_rng(seed)
    null = [
        _psi_from(edges, expected, np.concatenate([by_race[r] for r in rng.choice(races, size=k, replace=False)]))
        for _ in range(PSI_NULL_DRAWS)
    ]
    p95 = float(np.quantile(null, 0.95))
    return {
        "feature": feature,
        "psi": round(observed, 4),
        "null_p95": round(p95, 4),
        "drifted": bool(observed > max(p95, PSI_FLOOR)),
    }


def evaluate(spec: ModelSpec) -> dict | None:
    """Score the promoted model on every completed 2026 race. None when
    there are no eligible 2026 rows yet."""
    module = importlib.import_module(spec.module)
    df = module.prepare_dataset()
    current = df[df["season"].isin(EXCLUDED_SEASONS)]
    if current.empty:
        return None
    model = load_latest_model(spec.experiment)
    features, target = module.FEATURE_COLUMNS, module.TARGET
    notes = []

    if spec.kind == "multiclass":
        # 2026 added two constructors, so a finishing position beyond any
        # the model has a class for is a real possibility — counted, not
        # silently scored as wrong.
        seen = current[target].isin(set(model.classes_))
        if (~seen).any():
            notes.append(f"{int((~seen).sum())} rows with a finishing position the model has no class for")
        current = current[seen]
        proba = model.predict_proba(current[features])
        score = model.predict(current[features])
        metrics = {
            "top3_accuracy": top_k_accuracy_score(current[target], proba, k=3, labels=model.classes_),
            "mean_abs_position_error": mean_absolute_error(current[target], score),
        }
    elif spec.kind == "binary":
        if current[target].nunique() < 2:
            return None
        score = model.predict_proba(current[features])[:, 1]
        metrics = _binary_metrics(spec.experiment, current[target], score)
    else:
        score = model.predict(current[features])
        metrics = {"mae": mean_absolute_error(current[target], score), "rmse": _rmse(current[target], score)}

    stable_fn = getattr(module, "stable_regime_mask", None)
    stable = stable_fn(current).to_numpy() if stable_fn else None
    if stable is not None:
        metrics["mae_stable_regime"] = mean_absolute_error(current[target][stable], score[stable])

    per_race = []
    race_ids = current["race_id"].to_numpy()
    for race_id in sorted(set(race_ids), key=lambda r: int(r.split("_")[1])):
        mask = race_ids == race_id
        per_race.append(
            {
                "race_id": race_id,
                "n_rows": int(mask.sum()),
                "value": _per_race_value(
                    spec, current[target][mask], score[mask], None if stable is None else stable[mask]
                ),
            }
        )

    train = df[df["season"].isin(TRAIN_SEASONS)]
    drift = []
    for feature in module.NUMERIC_FEATURES:
        if feature in current.columns:
            d = feature_drift(train, current, feature)
            if d is not None:
                drift.append(d)
    # Most anomalous first: how far past its own noise, not raw PSI.
    drift.sort(key=lambda d: -(d["psi"] / max(d["null_p95"], 1e-6)))

    result = {
        "n_rows": int(len(current)),
        "n_races": int(current["race_id"].nunique()),
        "metrics": {f"current_test_{k}": float(v) for k, v in metrics.items()},
        "per_race": per_race,
        "feature_drift": drift,
        "notes": notes,
    }
    if spec.experiment == "lap_time_prediction":
        result["predictions"] = pd.DataFrame(
            {
                "race_id": current["race_id"].to_numpy(),
                "driver_id": current["driver_id"].astype(str).to_numpy(),
                "lap_number": current["lap_number"].astype(int).to_numpy(),
                "predicted": score,
                "actual": current[target].to_numpy(),
                "stable": stable,
            }
        )
    return result


def _compare(metric_name: str, current: float, baseline: float) -> str:
    if is_lower_better(metric_name):
        warn = baseline > 0 and current > baseline * _ERROR_METRIC_WARN_RATIO
    else:
        warn = current < baseline - _AUC_WARN_DROP
    return "WARN" if warn else "ok"


def run() -> dict:
    report = {}
    for spec in SPECS:
        evaluated = evaluate(spec)
        if evaluated is None:
            report[spec.experiment] = {"status": "skipped", "reason": "no eligible 2026 rows yet", "spec": spec}
            continue

        baseline_metrics = promoted_run_metrics(spec.experiment, "lightgbm")
        comparisons = {}
        for key, current_value in evaluated["metrics"].items():
            baseline_value = baseline_metrics.get(key.removeprefix("current_"))
            comparisons[key] = {
                "current": current_value,
                "baseline": baseline_value,
                "status": "no baseline" if baseline_value is None else _compare(key, current_value, baseline_value),
            }
        report[spec.experiment] = {
            **evaluated,
            "metrics": comparisons,
            "spec": spec,
            "per_race_baseline": baseline_metrics.get(f"test_{spec.per_race_metric}"),
        }
    return report


def write_to_warehouse(report: dict) -> None:
    """monitoring.model_health (one row per model) and
    monitoring.lap_time_predictions, replaced on every run."""
    from pipelines.gold.db import get_connection

    evaluated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = []
    for experiment, result in report.items():
        spec: ModelSpec = result["spec"]
        skipped = result.get("status") == "skipped"
        rows.append(
            {
                "experiment": experiment,
                "label": spec.label,
                "kind": spec.kind,
                "evaluated_at": evaluated_at,
                "status": "skipped" if skipped else ("warn" if any(m["status"] == "WARN" for m in result["metrics"].values()) else "ok"),
                "n_rows": None if skipped else result["n_rows"],
                "n_races": None if skipped else result["n_races"],
                "per_race_metric": spec.per_race_metric,
                "per_race_lower_is_better": is_lower_better(spec.per_race_metric),
                "per_race_baseline": None if skipped else result["per_race_baseline"],
                "metrics_json": json.dumps({} if skipped else result["metrics"]),
                "per_race_json": json.dumps([] if skipped else result["per_race"]),
                "drift_json": json.dumps([] if skipped else result["feature_drift"]),
                "notes_json": json.dumps([result["reason"]] if skipped else result["notes"]),
            }
        )
    health = pd.DataFrame(rows)
    predictions = report.get("lap_time_prediction", {}).get("predictions")

    con = get_connection()
    try:
        con.execute("CREATE SCHEMA IF NOT EXISTS monitoring")
        con.register("health_df", health)
        con.execute("CREATE OR REPLACE TABLE monitoring.model_health AS SELECT * FROM health_df")
        if predictions is not None:
            con.register("pred_df", predictions)
            con.execute("CREATE OR REPLACE TABLE monitoring.lap_time_predictions AS SELECT * FROM pred_df")
    finally:
        con.close()


def main() -> None:
    report = run()
    write_to_warehouse(report)

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    configure_experiment("monitoring")
    with mlflow.start_run():
        any_warning = False
        for experiment_name, result in report.items():
            print(f"=== {experiment_name} ===")
            if result.get("status") == "skipped":
                print(f"  skipped: {result['reason']}")
                continue
            print(f"  n_rows={result['n_rows']} races={result['n_races']}")
            for metric_key, comparison in result["metrics"].items():
                status = comparison["status"]
                if status == "WARN":
                    any_warning = True
                print(f"  {metric_key}: current={comparison['current']:.4f} baseline={comparison['baseline']} [{status}]")
                if comparison["baseline"] is not None:
                    mlflow.log_metric(f"{experiment_name}.{metric_key}", comparison["current"])
            shifted = [d for d in result["feature_drift"] if d["drifted"]]
            if shifted:
                print(
                    "  input drift beyond noise: "
                    + ", ".join(f"{d['feature']} PSI {d['psi']:.2f} (noise p95 {d['null_p95']:.2f})" for d in shifted)
                )
            for note in result["notes"]:
                print(f"  note: {note}")

        mlflow.set_tag("any_warning", str(any_warning))

    if any_warning:
        print("\nAt least one model shows meaningful drift against its promoted baseline — worth a look.")
    else:
        print("\nNo meaningful drift detected against any promoted baseline.")


if __name__ == "__main__":
    main()
