"""Load the most recently trained model for a given experiment.

The strategy engine needs to call all four "ingredient" models (Lap Time,
Tyre Degradation, Safety Car Probability, and — for scoring context — Win
Probability) without retraining them itself. Rather than adding a
separate fixed-path save alongside each training script's existing MLflow
logging, this just queries MLflow for each experiment's most recent run
by a given name and loads that run's logged model — one mechanism, no
duplicated artifact storage.
"""

from __future__ import annotations

from functools import lru_cache

import mlflow
from mlflow.tracking import MlflowClient

from models.common.tracking import MLFLOW_DB_PATH


@lru_cache(maxsize=None)
def load_latest_model(experiment_name: str, run_name: str = "lightgbm"):
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"No MLflow experiment named {experiment_name!r} — run its train.py first")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs named {run_name!r} in experiment {experiment_name!r}")

    # Loaded via the LightGBM flavor rather than generic pyfunc: every model
    # in this project is LightGBM, and the strategy engine needs the native
    # sklearn API (predict_proba for the two classifiers used stochastically
    # in simulation), which pyfunc's generic .predict()-only wrapper doesn't
    # expose.
    return mlflow.lightgbm.load_model(f"runs:/{runs[0].info.run_id}/model")
