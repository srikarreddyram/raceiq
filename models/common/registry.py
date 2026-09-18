"""Load a trained model for a given experiment, and promote a specific
run to be the one that gets loaded — the two halves of a real model
registry, not just "grab whatever was trained most recently."

Every `load_latest_*` function here prefers a model version some run has
been explicitly *promoted* to (via `register_and_promote`, tagged with
the `PRODUCTION_ALIAS` alias) over just picking the newest run by
timestamp. That distinction matters once `retraining/run.py` exists: a
retrain that fails validation should never silently become what the
strategy engine serves just because it happened to run most recently.
Until a model's first successful promotion, though, this falls back to
the original "latest run by name" behavior — so nothing here changes
until something actually calls `register_and_promote`.

Each (experiment_name, run_name) pair gets its own registered-model name
(`{experiment_name}__{run_name}`), not one registered model per
experiment: `lap_time_prediction` has two run_names that are both live,
non-comparison models in production (the LightGBM "lightgbm" run
`predict_next_lap_time` uses, and the LSTM "lstm_sequence" run the Monte
Carlo simulation uses via `lstm_oracle.py`) — they need independent
promotion, not one shared alias fighting over which model "lap_time_
prediction" even refers to.

`import lightgbm` below is otherwise unused and looks removable — it
isn't. `mlflow.lightgbm.load_model()` lazily imports the real `lightgbm`
package internally, and doing that cold (nothing else in the process has
imported lightgbm yet) reliably segfaults rather than raising a normal
exception, confirmed directly: every real caller in this project (the
strategy engine, the API, the test suite) already imports lightgbm
transitively through some other module first and never hits this, but
this module can be imported completely standalone (e.g. a one-off
inspection script), so it can't rely on that accident of import order.
Importing it here, once, up front, is cheap insurance.
"""

from __future__ import annotations

import lightgbm  # noqa: F401 -- see module docstring; import order matters here
import torch  # noqa: F401 -- same reasoning, for load_latest_pytorch_model

from functools import lru_cache

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from models.common.tracking import MLFLOW_DB_PATH

PRODUCTION_ALIAS = "production"


def _registered_model_name(experiment_name: str, run_name: str) -> str:
    return f"{experiment_name}__{run_name}"


def _latest_run_id(client: MlflowClient, experiment_name: str, run_name: str) -> str:
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
    return runs[0].info.run_id


def _resolve_model_uri(experiment_name: str, run_name: str) -> str:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()
    registered_name = _registered_model_name(experiment_name, run_name)

    try:
        version = client.get_model_version_by_alias(registered_name, PRODUCTION_ALIAS)
        return f"models:/{registered_name}@{PRODUCTION_ALIAS}", version.run_id
    except MlflowException:
        pass  # never promoted (yet) -- fall back to the newest run by name

    run_id = _latest_run_id(client, experiment_name, run_name)
    return f"runs:/{run_id}/model", run_id


@lru_cache(maxsize=None)
def load_latest_model(experiment_name: str, run_name: str = "lightgbm"):
    # Loaded via the LightGBM flavor rather than generic pyfunc: every
    # tabular model in this project is LightGBM, and the strategy engine
    # needs the native sklearn API (predict_proba for the two classifiers
    # used stochastically in simulation), which pyfunc's generic
    # .predict()-only wrapper doesn't expose.
    model_uri, _run_id = _resolve_model_uri(experiment_name, run_name)
    return mlflow.lightgbm.load_model(model_uri)


@lru_cache(maxsize=None)
def load_latest_pytorch_model(experiment_name: str, run_name: str):
    model_uri, _run_id = _resolve_model_uri(experiment_name, run_name)
    return mlflow.pytorch.load_model(model_uri)


def latest_run_id(experiment_name: str, run_name: str) -> str:
    """The most recent run's id for (experiment_name, run_name), regardless
    of promotion state. Used by retraining/run.py right after a fresh
    training call to find the run it just created, before deciding
    whether to promote it.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    return _latest_run_id(MlflowClient(), experiment_name, run_name)


def register_and_promote(experiment_name: str, run_name: str, run_id: str) -> None:
    """Create (or reuse) a registered model for this (experiment, run_name)
    pair, register `run_id`'s logged model as a new version of it, and
    point the `production` alias at that version — moving the alias off
    whatever version held it before. Called only by `retraining/run.py`,
    and only after that run's model has passed validation; nothing else
    in this project should call this directly.

    Uses the high-level `mlflow.register_model()` rather than
    `MlflowClient.create_model_version(source=f"runs:/{run_id}/model", ...)`
    deliberately: this MLflow version logs a model as a first-class
    "Logged Model" (`mlruns/<exp_id>/models/m-<id>/...`), not nested under
    the run's own artifact directory the classic `runs:/.../model` path
    assumes — the low-level call doesn't know to redirect and silently
    registers a version pointing nowhere. `mlflow.register_model()`
    detects that itself (a real, harmless warning: "Run ... has no
    artifacts at artifact path 'model', registering model based on
    models:/m-... instead") and resolves correctly.
    """
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = MlflowClient()
    registered_name = _registered_model_name(experiment_name, run_name)

    version = mlflow.register_model(f"runs:/{run_id}/model", registered_name)
    client.set_registered_model_alias(registered_name, PRODUCTION_ALIAS, version.version)

    load_latest_model.cache_clear()
    load_latest_pytorch_model.cache_clear()
