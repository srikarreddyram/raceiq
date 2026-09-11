"""MLflow setup shared by every model's training script (PRD Section 15/17
— "all six models tracked and versioned in MLflow with full
reproducibility").

Tracking store is a local SQLite database at the repo root (gitignored —
MLflow runs are regenerable from the training scripts, not source of
truth). MLflow 3.x deprecated the plain-filesystem tracking backend
("maintenance mode", refuses to open without an explicit opt-out) in
favor of a database backend, so this uses `sqlite:///mlflow.db` rather
than the older `file:./mlruns` scheme. Model/metric artifacts still land
on local disk under `mlartifacts/`. Every model gets its own experiment,
named after the model directory, so runs across models never mix in the
MLflow UI.
"""

from __future__ import annotations

from pathlib import Path

import mlflow

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MLFLOW_DB_PATH = REPO_ROOT / "mlflow.db"
MLARTIFACTS_DIR = REPO_ROOT / "mlartifacts"


def configure_experiment(name: str) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    mlflow.set_experiment(name)
