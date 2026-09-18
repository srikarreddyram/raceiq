"""Retrain -> validate -> promote — the decision logic PRD Section 15's
nightly retraining pipeline needs, without the Airflow orchestration it
also names. Flagged the same way this project has flagged its other
tooling deviations (raw SQL instead of dbt, no live telemetry ingestion):
there's no scheduler wired up to run this nightly on its own. What's here
is the part that actually matters — retrain every model against
whatever's currently in the Gold layer, validate the result against the
same real-scenario test suite this session built by hand into
`tests/`, and only let a model that passes ever become what the strategy
engine and serving API actually load. Wiring a cron job or Airflow DAG to
invoke `uv run python -m retraining.run` on a schedule is a separate,
much smaller task than the logic itself — and pointless to build before
this exists.

Every model gets retrained even though not all of them are currently
consumed anywhere at runtime (`pit_stop_recommendation` has no caller
today — see models/pit_stop/train.py's docstring on the PRD's own six
named models) — consistent, uniform behavior beats special-casing which
models "count".

If validation fails, NOTHING gets promoted — not even the models whose
own retrain looked fine. A partial promotion (some models updated, others
silently left on old versions) is a worse failure mode than a fully
stale-but-consistent deployment, since it's invisible from the outside
and no single model's own metrics would reveal it.

Usage:
    uv run python -m retraining.run
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from models.common.registry import latest_run_id, register_and_promote
from models.lap_time_sequence.train import EXPERIMENT_NAME as LSTM_EXPERIMENT
from models.lap_time_sequence.train import RUN_NAME as LSTM_RUN_NAME
from pipelines.gold.run import run as rebuild_gold


@dataclass(frozen=True)
class ModelTarget:
    experiment_name: str
    run_name: str
    train_module: str  # imported and its main() called, not run as a subprocess


# The seven (experiment, run_name) pairs retraining actually produces.
# lap_time's train_module trains three algorithms in one call (lightgbm,
# xgboost, catboost) under the same "lap_time_prediction" experiment —
# only "lightgbm" is ever loaded anywhere, so only it gets a ModelTarget;
# xgboost/catboost stay comparison-only runs, same as today.
MODEL_TARGETS = [
    ModelTarget("lap_time_prediction", "lightgbm", "models.lap_time.train"),
    ModelTarget("tyre_degradation", "lightgbm", "models.tyre_degradation.train"),
    ModelTarget("pit_stop_recommendation", "lightgbm", "models.pit_stop.train"),
    ModelTarget("safety_car_probability", "lightgbm", "models.safety_car.train"),
    ModelTarget("win_probability", "lightgbm", "models.win_probability.train"),
    ModelTarget("final_race_position", "lightgbm", "models.race_position.train"),
    ModelTarget(LSTM_EXPERIMENT, LSTM_RUN_NAME, "models.lap_time_sequence.train"),
]


def _retrain_all() -> list[tuple[ModelTarget, str]]:
    """Runs every target's train.py as its own subprocess (`uv run python
    -m <module>`), not an in-process `importlib.import_module(...).main()`
    call, and returns the (target, run_id) pairs for the runs just
    created. This isn't just tidiness: training six LightGBM models back
    to back in one process is fine, but this project's models span four
    native ML libraries (LightGBM, XGBoost, CatBoost, PyTorch), and
    running all of them in one long-lived process reliably segfaults
    partway through — confirmed directly: rebuilding Gold plus all six
    LightGBM models in a single process completed cleanly, but the full
    seven-target sequence (which also loads PyTorch for the LSTM) crashed
    with SIGSEGV before ever reaching validation, consistent with a
    known class of OpenMP-runtime conflict between ML libraries that each
    bundle their own. Subprocess isolation sidesteps the exact mechanism
    entirely, and is arguably the right design regardless: one model's
    training crash can't corrupt another's already-loaded native state,
    and this is exactly how every one of these scripts has already been
    run and verified all session (`uv run python -m models.<name>.train`).

    Any non-zero exit aborts the whole retraining run before validation
    or promotion ever happens — a partially-retrained set of models is
    not something this function tries to recover from or promote around.
    """
    produced = []
    for target in MODEL_TARGETS:
        print(f"--- retraining {target.experiment_name} ({target.run_name}) via {target.train_module} ---")
        result = subprocess.run([sys.executable, "-m", target.train_module])
        if result.returncode != 0:
            raise RuntimeError(
                f"{target.train_module} exited with code {result.returncode} — aborting retraining run"
            )
        run_id = latest_run_id(target.experiment_name, target.run_name)
        produced.append((target, run_id))
    return produced


def _validate() -> bool:
    """Runs this project's own pytest suite against the just-retrained
    models and freshly-rebuilt Gold layer. Deliberately the SAME suite a
    human would run by hand (tests/), not a separate "retraining-only"
    check — a model that can't pass the real scenario/oracle sanity
    checks has no business being promoted, and a second, looser bar here
    would just be a way to quietly let a worse model through.
    """
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"])
    return result.returncode == 0


def run() -> bool:
    print("=== Rebuilding Gold layer ===")
    rebuild_gold()

    print("=== Retraining all models ===")
    produced = _retrain_all()

    print("=== Validating against tests/ ===")
    if not _validate():
        print(
            "VALIDATION FAILED — promoting nothing. Every model's newest run is still logged in "
            "MLflow for inspection, but load_latest_model()/load_latest_pytorch_model() will keep "
            "serving whatever was last successfully promoted (or, for a model never yet promoted, "
            "the same 'latest run' behavior as before this pipeline existed)."
        )
        return False

    print("=== Validation passed — promoting all newly-trained models ===")
    for target, run_id in produced:
        register_and_promote(target.experiment_name, target.run_name, run_id)
        print(f"promoted {target.experiment_name} ({target.run_name}) -> run {run_id}")

    return True


def main() -> None:
    success = run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
