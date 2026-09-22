"""Model Performance View backend — monitoring/run.py's drift logic and the
tables it writes (run `uv run python -m monitoring.run` first)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from models.common.db import get_connection
from monitoring.run import SPECS, feature_drift


def _race_level_frame(n_races: int, shift: float, seed: int) -> pd.DataFrame:
    # One value per race repeated over 1,000 laps — how air temperature
    # actually looks in this data.
    rng = np.random.default_rng(seed)
    values = rng.normal(25 + shift, 4, n_races)
    return pd.DataFrame(
        {"race_id": np.repeat([f"r{seed}_{i}" for i in range(n_races)], 1000), "air_temp": np.repeat(values, 1000)}
    )


def test_race_level_feature_without_shift_is_not_flagged():
    # Plain PSI calls this massive drift (14 clusters, not 14,000 rows);
    # judged against its own bootstrap noise it isn't.
    train = _race_level_frame(130, 0.0, seed=1)
    current = _race_level_frame(14, 0.0, seed=2)
    d = feature_drift(train, current, "air_temp")
    assert d["psi"] > 0.25  # the textbook threshold would have fired
    assert not d["drifted"]


def test_real_shift_is_flagged():
    train = _race_level_frame(130, 0.0, seed=1)
    current = _race_level_frame(14, 12.0, seed=3)
    assert feature_drift(train, current, "air_temp")["drifted"]


def test_model_health_written_for_every_model():
    con = get_connection()
    try:
        rows = con.execute("SELECT experiment, status, per_race_json FROM monitoring.model_health").fetchall()
    finally:
        con.close()
    assert {r[0] for r in rows} == {s.experiment for s in SPECS}
    for experiment, status, per_race in rows:
        assert status in {"ok", "warn", "skipped"}
        if status != "skipped":
            assert json.loads(per_race)
