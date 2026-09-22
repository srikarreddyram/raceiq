"""The promoted Tyre Degradation model's predictions over one real race —
predicted remaining tyre life lap by lap, beside what actually happened.
Feeds the Tyre View's "predicted remaining life" (PRD Section 13.2).

"Actual" is the target as training defines it (models/common/targets.py):
laps until that stint really ended. For a race's final stint that's laps
to the flag, not laps the tyre could still have run.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from models.common.registry import load_latest_model
from models.tyre_degradation.train import FEATURE_COLUMNS, TARGET, prepare_dataset


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    return prepare_dataset()


def remaining_life_trace(team_id: str, season: int) -> dict | None:
    df = _dataset()
    team = df[(df["team_id"].astype(str) == team_id) & (df["season"] == season)]
    if team.empty:
        return None
    race_id = team.sort_values("round")["race_id"].iloc[-1]
    rows = team[team["race_id"] == race_id].sort_values(["driver_id", "lap_number"])
    predicted = load_latest_model("tyre_degradation").predict(rows[FEATURE_COLUMNS])
    return {
        "race_id": race_id,
        "laps": [
            {
                "driver_id": str(r.driver_id),
                "lap_number": int(r.lap_number),
                "stint_number": int(r.stint_number),
                "compound": str(r.compound),
                "tyre_age": int(r.tyre_age),
                "predicted_remaining": round(float(p), 1),
                "actual_remaining": int(getattr(r, TARGET)),
            }
            for r, p in zip(rows.itertuples(), predicted)
        ],
    }
