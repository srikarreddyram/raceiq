"""Categorical-column handling shared across models.

LightGBM's native categorical support needs pandas `category` dtype
columns, and — since a model is trained once on the full train split but
then scored on validation/test splits that might contain categories not
seen in training (a new driver code, a new circuit) — the category set
must be fixed from the *full* dataset before splitting, not fit
separately per split. Otherwise the same driver_id could get a different
internal code in train vs. test, or LightGBM would treat an unseen
category as missing in a way that's inconsistent run to run.
"""

from __future__ import annotations

import pandas as pd

CATEGORICAL_COLUMNS = [
    "driver_id",
    "team_id",
    "circuit_id",
    "compound",
    "rival_driver_id",
    "rival_team_id",
    "rival_compound",
]


def apply_categorical_dtypes(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    for col in columns or CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df
