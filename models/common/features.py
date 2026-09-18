"""Categorical-column handling shared across models.

LightGBM's native categorical support needs pandas `category` dtype
columns, and — since a model is trained once on the full train split but
then scored on validation/test splits that might contain categories not
seen in training (a new driver code, a new circuit) — the category set
must be fixed from the *full* dataset before splitting, not fit
separately per split. Otherwise the same driver_id could get a different
internal code in train vs. test, or LightGBM would treat an unseen
category as missing in a way that's inconsistent run to run.

Critically, "the full dataset" here means the full *eligible* (non-
EXCLUDED_SEASONS) dataset, not literally every row `load_race_features()`
returns. This project's raw data for the current in-progress season
(2026) keeps growing as more races get backfilled, and that season is
never used in train/val/test (see splits.py) — but a real, previously-
silent bug: deriving the category list from the full un-filtered
DataFrame meant a brand-new 2026 rookie driver or team (this project hit
`arvid_lindblad`, `cadillac`, `audi` — all real 2026 debutants absent
from every earlier season) would insert a new entry into the sorted
category list, shifting the integer code of every alphabetically-later
category by one. A tree's split thresholds reference those integer codes,
not the labels, so a code shift silently reassigns what driver/team/
circuit a previously-trained model's splits actually mean — for existing
categories that never changed at all. Restricting the vocabulary to
eligible seasons means it only changes when the eligible data itself
changes (a real fix, requiring a retrain anyway), not every time the
current season gets another race backfilled.
"""

from __future__ import annotations

import pandas as pd

from models.common.splits import EXCLUDED_SEASONS

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
    eligible = df[~df["season"].isin(EXCLUDED_SEASONS)] if "season" in df.columns else df
    for col in columns or CATEGORICAL_COLUMNS:
        if col in df.columns:
            categories = sorted(eligible[col].dropna().unique())
            df[col] = df[col].astype(pd.CategoricalDtype(categories=categories))
    return df
