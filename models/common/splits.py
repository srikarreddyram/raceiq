"""Temporal train/validation/test split — PRD Section 11: "training on
seasons 1 through N, validation on season N+1... test on season N+2. No
random splits are used anywhere in the pipeline."

Every model in this project uses the exact same three season ranges, so
the split is defined once here rather than re-decided per model. With
2018-2026 currently backfilled:

- train: 2018-2023 (6 seasons)
- validation: 2024
- test: 2025

2026 is excluded entirely — per the project's data-lookback decision, an
in-progress season isn't a fair held-out test set (its later races haven't
happened yet, and results/positions for future rounds don't exist). It'll
become the natural next test season once it's complete.
"""

from __future__ import annotations

import pandas as pd

TRAIN_SEASONS = list(range(2018, 2024))
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
EXCLUDED_SEASONS = {2026}


def temporal_split(
    df: pd.DataFrame, season_col: str = "season"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into (train, validation, test) by season — never by row order."""
    train = df[df[season_col].isin(TRAIN_SEASONS)]
    validation = df[df[season_col] == VALIDATION_SEASON]
    test = df[df[season_col] == TEST_SEASON]
    return train, validation, test
