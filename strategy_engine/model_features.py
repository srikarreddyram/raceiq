"""Bridges the strategy engine's synthetic simulated rows to the exact
categorical encoding each trained model expects.

This matters more than it looks: LightGBM's categorical-feature support
works by converting each category to an integer code, and the split
thresholds baked into a trained model's trees reference those specific
codes — not the category labels. `models/common/features.py`'s
`apply_categorical_dtypes` derives categories from whatever's in the
DataFrame it's given, which is correct during training (call it once on
the full dataset before splitting) but WRONG here: a synthetic one-row
prediction frame only "sees" that row's own value, so pandas would assign
it category code 0 regardless of what code that same value had during
training against the full ~44-driver, ~30-circuit vocabulary. Silently
wrong predictions, no error raised.

The fix is to fix the category vocabulary once (queried directly from
Gold rather than re-deriving it per prediction) and reuse that exact
`CategoricalDtype` for every synthetic row the strategy engine builds.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from models.common.db import get_connection
from models.common.features import CATEGORICAL_COLUMNS


@lru_cache(maxsize=None)
def _reference_dtype(column: str) -> pd.CategoricalDtype:
    con = get_connection()
    try:
        table = "gold.race_features" if column != "circuit_id" else "silver.races"
        values = con.execute(f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL").df()[
            column
        ]
    finally:
        con.close()
    return pd.CategoricalDtype(categories=sorted(values))


def apply_reference_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(_reference_dtype(col))
    return df
