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

That vocabulary must also exclude EXCLUDED_SEASONS, matching
`models/common/features.py`'s own restriction exactly — otherwise this
function silently drifts out of sync with whatever a loaded model was
actually trained on. Real, concrete case this project hit: the current
season (2026) keeps gaining races via ongoing backfill, each one
potentially introducing a category no earlier season had (this project's
data gained `arvid_lindblad`, `cadillac`, `audi` — real 2026 debutants).
Since this function re-queries Gold live on every call, an un-restricted
version would rebuild a *different* (larger) vocabulary than whatever the
model in hand was trained with weeks or days earlier, shifting every
alphabetically-later category's integer code and silently corrupting
predictions for drivers/teams/circuits that never actually changed.
Restricting both sides to the same eligible-seasons rule keeps them in
sync as long as the eligible (2018-2025) data itself is unchanged, which
is the actually-stable invariant here — not "whatever is in Gold right
now."
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from models.common.db import get_connection
from models.common.features import CATEGORICAL_COLUMNS
from models.common.splits import EXCLUDED_SEASONS


@lru_cache(maxsize=None)
def _reference_dtype(column: str) -> pd.CategoricalDtype:
    con = get_connection()
    try:
        excluded = ", ".join(str(s) for s in sorted(EXCLUDED_SEASONS))
        if column == "circuit_id":
            # silver.races has a real `season` column.
            query = (
                f"SELECT DISTINCT {column} FROM silver.races "
                f"WHERE {column} IS NOT NULL AND season NOT IN ({excluded})"
            )
        else:
            # gold.race_features only has `race_id` — season is minted as
            # its `{season}_{round}` prefix (see pipelines/silver/races.py),
            # so it's parsed back out here rather than joining silver.races
            # just for this one column.
            query = (
                f"SELECT DISTINCT {column} FROM gold.race_features "
                f"WHERE {column} IS NOT NULL "
                f"AND CAST(SPLIT_PART(race_id, '_', 1) AS INTEGER) NOT IN ({excluded})"
            )
        values = con.execute(query).df()[column]
    finally:
        con.close()
    return pd.CategoricalDtype(categories=sorted(values))


def apply_reference_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(_reference_dtype(col))
    return df
