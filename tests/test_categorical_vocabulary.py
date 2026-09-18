"""Regression test for a real bug found and fixed this session: the
categorical vocabulary used to train each model and the vocabulary
`strategy_engine/model_features.py` uses at inference time silently drifted
apart as the in-progress season (2026) kept gaining backfilled races,
introducing new categories (a real example hit: driver `arvid_lindblad`,
teams `cadillac`/`audi`) that shifted every alphabetically-later
category's integer code. LightGBM's tree splits reference those integer
codes, not the labels, so a drifted vocabulary silently corrupts
predictions with no error raised.

The fix (models/common/features.py, strategy_engine/model_features.py)
makes both sides derive their vocabulary from the same rule: every season
except EXCLUDED_SEASONS. These tests assert that invariant holds, rather
than asserting anything about a specific driver/team name — a new season
being excluded is the durable property; which names happen to be 2026-only
will change every time more of that season gets backfilled.
"""

from __future__ import annotations

import pandas as pd
import pytest

from models.common.data import load_race_features
from models.common.features import CATEGORICAL_COLUMNS, apply_categorical_dtypes
from models.common.splits import EXCLUDED_SEASONS
from strategy_engine.model_features import _reference_dtype


@pytest.fixture(scope="module")
def full_df():
    return load_race_features()


def test_excluded_seasons_actually_present(full_df):
    """Sanity-check the fixture this whole test file leans on: if the
    excluded season(s) have no rows at all, the other assertions here
    would pass vacuously without actually exercising the fix.
    """
    assert full_df["season"].isin(EXCLUDED_SEASONS).any(), (
        f"No rows found for EXCLUDED_SEASONS={EXCLUDED_SEASONS} — this test can't verify the "
        "vocabulary-exclusion fix without at least one excluded-season row in the local warehouse."
    )


@pytest.mark.parametrize("column", CATEGORICAL_COLUMNS)
def test_training_vocabulary_excludes_excluded_seasons(full_df, column):
    if column not in full_df.columns:
        pytest.skip(f"{column} not present in gold.race_features (e.g. no rival at some rows)")

    eligible_values = set(full_df.loc[~full_df["season"].isin(EXCLUDED_SEASONS), column].dropna().unique())
    excluded_only_values = set(full_df.loc[full_df["season"].isin(EXCLUDED_SEASONS), column].dropna().unique())
    category_introduced_only_by_excluded_season = excluded_only_values - eligible_values

    encoded = apply_categorical_dtypes(full_df, [column])
    trained_categories = set(encoded[column].cat.categories)

    assert not (trained_categories & category_introduced_only_by_excluded_season), (
        f"apply_categorical_dtypes leaked a category for {column!r} that only exists in an "
        f"excluded season: {trained_categories & category_introduced_only_by_excluded_season}"
    )


@pytest.mark.parametrize("column", ["driver_id", "team_id", "compound", "circuit_id"])
def test_inference_vocabulary_matches_training_vocabulary(full_df, column):
    """The exact invariant that broke: strategy_engine's live-queried
    reference vocabulary must equal what a model was actually trained
    against, or a tree's split thresholds silently point at the wrong
    category.
    """
    _reference_dtype.cache_clear()
    trained = apply_categorical_dtypes(full_df, [column])[column].cat.categories
    inference = _reference_dtype(column).categories

    assert list(trained) == list(inference), (
        f"Training and inference vocabularies for {column!r} disagree — "
        "a loaded model's category codes would no longer match what strategy_engine encodes."
    )


def test_reference_dtype_is_a_categorical_dtype():
    _reference_dtype.cache_clear()
    dtype = _reference_dtype("driver_id")
    assert isinstance(dtype, pd.CategoricalDtype)
    assert len(dtype.categories) > 0
