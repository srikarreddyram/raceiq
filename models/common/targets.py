"""Target engineering for all six models.

Every function here adds a target column that looks *forward* in time —
that's expected and correct for a supervised-learning target (you're
always predicting something not yet known). The leakage rule that matters
is about FEATURES, not targets: gold.race_features' columns are all
computed from information available at-or-before the current lap (see
pipelines/gold/lap_features.py), so as long as a model's feature set stays
within that table, adding a forward-looking target here doesn't
reintroduce the leakage the Gold layer was built to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.common.data import load_final_classifications


def add_next_lap_time_target(df: pd.DataFrame) -> pd.DataFrame:
    """PRD 11.1 — `next_lap_time_seconds`, the lap time oracle's target."""
    df = df.sort_values(["race_id", "driver_id", "lap_number"]).copy()
    df["next_lap_time_seconds"] = df.groupby(["race_id", "driver_id"])["lap_time_seconds"].shift(-1)
    return df


def add_remaining_tyre_life_target(df: pd.DataFrame) -> pd.DataFrame:
    """PRD 11.2 — `predicted_remaining_life_laps`.

    Defined as (this stint's observed final tyre age) minus (tyre age at
    this lap). For the last stint of a race this measures life used up to
    the chequered flag, not life the tyre could still have given, which
    matches how every tyre-degradation dataset like this is necessarily
    built: you only ever observe a stint's true remaining life at the lap
    it actually ended.
    """
    df = df.copy()
    stint_max_age = df.groupby(["race_id", "driver_id", "stint_number"])["tyre_age"].transform("max")
    df["predicted_remaining_life_laps"] = stint_max_age - df["tyre_age"]
    return df


def add_safety_car_within_n_target(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """PRD 11.4 — `safety_car_within_N_laps`.

    Safety car status is race-wide (identical across every driver at a
    given lap by construction — see the ASOF join in lap_features.py), so
    this collapses to one boolean series per (race_id, lap_number), looks
    ahead N laps on that race-level series, then broadcasts the result
    back to every driver's row for that lap.
    """
    race_level = (
        df.groupby(["race_id", "lap_number"])["safety_car_active"].max().reset_index()
        .sort_values(["race_id", "lap_number"])
    )
    race_level["safety_car_within_n_laps"] = (
        race_level.groupby("race_id")["safety_car_active"]
        .transform(lambda s: s.shift(-1).rolling(window=n, min_periods=1).max())
        .fillna(0)
        .astype(int)
    )
    return df.merge(
        race_level[["race_id", "lap_number", "safety_car_within_n_laps"]],
        on=["race_id", "lap_number"],
        how="left",
    )


def add_race_outcome_targets(df: pd.DataFrame) -> pd.DataFrame:
    """PRD 11.5/11.6 — `final_position` and `wins_race`, from Ergast results."""
    finals = load_final_classifications()
    df = df.merge(finals, on=["season", "round", "driver_id"], how="left")
    df["wins_race"] = np.where(df["final_position"].notna(), (df["final_position"] == 1).astype(int), np.nan)
    return df
