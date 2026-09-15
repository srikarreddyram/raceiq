"""Sequence construction for the LSTM lap-time model.

Unlike models/lap_time/train.py (which predicts the *next* lap from the
*current* lap's own realized time — an autoregressive framing that broke
down when chained across 30+ simulated laps, see
strategy_engine/simulation/monte_carlo.py's docstring), this model
predicts each lap's time directly from that lap's own tyre/weather/race-
state covariates. Those covariates are exactly what a candidate strategy
already determines in advance (tyre_age, compound, stint_number — see
strategy_engine/search/candidates.py's deterministic plan), so a trained
model here can score an entire remaining-race trajectory in one forward
pass, with no self-referential feedback loop to go unstable.

One sequence = one driver's full race. Padded to the longest race in a
batch; a mask marks which timesteps are real so padding never contributes
to the loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from models.common.data import load_race_features
from models.common.splits import temporal_split

SEQUENCE_NUMERIC_FEATURES = [
    "tyre_age",
    "stint_number",
    "degradation_rate",
    "grip_estimate",
    "laps_remaining",
    "track_temp",
    "air_temp",
    "humidity",
    "wind_speed",
    "gap_to_car_ahead",
    "gap_to_car_behind",
    "current_position",
    "field_avg_lap_time_seconds",
]
SEQUENCE_BOOLEAN_FEATURES = [
    "is_pit_lap",
    "safety_car_active",
    "yellow_active",
    "vsc_active",
    "red_flag_active",
    "rainfall_flag",
]
STATIC_NUMERIC_FEATURES = [
    "driver_avg_pace_delta",
    "driver_consistency_score",
    "condition_delta",
    "historical_sc_rate",
]
CATEGORICAL_FEATURES = ["driver_id", "team_id", "circuit_id", "compound"]
TARGET = "lap_time_seconds"

NUM_NUMERIC = len(SEQUENCE_NUMERIC_FEATURES) + len(SEQUENCE_BOOLEAN_FEATURES) + len(STATIC_NUMERIC_FEATURES)


@dataclass
class Vocab:
    """Integer-encodes a categorical column; index 0 is reserved for
    values never seen in training (a new driver/team/circuit at inference
    time degrades to an "unknown" embedding rather than erroring).
    """

    token_to_index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, values: pd.Series) -> "Vocab":
        vocab = cls()
        for i, value in enumerate(sorted(values.dropna().unique()), start=1):
            vocab.token_to_index[value] = i
        return vocab

    def encode(self, value) -> int:
        return self.token_to_index.get(value, 0)

    def __len__(self) -> int:
        return len(self.token_to_index) + 1  # +1 for the unknown index


class LapSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, vocabs: dict[str, Vocab]):
        self.examples = []
        df = df.sort_values(["race_id", "driver_id", "lap_number"])

        for (_, _), group in df.groupby(["race_id", "driver_id"], sort=False):
            if len(group) < 3:  # too short to be a meaningful sequence (e.g. lap-1 retirement)
                continue

            numeric = group[SEQUENCE_NUMERIC_FEATURES].to_numpy(dtype=np.float32)
            numeric = np.nan_to_num(numeric, nan=0.0)
            boolean = group[SEQUENCE_BOOLEAN_FEATURES].astype(float).to_numpy(dtype=np.float32)
            static = group[STATIC_NUMERIC_FEATURES].iloc[0].to_numpy(dtype=np.float32)
            static = np.nan_to_num(static, nan=0.0)
            static_tiled = np.tile(static, (len(group), 1))

            features = np.concatenate([numeric, boolean, static_tiled], axis=1)

            categorical = np.stack(
                [
                    np.full(len(group), vocabs["driver_id"].encode(group["driver_id"].iloc[0])),
                    np.full(len(group), vocabs["team_id"].encode(group["team_id"].iloc[0])),
                    np.full(len(group), vocabs["circuit_id"].encode(group["circuit_id"].iloc[0])),
                    group["compound"].map(vocabs["compound"].encode).to_numpy(),
                ],
                axis=1,
            ).astype(np.int64)

            target = group[TARGET].to_numpy(dtype=np.float32)
            # A lap during which a red flag is thrown has "lap time" spanning
            # the full session-clock stoppage in FastF1's raw data — one
            # 2024 race shows every driver's lap 1 recorded at ~2,500
            # seconds (42 minutes) after a red flag, not a real pace value.
            # Training or evaluating a lap-time model against that would be
            # meaningless (no team expects to predict a stoppage duration as
            # "pace"), and its huge squared error dominates an MSE-based
            # loss disproportionately once the model gets good enough at
            # everything else that this becomes the largest residual by far.
            red_flag = group["red_flag_active"].to_numpy(dtype=bool)
            valid = ~np.isnan(target) & ~red_flag
            if valid.sum() < 3:
                continue
            target = np.nan_to_num(target, nan=0.0)

            self.examples.append(
                {
                    "features": torch.from_numpy(features),
                    "categorical": torch.from_numpy(categorical),
                    "target": torch.from_numpy(target),
                    "mask": torch.from_numpy(valid),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate(batch: list[dict]) -> dict:
    lengths = [len(item["target"]) for item in batch]
    max_len = max(lengths)
    n_categorical = batch[0]["categorical"].shape[1]
    n_features = batch[0]["features"].shape[1]

    features = torch.zeros(len(batch), max_len, n_features)
    categorical = torch.zeros(len(batch), max_len, n_categorical, dtype=torch.long)
    target = torch.zeros(len(batch), max_len)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = len(item["target"])
        features[i, :length] = item["features"]
        categorical[i, :length] = item["categorical"]
        target[i, :length] = item["target"]
        mask[i, :length] = item["mask"]

    return {"features": features, "categorical": categorical, "target": target, "mask": mask}


def build_vocabs(df: pd.DataFrame) -> dict[str, Vocab]:
    return {col: Vocab.build(df[col]) for col in CATEGORICAL_FEATURES}


def load_datasets() -> tuple[LapSequenceDataset, LapSequenceDataset, LapSequenceDataset, dict[str, Vocab]]:
    df = load_race_features()
    # Vocabularies are fit on the full dataset (all seasons) rather than
    # just train — the same reasoning as strategy_engine/model_features.py:
    # a category's encoding must be stable across splits, and an unseen
    # driver/team/circuit at inference time should degrade to the
    # reserved "unknown" index rather than error.
    vocabs = build_vocabs(df)

    train_df, val_df, test_df = temporal_split(df)
    return (
        LapSequenceDataset(train_df, vocabs),
        LapSequenceDataset(val_df, vocabs),
        LapSequenceDataset(test_df, vocabs),
        vocabs,
    )
