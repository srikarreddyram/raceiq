"""LSTM lap-time model architecture.

Bidirectional, not causal: this deliberately looks both directions along
the sequence, which would be a leakage bug for a normal "forecast the
future from the past" model — but here it's correct. The whole point is
that a candidate strategy already fixes every lap's tyre/pit covariates
in advance (see data.py's docstring), so this is a sequence-labeling
problem (every input known upfront, predict every output at once), not a
forecasting one — closer to part-of-speech tagging than to next-token
prediction.
"""

from __future__ import annotations

import torch
from torch import nn

EMBEDDING_DIMS = {"driver_id": 8, "team_id": 4, "circuit_id": 6, "compound": 3}


class LapTimeSequenceModel(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], num_numeric_features: int, hidden_size: int = 64):
        super().__init__()
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes[name], dim) for name, dim in EMBEDDING_DIMS.items()}
        )
        embedding_total = sum(EMBEDDING_DIMS.values())

        self.lstm = nn.LSTM(
            input_size=num_numeric_features + embedding_total,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.output = nn.Linear(hidden_size * 2, 1)

    def forward(self, features: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        embedded = [
            self.embeddings[name](categorical[:, :, i])
            for i, name in enumerate(EMBEDDING_DIMS)
        ]
        x = torch.cat([features] + embedded, dim=-1)
        out, _ = self.lstm(x)
        return self.output(out).squeeze(-1)
