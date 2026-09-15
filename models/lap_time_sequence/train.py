"""Train the LSTM lap-time sequence model and compare it against the
LightGBM baseline (models/lap_time/train.py) on the same test season,
using the same "stable regime" (no pit/SC/yellow/VSC/red) slice for a fair
comparison against that model's own headline numbers.

Usage:
    uv run python -m models.lap_time_sequence.train
"""

from __future__ import annotations

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.common.tracking import configure_experiment
from models.lap_time_sequence.data import (
    NUM_NUMERIC,
    SEQUENCE_BOOLEAN_FEATURES,
    SEQUENCE_NUMERIC_FEATURES,
    collate,
    load_datasets,
)
from models.lap_time_sequence.model import LapTimeSequenceModel

EXPERIMENT_NAME = "lap_time_prediction"
RUN_NAME = "lstm_sequence"

BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 10
LEARNING_RATE = 1e-3


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) ** 2
    return (diff * mask).sum() / mask.sum().clamp(min=1)


def evaluate(model, loader) -> dict:
    model.eval()
    all_pred, all_true, all_stable = [], [], []
    # "is_pit_lap"/"safety_car_active"/"yellow_active"/"vsc_active"/"red_flag_active"
    # are the first five boolean columns appended right after the numeric
    # sequence features — see data.py's feature concatenation order.
    boolean_start = len(SEQUENCE_NUMERIC_FEATURES)
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["features"], batch["categorical"])
            mask = batch["mask"]
            all_pred.append(pred[mask].numpy())
            all_true.append(batch["target"][mask].numpy())
            booleans = batch["features"][:, :, boolean_start : boolean_start + len(SEQUENCE_BOOLEAN_FEATURES)]
            stable = (booleans.sum(dim=-1) == 0) & mask
            all_stable.append(stable[mask].numpy())

    pred = np.concatenate(all_pred)
    true = np.concatenate(all_true)
    stable = np.concatenate(all_stable)

    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    stable_rmse = float(np.sqrt(np.mean((pred[stable] - true[stable]) ** 2))) if stable.any() else float("nan")
    stable_mae = float(np.mean(np.abs(pred[stable] - true[stable]))) if stable.any() else float("nan")
    return {"rmse": rmse, "mae": mae, "stable_rmse": stable_rmse, "stable_mae": stable_mae}


def main() -> None:
    configure_experiment(EXPERIMENT_NAME)

    train_ds, val_ds, test_ds, vocabs = load_datasets()
    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} sequences")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate)

    vocab_sizes = {name: len(vocab) for name, vocab in vocabs.items()}
    model = LapTimeSequenceModel(vocab_sizes, num_numeric_features=NUM_NUMERIC)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_rmse = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            pred = model(batch["features"], batch["categorical"])
            loss = masked_mse(pred, batch["target"], batch["mask"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, val_loader)
        print(f"epoch {epoch}: train_loss={total_loss / len(train_loader):.3f} val_rmse={val_metrics['rmse']:.3f}")

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader)
    print(f"test: {test_metrics}")

    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.set_tag("algorithm", "lstm")
        mlflow.set_tag("role", "comparison")
        mlflow.log_params(
            {"batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "hidden_size": 64, "num_layers": 2}
        )
        mlflow.log_metrics(
            {
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_rmse_stable_regime": test_metrics["stable_rmse"],
                "test_mae_stable_regime": test_metrics["stable_mae"],
                "best_val_rmse": best_val_rmse,
            }
        )
        mlflow.pytorch.log_model(model, name="model", serialization_format="pickle")


if __name__ == "__main__":
    main()
