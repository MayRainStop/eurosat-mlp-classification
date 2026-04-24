from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data_utils import DatasetSplit, iterate_minibatches
from .metrics import accuracy_score
from .mlp import MLPClassifier


@dataclass
class EpochMetrics:
    epoch: int
    learning_rate: float
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


def evaluate_split(
    model: MLPClassifier,
    split: DatasetSplit,
    batch_size: int,
    weight_decay: float,
) -> tuple[float, float, np.ndarray]:
    total_loss = 0.0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    total_examples = 0

    for batch_x, batch_y in iterate_minibatches(split.features, split.labels, batch_size, shuffle=False):
        loss, _ = model.compute_loss_and_grads(batch_x, batch_y, weight_decay=weight_decay)
        preds = model.predict(batch_x)
        total_loss += loss * len(batch_y)
        total_examples += len(batch_y)
        predictions.append(preds)
        labels.append(batch_y)

    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    avg_loss = total_loss / max(total_examples, 1)
    accuracy = accuracy_score(y_true, y_pred)
    return float(avg_loss), accuracy, y_pred


def train_model(
    model: MLPClassifier,
    train_split: DatasetSplit,
    val_split: DatasetSplit,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lr_decay: float,
    weight_decay: float,
    checkpoint_path: str | Path,
    patience: int | None = None,
    seed: int = 42,
) -> list[EpochMetrics]:
    history: list[EpochMetrics] = []
    rng = np.random.default_rng(seed)
    best_val_accuracy = -np.inf
    epochs_without_improvement = 0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        current_lr = learning_rate * (lr_decay ** (epoch - 1))
        running_loss = 0.0
        running_correct = 0
        total = 0

        for batch_x, batch_y in iterate_minibatches(
            train_split.features,
            train_split.labels,
            batch_size=batch_size,
            rng=rng,
            shuffle=True,
        ):
            loss, grads = model.compute_loss_and_grads(batch_x, batch_y, weight_decay=weight_decay)
            model.apply_gradients(grads, learning_rate=current_lr)

            preds = model.predict(batch_x)
            running_loss += loss * len(batch_y)
            running_correct += int(np.sum(preds == batch_y))
            total += len(batch_y)

        train_loss = running_loss / max(total, 1)
        train_accuracy = running_correct / max(total, 1)
        val_loss, val_accuracy, _ = evaluate_split(model, val_split, batch_size, weight_decay)

        metrics = EpochMetrics(
            epoch=epoch,
            learning_rate=current_lr,
            train_loss=float(train_loss),
            train_accuracy=float(train_accuracy),
            val_loss=float(val_loss),
            val_accuracy=float(val_accuracy),
        )
        history.append(metrics)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            np.savez(checkpoint_path, **model.state_dict())
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            break

    return history


def load_checkpoint(model: MLPClassifier, checkpoint_path: str | Path) -> None:
    state = np.load(Path(checkpoint_path), allow_pickle=True)
    model.load_state_dict({key: state[key] for key in state.files})
