from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mlp_numpy.data_utils import load_eurosat_dataset
from mlp_numpy.mlp import MLPClassifier
from mlp_numpy.trainer import EpochMetrics
from mlp_numpy.trainer import evaluate_split, load_checkpoint
from mlp_numpy.visualization import (
    save_confusion_matrix,
    save_first_layer_weight_grid,
    save_loss_curves,
    save_misclassified_grid,
    save_training_curves,
    save_validation_accuracy_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create submission visualizations for a trained NumPy MLP.")
    parser.add_argument("--run-dir", type=str, default="results/best_model")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--weight-grid-columns", type=int, default=64)
    parser.add_argument("--top-k-weights", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"
    checkpoint_path = run_dir / "best_model.npz"

    history = _load_history(history_path)
    checkpoint = np.load(checkpoint_path, allow_pickle=True)
    first_layer_weights = checkpoint["W1"]
    metrics = _load_metrics(run_dir / "metrics.json")
    class_names = list(metrics["per_class"].keys())
    confusion_matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    _save_misclassified_examples(args, metrics, run_dir, class_names)

    save_training_curves(history, run_dir / "training_curves.png")
    save_loss_curves(history, run_dir / "loss_curves.png")
    save_validation_accuracy_curve(history, run_dir / "validation_accuracy_curve.png")
    save_confusion_matrix(confusion_matrix, class_names, run_dir / "confusion_matrix.png")
    save_first_layer_weight_grid(
        weights=first_layer_weights,
        image_size=args.image_size,
        output_path=run_dir / "first_layer_weights.png",
        columns=args.weight_grid_columns,
        tile_scale=3,
    )
    top_indices = np.argsort(np.linalg.norm(first_layer_weights, axis=0))[::-1][: args.top_k_weights]
    save_first_layer_weight_grid(
        weights=first_layer_weights[:, top_indices],
        image_size=args.image_size,
        output_path=run_dir / f"first_layer_weights_top{args.top_k_weights}.png",
        columns=8,
        tile_scale=6,
    )

    print(f"Saved combined training curves to: {(run_dir / 'training_curves.png').resolve()}")
    print(f"Saved loss curves to: {(run_dir / 'loss_curves.png').resolve()}")
    print(f"Saved validation accuracy curve to: {(run_dir / 'validation_accuracy_curve.png').resolve()}")
    print(f"Saved confusion matrix to: {(run_dir / 'confusion_matrix.png').resolve()}")
    print(f"Saved misclassified examples to: {(run_dir / 'misclassified_examples.png').resolve()}")
    print(f"Saved first-layer weight grid to: {(run_dir / 'first_layer_weights.png').resolve()}")
    print(
        "Saved top first-layer weight grid to: "
        f"{(run_dir / f'first_layer_weights_top{args.top_k_weights}.png').resolve()}"
    )


def _load_history(history_path: Path) -> list[EpochMetrics]:
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return [
        EpochMetrics(
            epoch=int(item["epoch"]),
            learning_rate=float(item["learning_rate"]),
            train_loss=float(item["train_loss"]),
            train_accuracy=float(item["train_accuracy"]),
            val_loss=float(item["val_loss"]),
            val_accuracy=float(item["val_accuracy"]),
        )
        for item in payload["history"]
    ]


def _load_metrics(metrics_path: Path) -> dict:
    with metrics_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_misclassified_examples(
    args: argparse.Namespace,
    metrics: dict,
    run_dir: Path,
    class_names: list[str],
) -> None:
    config = metrics["config"]
    dataset = load_eurosat_dataset(
        root_dir=args.data_dir,
        image_size=int(config["image_size"]),
        seed=int(config["seed"]),
    )
    model = MLPClassifier(
        input_dim=dataset.input_dim,
        hidden_dims=tuple(config["hidden_dims"]),
        num_classes=len(dataset.class_names),
        activation=config["activation"],
        seed=int(config["seed"]),
    )
    load_checkpoint(model, run_dir / "best_model.npz")
    _, _, test_predictions = evaluate_split(
        model,
        dataset.test,
        batch_size=int(config["batch_size"]),
        weight_decay=float(config["weight_decay"]),
    )
    misclassified_indices = np.where(dataset.test.labels != test_predictions)[0]
    if len(misclassified_indices) == 0:
        return

    save_misclassified_grid(
        image_paths=dataset.test.paths[misclassified_indices].tolist(),
        true_names=[class_names[idx] for idx in dataset.test.labels[misclassified_indices]],
        pred_names=[class_names[idx] for idx in test_predictions[misclassified_indices]],
        output_path=run_dir / "misclassified_examples.png",
    )


if __name__ == "__main__":
    main()
