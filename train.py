from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from mlp_numpy.data_utils import load_eurosat_dataset, train_val_test_counts
from mlp_numpy.metrics import (
    classification_report_dict,
    confusion_matrix,
    save_json,
)
from mlp_numpy.mlp import MLPClassifier
from mlp_numpy.trainer import evaluate_split, load_checkpoint, train_model
from mlp_numpy.visualization import (
    save_confusion_matrix,
    save_misclassified_grid,
    save_training_curves,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy-only MLP on EuroSAT RGB.")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="results/final_model")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[512, 256])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    dataset = load_eurosat_dataset(
        root_dir=args.data_dir,
        image_size=args.image_size,
        seed=args.seed,
    )

    model = MLPClassifier(
        input_dim=dataset.input_dim,
        hidden_dims=tuple(args.hidden_dims),
        num_classes=len(dataset.class_names),
        activation=args.activation,
        seed=args.seed,
    )

    checkpoint_path = output_dir / "best_model.npz"
    history = train_model(
        model=model,
        train_split=dataset.train,
        val_split=dataset.val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_decay=args.lr_decay,
        weight_decay=args.weight_decay,
        checkpoint_path=checkpoint_path,
        patience=args.patience,
        seed=args.seed,
    )

    load_checkpoint(model, checkpoint_path)
    test_loss, test_accuracy, test_predictions = evaluate_split(
        model,
        dataset.test,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )

    matrix = confusion_matrix(dataset.test.labels, test_predictions, len(dataset.class_names))
    report = classification_report_dict(dataset.test.labels, test_predictions, dataset.class_names)

    history_payload = [
        {
            "epoch": item.epoch,
            "learning_rate": item.learning_rate,
            "train_loss": item.train_loss,
            "train_accuracy": item.train_accuracy,
            "val_loss": item.val_loss,
            "val_accuracy": item.val_accuracy,
        }
        for item in history
    ]
    best_epoch = max(history, key=lambda item: item.val_accuracy)
    metrics_payload = {
        "config": vars(args),
        "dataset_counts": train_val_test_counts(dataset),
        "channel_mean": dataset.channel_mean.tolist(),
        "channel_std": dataset.channel_std.tolist(),
        "best_epoch": best_epoch.epoch,
        "stopped_epoch": history[-1].epoch,
        "best_val_accuracy": best_epoch.val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "per_class": report,
        "confusion_matrix": matrix.tolist(),
        "elapsed_seconds": time.time() - start_time,
    }

    save_json({"history": history_payload}, output_dir / "history.json")
    save_json(metrics_payload, output_dir / "metrics.json")
    save_training_curves(history, output_dir / "training_curves.png")
    save_confusion_matrix(matrix, dataset.class_names, output_dir / "confusion_matrix.png")
    misclassified_indices = np.where(dataset.test.labels != test_predictions)[0]
    if len(misclassified_indices) > 0:
        save_misclassified_grid(
            image_paths=dataset.test.paths[misclassified_indices].tolist(),
            true_names=[dataset.class_names[idx] for idx in dataset.test.labels[misclassified_indices]],
            pred_names=[dataset.class_names[idx] for idx in test_predictions[misclassified_indices]],
            output_path=output_dir / "misclassified_examples.png",
        )

    print("Training finished.")
    print(f"Stopped epoch: {history[-1].epoch}")
    print(f"Best epoch: {best_epoch.epoch}")
    print(f"Best val accuracy: {best_epoch.val_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
