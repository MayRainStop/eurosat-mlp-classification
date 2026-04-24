from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mlp_numpy.data_utils import load_eurosat_dataset
from mlp_numpy.metrics import classification_report_dict, confusion_matrix, save_json
from mlp_numpy.mlp import MLPClassifier
from mlp_numpy.trainer import evaluate_split, load_checkpoint
from mlp_numpy.visualization import save_confusion_matrix, save_misclassified_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved NumPy MLP checkpoint on EuroSAT RGB.")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/evaluation")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dims", type=int, nargs=2, default=[512, 256])
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "sigmoid", "tanh"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    load_checkpoint(model, args.checkpoint)

    test_loss, test_accuracy, test_predictions = evaluate_split(
        model,
        dataset.test,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
    )
    matrix = confusion_matrix(dataset.test.labels, test_predictions, len(dataset.class_names))
    report = classification_report_dict(dataset.test.labels, test_predictions, dataset.class_names)

    save_json(
        {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "per_class": report,
            "confusion_matrix": matrix.tolist(),
        },
        output_dir / "evaluation_metrics.json",
    )
    save_confusion_matrix(matrix, dataset.class_names, output_dir / "evaluation_confusion_matrix.png")
    misclassified_indices = np.where(dataset.test.labels != test_predictions)[0]
    if len(misclassified_indices) > 0:
        save_misclassified_grid(
            image_paths=dataset.test.paths[misclassified_indices].tolist(),
            true_names=[dataset.class_names[idx] for idx in dataset.test.labels[misclassified_indices]],
            pred_names=[dataset.class_names[idx] for idx in test_predictions[misclassified_indices]],
            output_path=output_dir / "evaluation_misclassified_examples.png",
        )

    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
