from __future__ import annotations

import argparse
import csv
from pathlib import Path

from mlp_numpy.data_utils import load_eurosat_dataset
from mlp_numpy.metrics import save_json
from mlp_numpy.mlp import MLPClassifier
from mlp_numpy.trainer import evaluate_split, load_checkpoint, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small hyperparameter sweep for the NumPy MLP.")
    parser.add_argument("--data-dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--output-dir", type=str, default="results/search")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--hidden-dim1s", type=int, nargs="+", default=[512])
    parser.add_argument("--hidden-dim2s", type=int, nargs="+", default=[256])
    parser.add_argument("--activations", type=str, nargs="+", default=["relu", "tanh"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.05])
    parser.add_argument("--lr-decay", type=float, default=0.95)
    parser.add_argument("--weight-decays", type=float, nargs="+", default=[1e-4])
    parser.add_argument("--patience", type=int, default=10)
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

    rows: list[dict[str, float | int | str]] = []
    for hidden_dim1 in args.hidden_dim1s:
        for hidden_dim2 in args.hidden_dim2s:
            if hidden_dim2 > hidden_dim1:
                continue
            for activation in args.activations:
                for learning_rate in args.learning_rates:
                    for weight_decay in args.weight_decays:
                        run_name = (
                            f"h{hidden_dim1}_{hidden_dim2}_{activation}_"
                            f"lr{learning_rate:g}_wd{weight_decay:g}"
                        )
                        run_dir = output_dir / run_name
                        checkpoint_path = run_dir / "best_model.npz"

                        model = MLPClassifier(
                            input_dim=dataset.input_dim,
                            hidden_dims=(hidden_dim1, hidden_dim2),
                            num_classes=len(dataset.class_names),
                            activation=activation,
                            seed=args.seed,
                        )
                        history = train_model(
                            model=model,
                            train_split=dataset.train,
                            val_split=dataset.val,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=learning_rate,
                            lr_decay=args.lr_decay,
                            weight_decay=weight_decay,
                            checkpoint_path=checkpoint_path,
                            patience=args.patience,
                            seed=args.seed,
                        )
                        load_checkpoint(model, checkpoint_path)
                        test_loss, test_accuracy, _ = evaluate_split(
                            model,
                            dataset.test,
                            batch_size=args.batch_size,
                            weight_decay=weight_decay,
                        )
                        best_epoch = max(history, key=lambda item: item.val_accuracy)
                        rows.append(
                            {
                                "run_name": run_name,
                                "hidden_dim1": hidden_dim1,
                                "hidden_dim2": hidden_dim2,
                                "activation": activation,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "stopped_epoch": history[-1].epoch,
                                "best_epoch": best_epoch.epoch,
                                "best_val_accuracy": best_epoch.val_accuracy,
                                "test_loss": test_loss,
                                "test_accuracy": test_accuracy,
                            }
                        )
                        print(
                            f"{run_name}: best_val_accuracy={best_epoch.val_accuracy:.4f}, "
                            f"test_accuracy={test_accuracy:.4f}"
                        )

    csv_path = output_dir / "experiment_summary.csv"
    if csv_path.exists():
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(
                    {
                        "run_name": row["run_name"],
                        "hidden_dim1": int(row["hidden_dim1"]),
                        "hidden_dim2": int(row["hidden_dim2"]),
                        "activation": row["activation"],
                        "learning_rate": float(row["learning_rate"]),
                        "weight_decay": float(row["weight_decay"]),
                        "stopped_epoch": int(row["stopped_epoch"]),
                        "best_epoch": int(row["best_epoch"]),
                        "best_val_accuracy": float(row["best_val_accuracy"]),
                        "test_loss": float(row["test_loss"]),
                        "test_accuracy": float(row["test_accuracy"]),
                    }
                )

    deduped: dict[str, dict[str, float | int | str]] = {}
    for row in rows:
        deduped[str(row["run_name"])] = row
    rows = list(deduped.values())
    rows.sort(
        key=lambda item: (float(item["best_val_accuracy"]), float(item["test_accuracy"])),
        reverse=True,
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json({"runs": rows}, output_dir / "experiment_summary.json")
    print(f"Saved experiment summary to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()
