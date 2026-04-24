from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label, pred_label] += 1
    return matrix


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, class_name in enumerate(class_names):
        tp = float(np.sum((y_true == index) & (y_pred == index)))
        fp = float(np.sum((y_true != index) & (y_pred == index)))
        fn = float(np.sum((y_true == index) & (y_pred != index)))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        support = int(np.sum(y_true == index))
        report[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return report


def save_json(payload: dict, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
