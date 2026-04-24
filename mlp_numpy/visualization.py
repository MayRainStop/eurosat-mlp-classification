from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("results") / ".matplotlib_cache").resolve()))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image, ImageDraw, ImageFont

from .trainer import EpochMetrics


PAPER_DPI = 400
PAPER_COLORS = {
    "train": "#1f77b4",
    "val": "#d62728",
    "best": "#2ca02c",
}


def save_loss_curves(history: list[EpochMetrics], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([item.epoch for item in history], dtype=np.float32)
    train_loss = np.array([item.train_loss for item in history], dtype=np.float32)
    val_loss = np.array([item.val_loss for item in history], dtype=np.float32)
    best_epoch = max(history, key=lambda item: item.val_accuracy)

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(epochs, train_loss, label="Training loss", color=PAPER_COLORS["train"], linewidth=2.4)
    ax.plot(epochs, val_loss, label="Validation loss", color=PAPER_COLORS["val"], linewidth=2.4)
    ax.axvline(
        best_epoch.epoch,
        color=PAPER_COLORS["best"],
        linestyle="--",
        linewidth=1.8,
        label=f"Best epoch ({best_epoch.epoch})",
    )
    ax.set_title("Training and Validation Loss", pad=12, weight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss with L2 regularization")
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=PAPER_DPI, facecolor="white")
    plt.close(fig)


def save_validation_accuracy_curve(history: list[EpochMetrics], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([item.epoch for item in history], dtype=np.float32)
    val_accuracy = np.array([item.val_accuracy for item in history], dtype=np.float32)
    best_epoch = max(history, key=lambda item: item.val_accuracy)

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(epochs, val_accuracy, label="Validation accuracy", color=PAPER_COLORS["val"], linewidth=2.4)
    ax.scatter(
        [best_epoch.epoch],
        [best_epoch.val_accuracy],
        s=70,
        color=PAPER_COLORS["best"],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        label=f"Best: {best_epoch.val_accuracy:.2%}",
    )
    ax.axvline(best_epoch.epoch, color=PAPER_COLORS["best"], linestyle="--", linewidth=1.6, alpha=0.8)
    ax.set_title("Validation Accuracy", pad=12, weight="semibold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(max(0.0, float(val_accuracy.min()) - 0.05), min(1.0, float(val_accuracy.max()) + 0.05))
    ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#cccccc", loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(output_path, dpi=PAPER_DPI, facecolor="white")
    plt.close(fig)


def save_training_curves(history: list[EpochMetrics], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.array([item.epoch for item in history], dtype=np.float32)
    train_loss = np.array([item.train_loss for item in history], dtype=np.float32)
    val_loss = np.array([item.val_loss for item in history], dtype=np.float32)
    train_acc = np.array([item.train_accuracy for item in history], dtype=np.float32)
    val_acc = np.array([item.val_accuracy for item in history], dtype=np.float32)
    best_epoch = max(history, key=lambda item: item.val_accuracy)

    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    axes[0].plot(epochs, train_loss, label="Training loss", color=PAPER_COLORS["train"], linewidth=2.3)
    axes[0].plot(epochs, val_loss, label="Validation loss", color=PAPER_COLORS["val"], linewidth=2.3)
    axes[0].axvline(best_epoch.epoch, color=PAPER_COLORS["best"], linestyle="--", linewidth=1.6)
    axes[0].set_title("Loss", weight="semibold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(frameon=True, framealpha=0.95, edgecolor="#cccccc")

    axes[1].plot(epochs, train_acc, label="Training accuracy", color=PAPER_COLORS["train"], linewidth=2.3)
    axes[1].plot(epochs, val_acc, label="Validation accuracy", color=PAPER_COLORS["val"], linewidth=2.3)
    axes[1].scatter(
        [best_epoch.epoch],
        [best_epoch.val_accuracy],
        s=70,
        color=PAPER_COLORS["best"],
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
        label=f"Best val: {best_epoch.val_accuracy:.2%}",
    )
    axes[1].axvline(best_epoch.epoch, color=PAPER_COLORS["best"], linestyle="--", linewidth=1.6)
    axes[1].set_title("Accuracy", weight="semibold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(frameon=True, framealpha=0.95, edgecolor="#cccccc")

    for ax in axes:
        ax.grid(True, which="major", color="#d9d9d9", linewidth=0.8, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Training Dynamics of the Best MLP", fontsize=15, weight="semibold")
    fig.savefig(output_path, dpi=PAPER_DPI, facecolor="white")
    plt.close(fig)


def save_first_layer_weight_grid(
    weights: np.ndarray,
    image_size: int,
    output_path: str | Path,
    columns: int = 64,
    padding: int = 2,
    tile_scale: int = 3,
    title: str | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if weights.ndim != 2:
        raise ValueError(f"Expected a 2D first-layer weight matrix, got shape {weights.shape}.")

    input_dim, hidden_dim = weights.shape
    expected_dim = image_size * image_size * 3
    if input_dim != expected_dim:
        raise ValueError(f"Expected input dimension {expected_dim}, got {input_dim}.")

    rows = int(np.ceil(hidden_dim / columns))
    tile_size = image_size * tile_scale
    width = columns * tile_size + (columns + 1) * padding
    title_height = 90 if title else 0
    height = rows * tile_size + (rows + 1) * padding + title_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        draw.text((padding, 24), title, fill=(20, 20, 20), font=font)

    for hidden_index in range(hidden_dim):
        row = hidden_index // columns
        col = hidden_index % columns
        x = padding + col * (tile_size + padding)
        y = title_height + padding + row * (tile_size + padding)

        filter_image = weights[:, hidden_index].reshape(image_size, image_size, 3)
        tile = _normalize_filter_image(filter_image)
        tile_image = Image.fromarray(tile, mode="RGB").resize((tile_size, tile_size), Image.Resampling.NEAREST)
        canvas.paste(tile_image, (x, y))

    canvas.save(output_path)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _set_paper_style()
    fig, ax = plt.subplots(figsize=(8.8, 7.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.set_ylabel("Number of test samples", rotation=270, labelpad=18)

    ax.set_title("Confusion Matrix on the Test Set", pad=14, weight="semibold")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(class_names)

    threshold = matrix.max() * 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            color = "white" if matrix[row, col] > threshold else "#1a1a1a"
            ax.text(col, row, str(int(matrix[row, col])), ha="center", va="center", color=color, fontsize=8.5)

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.savefig(output_path, dpi=PAPER_DPI, facecolor="white")
    plt.close(fig)


def save_misclassified_grid(
    image_paths: list[str],
    true_names: list[str],
    pred_names: list[str],
    output_path: str | Path,
    max_items: int = 16,
    thumb_size: int = 96,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(zip(image_paths, true_names, pred_names))[:max_items]
    if not items:
        return

    cols = 4
    rows = max((len(items) + cols - 1) // cols, 1)

    _set_paper_style()
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.25), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(rows, cols)

    for index, (path, true_name, pred_name) in enumerate(items):
        row = index // cols
        col = index % cols
        ax = axes_array[row, col]
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", fontsize=9, pad=6)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("#444444")

    for index in range(len(items), rows * cols):
        row = index // cols
        col = index % cols
        axes_array[row, col].axis("off")

    fig.suptitle("Representative Misclassified Test Images", fontsize=14, weight="semibold")
    fig.savefig(output_path, dpi=PAPER_DPI, facecolor="white")
    plt.close(fig)


def _normalize_filter_image(filter_image: np.ndarray) -> np.ndarray:
    filter_image = filter_image.astype(np.float32)
    centered = filter_image - float(np.mean(filter_image))
    scale = float(np.percentile(np.abs(centered), 99.0))
    if scale < 1e-8:
        normalized = np.full_like(filter_image, 0.5, dtype=np.float32)
    else:
        normalized = 0.5 + centered / (2.0 * scale)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": PAPER_DPI,
            "savefig.dpi": PAPER_DPI,
            "axes.linewidth": 1.0,
            "lines.solid_capstyle": "round",
        }
    )


def _draw_series_panel(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    origin_x: int,
    origin_y: int,
    width: int,
    height: int,
    x_values: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
) -> None:
    x0, y0 = origin_x, origin_y
    x1, y1 = origin_x + width, origin_y + height
    draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
    draw.text((x0, y0 - 28), title, fill="black", font=font)

    y_values = np.concatenate([values for _, values, _ in series])
    y_min = float(np.min(y_values))
    y_max = float(np.max(y_values))
    if abs(y_max - y_min) < 1e-8:
        y_max += 1.0
        y_min -= 1.0

    for step in range(6):
        y = y0 + height * step / 5.0
        draw.line([x0, y, x1, y], fill=(220, 220, 220))
        tick_value = y_max - (y_max - y_min) * step / 5.0
        draw.text((x0 - 70, y - 8), f"{tick_value:.2f}", fill="black", font=font)

    for label, values, color in series:
        points: list[tuple[float, float]] = []
        for x_value, y_value in zip(x_values, values):
            x = x0 + (x_value - x_values.min()) / max(x_values.max() - x_values.min(), 1.0) * width
            y = y0 + (y_max - y_value) / (y_max - y_min) * height
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=color, width=3)

    legend_x = x0
    legend_y = y1 + 20
    for idx, (label, _, color) in enumerate(series):
        draw.rectangle([legend_x, legend_y + idx * 24, legend_x + 16, legend_y + idx * 24 + 16], fill=color)
        draw.text((legend_x + 24, legend_y + idx * 24), label, fill="black", font=font)
