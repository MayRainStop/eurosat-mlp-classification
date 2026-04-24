from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from PIL import UnidentifiedImageError


@dataclass(frozen=True)
class DatasetSplit:
    features: np.ndarray
    labels: np.ndarray
    paths: np.ndarray


@dataclass(frozen=True)
class DatasetBundle:
    class_names: list[str]
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    image_size: int
    input_dim: int
    channel_mean: np.ndarray
    channel_std: np.ndarray


def load_eurosat_dataset(
    root_dir: str | Path,
    image_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> DatasetBundle:
    root = Path(root_dir)
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {root.resolve()}")

    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    skipped_paths: list[str] = []
    class_names = [path.name for path in class_dirs]

    for class_index, class_dir in enumerate(class_dirs):
        image_paths = sorted(class_dir.glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(f"No .jpg files found in {class_dir.resolve()}")
        for image_path in image_paths:
            try:
                features.append(_load_image(image_path, image_size))
                labels.append(class_index)
                paths.append(str(image_path.resolve()))
            except (UnidentifiedImageError, OSError):
                skipped_paths.append(str(image_path.resolve()))

    if skipped_paths:
        print(f"Skipped {len(skipped_paths)} unreadable images.")

    X = np.vstack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    sample_paths = np.asarray(paths, dtype=object)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    X = X[indices]
    y = y[indices]
    sample_paths = sample_paths[indices]

    total = len(y)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    train_paths = sample_paths[:train_end]
    val_paths = sample_paths[train_end:val_end]
    test_paths = sample_paths[val_end:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    channel_mean = mean.reshape(image_size, image_size, 3).mean(axis=(0, 1))
    channel_std = std.reshape(image_size, image_size, 3).mean(axis=(0, 1))

    return DatasetBundle(
        class_names=class_names,
        train=DatasetSplit(X_train, y_train, train_paths),
        val=DatasetSplit(X_val, y_val, val_paths),
        test=DatasetSplit(X_test, y_test, test_paths),
        image_size=image_size,
        input_dim=X_train.shape[1],
        channel_mean=channel_mean.astype(np.float32),
        channel_std=channel_std.astype(np.float32),
    )


def iterate_minibatches(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    rng: np.random.Generator | None = None,
    shuffle: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng()
    indices = np.arange(len(labels))
    if shuffle:
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield features[batch_indices], labels[batch_indices]


def train_val_test_counts(bundle: DatasetBundle) -> dict[str, int]:
    return {
        "train": len(bundle.train.labels),
        "val": len(bundle.val.labels),
        "test": len(bundle.test.labels),
    }


def _load_image(image_path: Path, image_size: int) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.reshape(-1)
