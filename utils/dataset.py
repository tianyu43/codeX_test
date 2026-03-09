from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class NpySequenceDataset(Dataset):
    def __init__(
        self,
        x_path: str,
        y_path: str,
        indices: np.ndarray,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> None:
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = indices.astype(np.int64, copy=False)
        self.mean = mean.astype(np.float32, copy=False) if mean is not None else None
        self.std = std.astype(np.float32, copy=False) if std is not None else None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx = int(self.indices[idx])
        x = np.array(self.x[sample_idx], dtype=np.float32, copy=True)
        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / self.std
        y = np.float32(self.y[sample_idx])
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


def sample_indices(labels: np.ndarray, sample_size: int, random_seed: int) -> np.ndarray:
    total = labels.shape[0]
    if sample_size >= total:
        return np.arange(total, dtype=np.int64)

    all_indices = np.arange(total, dtype=np.int64)
    sampled, _ = train_test_split(
        all_indices,
        train_size=sample_size,
        random_state=random_seed,
        stratify=labels,
    )
    return np.sort(sampled)


def split_train_val(
    sampled_indices: np.ndarray,
    sampled_labels: np.ndarray,
    val_ratio: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        sampled_indices,
        test_size=val_ratio,
        random_state=random_seed,
        stratify=sampled_labels,
    )
    return train_idx, val_idx


def compute_normalization_stats(
    x_path: str,
    train_indices: np.ndarray,
    max_stats_samples: int,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path, mmap_mode="r")
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    if len(train_indices) > max_stats_samples:
        chosen = rng.choice(train_indices, size=max_stats_samples, replace=False)
    else:
        chosen = train_indices
    subset = np.asarray(x[chosen], dtype=np.float32)
    mean = subset.mean(axis=(0, 1))
    std = subset.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def sample_holdout_indices(
    labels: np.ndarray,
    used_indices: np.ndarray,
    holdout_size: int,
    random_seed: int,
) -> np.ndarray:
    used_mask = np.zeros(labels.shape[0], dtype=bool)
    used_mask[used_indices] = True
    remaining_indices = np.flatnonzero(~used_mask)
    remaining_labels = np.asarray(labels[remaining_indices], dtype=np.int64)
    if holdout_size >= len(remaining_indices):
        return np.sort(remaining_indices)

    holdout_idx, _ = train_test_split(
        remaining_indices,
        train_size=holdout_size,
        random_state=random_seed,
        stratify=remaining_labels,
    )
    return np.sort(holdout_idx)
