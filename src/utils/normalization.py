"""Dataset normalization utilities shared across scripts."""
from __future__ import annotations

from typing import Tuple

import torch

# Normalization statistics mirror the preprocessing used during training.
NORMALIZATION_STATS: dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "mnist": ((0.1307,), (0.3081,)),
    "fashionmnist": ((0.1307,), (0.3081,)),
}


def get_normalization_stats(dataset: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Return the mean/std tuple associated with ``dataset``.

    Parameters
    ----------
    dataset:
        Dataset identifier (case-insensitive).

    Raises
    ------
    KeyError
        If ``dataset`` is not registered.
    """

    key = dataset.lower()
    if key not in NORMALIZATION_STATS:
        raise KeyError(f"Unknown dataset '{dataset}'. Available keys: {sorted(NORMALIZATION_STATS)}")
    return NORMALIZATION_STATS[key]


def denormalize(tensor: torch.Tensor, stats: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> torch.Tensor:
    """Undo normalization for a batch of images.

    Parameters
    ----------
    tensor:
        Tensor shaped ``(N, C, H, W)`` containing normalized images.
    stats:
        ``(mean, std)`` tuple matching the dataset preprocessing.
    """

    mean, std = stats
    mean_tensor = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, -1, 1, 1)
    return tensor * std_tensor + mean_tensor


__all__ = ["NORMALIZATION_STATS", "get_normalization_stats", "denormalize"]