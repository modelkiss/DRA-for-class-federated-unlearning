"""Metrics helpers used across experiments."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    """Compute overall accuracy for ``model`` on ``dataloader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.numel()
    return correct / max(total, 1)


def per_class_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return per-class accuracies.

    Parameters
    ----------
    model: nn.Module
        Model under evaluation.
    dataloader: DataLoader
        Evaluation data loader.
    num_classes: int
        Total number of classes.
    device: torch.device
        Torch device used for evaluation.
    """
    model.eval()
    correct = torch.zeros(num_classes, device=device)
    total = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            for cls in range(num_classes):
                mask = targets == cls
                total[cls] += mask.sum()
                correct[cls] += (predictions[mask] == cls).sum()
    return torch.where(total > 0, correct / total.clamp(min=1), torch.zeros_like(total))


def confusion_matrix(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute a confusion matrix."""
    matrix = torch.zeros((num_classes, num_classes), device=device)
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)
            for true, pred in zip(targets, predictions):
                matrix[true, pred] += 1
    return matrix


def accuracy_dict(per_class: torch.Tensor) -> Dict[int, float]:
    """Return a dictionary keyed by class index with accuracy values."""
    return {index: per_class[index].item() for index in range(len(per_class))}


__all__ = ["accuracy", "per_class_accuracy", "confusion_matrix", "accuracy_dict"]
