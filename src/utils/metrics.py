"""Metrics helpers used across experiments."""
from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict

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


def average_parameter_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    max_batches: int | None = None,
    predicate: Callable[[str, nn.Parameter], bool] | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Estimate average gradients across ``dataloader`` batches.

    Parameters
    ----------
    model:
        Model whose gradients are to be estimated.
    dataloader:
        Data loader providing (input, target) pairs.
    device:
        Torch device.
    max_batches:
        Optional cap on the number of batches processed.
    predicate:
        Optional callable filtering parameters by name.
    """

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and (predicate is None or predicate(name, param))
    ]

    if not named_params:
        return OrderedDict()

    accumulators: "OrderedDict[str, torch.Tensor]" = OrderedDict(
        (name, torch.zeros_like(param, device=device)) for name, param in named_params
    )

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    for batch_index, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        grads = torch.autograd.grad(loss, [param for _, param in named_params], retain_graph=False)
        for (name, _), grad in zip(named_params, grads):
            accumulators[name] += grad.detach()
        total += 1
        if max_batches is not None and batch_index + 1 >= max_batches:
            break

    if total == 0:
        return OrderedDict((name, tensor.detach().cpu()) for name, tensor in accumulators.items())

    return OrderedDict((name, (tensor / total).detach().cpu()) for name, tensor in accumulators.items())


def gradient_delta_norms(gradient_before: Dict[str, torch.Tensor], gradient_after: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute L2 norms of gradient differences for matched parameters."""

    norms: Dict[str, float] = {}
    for name, before in gradient_before.items():
        if name not in gradient_after:
            continue
        diff = before - gradient_after[name]
        norms[name] = float(diff.norm().item())
    return norms


__all__ = [
    "accuracy",
    "per_class_accuracy",
    "confusion_matrix",
    "accuracy_dict",
    "average_parameter_gradients",
    "gradient_delta_norms",
]
