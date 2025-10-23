"""Label inference attack leveraging differences between models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.metrics import accuracy_dict, confusion_matrix, per_class_accuracy


@dataclass
class LabelInferenceResult:
    predicted_class: int
    score_vector: torch.Tensor
    per_class_before: torch.Tensor
    per_class_after: torch.Tensor
    confusion_before: torch.Tensor
    confusion_after: torch.Tensor
    ground_truth: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "predicted_class": int(self.predicted_class),
            "score_vector": self.score_vector.tolist(),
            "per_class_before": accuracy_dict(self.per_class_before),
            "per_class_after": accuracy_dict(self.per_class_after),
            "ground_truth": None if self.ground_truth is None else int(self.ground_truth),
        }


ScoreTransform = Callable[[torch.Tensor], torch.Tensor]


def infer_forgotten_label(
    before: nn.Module,
    after: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    transform: Optional[ScoreTransform] = None,
    ground_truth: Optional[int] = None,
) -> LabelInferenceResult:
    """Predict the removed class using per-class accuracy differences."""
    before = before.to(device)
    after = after.to(device)

    per_before = per_class_accuracy(before, dataloader, num_classes, device)
    per_after = per_class_accuracy(after, dataloader, num_classes, device)
    confusion_before = confusion_matrix(before, dataloader, num_classes, device)
    confusion_after = confusion_matrix(after, dataloader, num_classes, device)

    score_vector = per_before - per_after
    if transform is not None:
        score_vector = transform(score_vector)

    predicted = int(torch.argmax(score_vector).item())
    return LabelInferenceResult(
        predicted_class=predicted,
        score_vector=score_vector.detach().cpu(),
        per_class_before=per_before.detach().cpu(),
        per_class_after=per_after.detach().cpu(),
        confusion_before=confusion_before.detach().cpu(),
        confusion_after=confusion_after.detach().cpu(),
        ground_truth=ground_truth,
    )


__all__ = ["LabelInferenceResult", "infer_forgotten_label"]
