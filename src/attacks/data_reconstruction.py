"""Data reconstruction utilities for the federated forgetting attack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class ReconstructionConfig:
    steps: int = 300
    lr: float = 0.1
    lambda_after: float = 1.0
    total_variation: float = 1e-4
    clip_range: tuple[float, float] = (0.0, 1.0)
    device: torch.device = torch.device("cpu")


class GradientReconstructor:
    """Reconstruct inputs that activate the forgotten class."""

    def __init__(self, config: ReconstructionConfig) -> None:
        self.config = config

    def reconstruct(
        self,
        model_before: nn.Module,
        model_after: nn.Module,
        target_class: int,
        num_samples: int,
        input_shape: Sequence[int],
    ) -> torch.Tensor:
        device = self.config.device
        model_before = model_before.to(device).eval()
        model_after = model_after.to(device).eval()
        outputs = []
        for _ in range(num_samples):
            image = torch.randn(1, *input_shape, device=device, requires_grad=True)
            optimizer = torch.optim.Adam([image], lr=self.config.lr)
            for _ in range(self.config.steps):
                optimizer.zero_grad()
                logits_before = model_before(image)
                logits_after = model_after(image)
                activation_before = logits_before[:, target_class].mean()
                activation_after = logits_after[:, target_class].mean()
                tv_reg = self._total_variation(image)
                loss = -activation_before + self.config.lambda_after * activation_after + self.config.total_variation * tv_reg
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    image.clamp_(*self.config.clip_range)
            outputs.append(image.detach().cpu())
        return torch.cat(outputs, dim=0)

    @staticmethod
    def _total_variation(tensor: torch.Tensor) -> torch.Tensor:
        diff_h = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
        diff_v = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
        return diff_h.abs().mean() + diff_v.abs().mean()


__all__ = ["ReconstructionConfig", "GradientReconstructor"]