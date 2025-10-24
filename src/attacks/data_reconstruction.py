"""Data reconstruction utilities for the federated forgetting attack."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class ReconstructionConfig:
    """Gradient-based reconstruction hyper-parameters."""

    steps: int = 300
    lr: float = 0.1
    lambda_after: float = 1.0
    total_variation: float = 1e-4
    clip_range: tuple[float, float] = (0.0, 1.0)
    device: torch.device = torch.device("cpu")


class GradientReconstructor:
    """Reconstruct inputs that reactivate the forgotten class."""

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
                loss = (
                    -activation_before
                    + self.config.lambda_after * activation_after
                    + self.config.total_variation * tv_reg
                )
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


@dataclass
class DiffusionConfig:
    """Configuration for diffusion-based reconstruction."""

    model_id: str
    guidance_scale: float = 7.5
    num_inference_steps: int = 50
    height: Optional[int] = None
    width: Optional[int] = None
    prompt_template: str = "a photo of a {label}"
    negative_prompt: Optional[str] = None
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float16


class DiffusionReconstructor:
    """Sample reconstructions using a text-to-image diffusion pipeline."""

    def __init__(self, config: DiffusionConfig) -> None:
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Diffusion reconstruction requires the 'diffusers' package. Install it via 'pip install diffusers'."
            ) from exc

        self.config = config
        self.pipeline = StableDiffusionPipeline.from_pretrained(config.model_id)
        self.pipeline.to(config.device, dtype=config.dtype)

    def reconstruct(
        self,
        target_class: int,
        num_samples: int,
        *,
        class_label: str | None = None,
    ) -> torch.Tensor:
        label = class_label or str(target_class)
        prompt = self.config.prompt_template.format(label=label)
        kwargs = {
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
        }
        if self.config.height is not None:
            kwargs["height"] = self.config.height
        if self.config.width is not None:
            kwargs["width"] = self.config.width

        images = []
        for _ in range(num_samples):
            result = self.pipeline(
                prompt,
                negative_prompt=self.config.negative_prompt,
                **kwargs,
            )
            pil_image = result.images[0]
            image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            images.append(image)

        return torch.stack(images, dim=0)


__all__ = [
    "ReconstructionConfig",
    "GradientReconstructor",
    "DiffusionConfig",
    "DiffusionReconstructor",
]
