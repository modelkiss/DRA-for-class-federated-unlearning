"""Diffusion-based reconstruction utilities for unlearning evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


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
    "DiffusionConfig",
    "DiffusionReconstructor",
]