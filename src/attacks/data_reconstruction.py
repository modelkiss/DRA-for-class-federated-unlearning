"""Diffusion-based reconstruction utilities for unlearning evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from .label_inference import SensitiveFeature

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
        self._base_guidance_scale = float(config.guidance_scale)
        self._guided_prompt_suffix: str = ""
        self._active_features: list[SensitiveFeature] = []
        self._guidance_hparams: dict[str, float] = {}

    def reset_guidance(self) -> None:
        """Reset any sensitive feature guidance applied to the pipeline."""

        self.config.guidance_scale = self._base_guidance_scale
        self._guided_prompt_suffix = ""
        self._active_features = []
        self._guidance_hparams: dict[str, float] = {}

    def fine_tune_with_guidance(
            self,
            features: Sequence[SensitiveFeature],
            *,
            epochs: int = 1,
            learning_rate: float = 1e-4,
    ) -> None:
        """Heuristically adapt prompts/guidance according to sensitive features."""

        self.reset_guidance()
        if not features:
            return

        self._active_features = list(features)
        keywords = []
        scores = []
        for feature in features:
            token = feature.name.replace("_", " ")
            keywords.append(token)
            scores.append(abs(float(feature.score)))

        if keywords:
            unique_keywords = []
            seen = set()
            for keyword in keywords:
                if keyword in seen:
                    continue
                seen.add(keyword)
                unique_keywords.append(keyword)
            self._guided_prompt_suffix = ", ".join(unique_keywords)

        if scores:
            mean_score = float(np.mean(scores))
            scale_delta = 0.1 * np.tanh(mean_score * max(1, epochs))
            self.config.guidance_scale = float(self._base_guidance_scale * (1.0 + scale_delta))

        # Store guidance metadata for potential reproducibility.
        self._guidance_hparams = {"epochs": epochs, "learning_rate": learning_rate,
                                  "mean_score": float(np.mean(scores)) if scores else 0.0}

    def reconstruct(
        self,
        target_class: int,
        num_samples: int,
        *,
        class_label: str | None = None,
    ) -> torch.Tensor:
        label = class_label or str(target_class)
        prompt = self.config.prompt_template.format(label=label)
        if self._guided_prompt_suffix:
            prompt = f"{prompt}, {self._guided_prompt_suffix}"
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