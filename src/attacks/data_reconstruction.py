"""Diffusion-based reconstruction utilities for unlearning evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

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
        self._prior_latent: torch.Tensor | None = None
        self._heatmap_mask: torch.Tensor | None = None

    def reset_guidance(self) -> None:
        """Reset any sensitive feature guidance applied to the pipeline."""

        self.config.guidance_scale = self._base_guidance_scale
        self._guided_prompt_suffix = ""
        self._active_features = []
        self._guidance_hparams: dict[str, float] = {}
        self._prior_latent = None
        self._heatmap_mask = None

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

    def ingest_priors(self, samples: torch.Tensor, dataset: str) -> None:
        """将真实样本转换为潜空间均值，作为采样先验。"""

        _ = dataset  # 参数保留用于接口兼容
        if samples is None or samples.numel() == 0:
            self._prior_latent = None
            return

        if samples.size(1) == 1:
            samples = samples.repeat(1, 3, 1, 1)

        try:
            to_device = samples.to(self.config.device, dtype=self.pipeline.unet.dtype).clamp(0.0, 1.0)
            latents = self.pipeline.vae.encode((to_device * 2.0) - 1.0).latent_dist.mean
            self._prior_latent = latents.mean(dim=0, keepdim=True).detach()
        except Exception:  # pragma: no cover - 依赖diffusers内部实现
            self._prior_latent = None

    def set_heatmap_guidance(self, mask: torch.Tensor | None) -> None:
        """设置热力图掩模用于后处理强化。"""

        if mask is None or mask.numel() == 0:
            self._heatmap_mask = None
            return
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        self._heatmap_mask = mask.detach().to(self.config.device, dtype=torch.float32).clamp(0.0, 1.0)

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

        latents = None
        if self._prior_latent is not None:
            latents = self._prior_latent.repeat(num_samples, 1, 1, 1)

        images = []
        for _ in range(num_samples):
            result = self.pipeline(
                prompt,
                negative_prompt=self.config.negative_prompt,
                latents=None if latents is None else latents.clone(),
                **kwargs,
            )
            pil_image = result.images[0]
            image = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
            images.append(image)

        batch = torch.stack(images, dim=0)

        if self._heatmap_mask is not None:
            mask = self._heatmap_mask
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.to(batch.device, dtype=batch.dtype)
            mask = F.interpolate(mask.unsqueeze(0), size=batch.shape[-2:], mode="bilinear", align_corners=False)
            mask = mask.squeeze(0)
            mask = mask.expand(batch.size(0), -1, -1)
            batch = batch * (0.5 + 0.5 * mask.unsqueeze(1)) + batch * (1 - mask.unsqueeze(1))

        return batch


__all__ = [
    "DiffusionConfig",
    "DiffusionReconstructor",
]