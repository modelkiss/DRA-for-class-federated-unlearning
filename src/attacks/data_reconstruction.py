"""Diffusion-based reconstruction utilities for unlearning evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .label_inference import SensitiveFeature
from src.utils.normalization import denormalize, get_normalization_stats

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
    train_batch_size: int = 1
    max_train_steps: Optional[int] = None
    prior_blend_weight: float = 0.3
    noise_offset: float = 0.05


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
        target_dtype = config.dtype
        if config.device.type == "cpu":
            target_dtype = torch.float32
        self.pipeline.to(config.device, dtype=target_dtype)
        self.config.dtype = target_dtype
        self._base_guidance_scale = float(config.guidance_scale)
        self._guided_prompt_suffix: str = ""
        self._active_features: list[SensitiveFeature] = []
        self._guidance_hparams: dict[str, float] = {}
        self._prior_latent: torch.Tensor | None = None
        self._prior_latent_bank: list[torch.Tensor] = []
        self._latent_prior_template: torch.Tensor | None = None
        self._heatmap_mask: torch.Tensor | None = None
        self._latent_mask_cache: torch.Tensor | None = None
        self._trained_steps: int = 0

    def reset_guidance(self) -> None:
        """Reset any sensitive feature guidance applied to the pipeline."""

        self.config.guidance_scale = self._base_guidance_scale
        self._guided_prompt_suffix = ""
        self._active_features = []
        self._guidance_hparams: dict[str, float] = {}
        self._prior_latent = None
        self._prior_latent_bank = []
        self._latent_prior_template = None
        self._heatmap_mask = None
        self._latent_mask_cache = None
        self._trained_steps = 0

    def fine_tune_with_guidance(
        self,
        features: Sequence[SensitiveFeature],
        *,
        epochs: int = 1,
        learning_rate: float = 1e-4,
        images: torch.Tensor | None = None,
        class_label: str | None = None,
        batch_size: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> None:
        """Heuristically adapt prompts/guidance according to sensitive features."""

        if not features:
            self._active_features = []
            self._guided_prompt_suffix = ""
            self.config.guidance_scale = self._base_guidance_scale
            return

        self.config.guidance_scale = self._base_guidance_scale
        self._guided_prompt_suffix = ""
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
        self._guidance_hparams = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
        }

        if images is None or images.numel() == 0:
            return

        prompt_label = class_label or (self._active_features[0].name if self._active_features else "target")
        prompt = self.config.prompt_template.format(label=prompt_label)
        if self._guided_prompt_suffix:
            prompt = f"{prompt}, {self._guided_prompt_suffix}"

        batch_size = batch_size or self.config.train_batch_size
        max_steps = max_steps or self.config.max_train_steps

        steps = self._train_on_samples(
            images,
            prompt,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_steps=max_steps,
        )
        self._trained_steps += steps
        self._guidance_hparams["train_steps"] = float(self._trained_steps)

    def ingest_priors(self, samples: torch.Tensor, dataset: str) -> None:
        """将真实样本转换为潜空间均值，作为采样先验。"""

        _ = dataset  # 参数保留用于接口兼容
        if samples is None or samples.numel() == 0:
            self._prior_latent = None
            self._prior_latent_bank = []
            self._latent_prior_template = None
            self._latent_mask_cache = None
            return

        if samples.size(1) == 1:
            samples = samples.repeat(1, 3, 1, 1)

        try:
            stats = get_normalization_stats(dataset)
            samples = denormalize(samples, stats)
        except KeyError:
            pass

        try:
            to_device = samples.to(self.config.device, dtype=self.pipeline.unet.dtype).clamp(0.0, 1.0)
            latents = self.pipeline.vae.encode((to_device * 2.0) - 1.0).latent_dist.mean
            scale = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215)
            latents = (latents * scale).detach()
            self._prior_latent_bank = [latent.unsqueeze(0) for latent in latents]
            self._prior_latent = latents.mean(dim=0, keepdim=True)
            self._latent_prior_template = self._prior_latent.clone()
            self._latent_mask_cache = None
        except Exception:  # pragma: no cover - 依赖diffusers内部实现
            self._prior_latent = None
            self._prior_latent_bank = []
            self._latent_prior_template = None
            self._latent_mask_cache = None

    def set_heatmap_guidance(self, mask: torch.Tensor | None) -> None:
        """设置热力图掩模用于后处理强化。"""

        if mask is None or mask.numel() == 0:
            self._heatmap_mask = None
            self._latent_mask_cache = None
            return
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        self._heatmap_mask = mask.detach().to(self.config.device, dtype=torch.float32).clamp(0.0, 1.0)
        self._latent_mask_cache = None

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

        latents = self._prepare_initial_latents(num_samples)

        images = []
        callback = None
        if self._latent_prior_template is not None and self._heatmap_mask is not None:
            callback = self._make_latent_guidance_callback(num_samples)

        for _ in range(num_samples):
            result = self.pipeline(
                prompt,
                negative_prompt=self.config.negative_prompt,
                latents=None if latents is None else latents.clone(),
                callback=callback,
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _train_on_samples(
        self,
        images: torch.Tensor,
        prompt: str,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        max_steps: Optional[int],
    ) -> int:
        """执行一次基于遗忘样本的扩散模型微调。"""

        device = self.config.device
        dtype = self.pipeline.unet.dtype
        images = images.to(device=device, dtype=dtype)
        images = images.clamp(0.0, 1.0)

        try:
            from diffusers.models.attention_processor import LoRAAttnProcessor
        except ImportError:  # pragma: no cover - diffusers 版本差异
            LoRAAttnProcessor = None

        if LoRAAttnProcessor is not None:
            self._ensure_lora_layers(LoRAAttnProcessor)
            trainable_params = list(self._lora_layers.parameters()) if hasattr(self, "_lora_layers") else []
        else:
            trainable_params = list(self.pipeline.unet.parameters())

        for param in self.pipeline.unet.parameters():
            param.requires_grad = False

        for param in trainable_params:
            param.requires_grad = True

        if not trainable_params:
            trainable_params = []
            for module in self.pipeline.unet.modules():
                if hasattr(module, "weight"):
                    module.weight.requires_grad = True
                    trainable_params.append(module.weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.requires_grad = True
                    trainable_params.append(module.bias)

        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        total_steps = 0
        num_samples = images.size(0)
        scheduler = self.pipeline.scheduler
        scale = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215)

        self.pipeline.unet.train()
        self.pipeline.text_encoder.eval()

        for _ in range(epochs):
            indices = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                if max_steps is not None and total_steps >= max_steps:
                    break
                batch_indices = indices[start : start + batch_size]
                batch = images[batch_indices]
                latents = self.pipeline.vae.encode((batch * 2.0) - 1.0).latent_dist.sample()
                latents = latents * scale

                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.size(0),),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)

                text_embeddings = self._encode_prompt_embeddings(prompt, latents.size(0))

                noise_pred = self.pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample
                loss = self._weighted_mse(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_steps += 1

            if max_steps is not None and total_steps >= max_steps:
                break

        self.pipeline.unet.eval()
        for param in self.pipeline.unet.parameters():
            param.requires_grad = False

        return total_steps

    def _weighted_mse(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = (prediction - target) ** 2
        if self._heatmap_mask is None:
            return diff.mean()

        mask = self._latent_guidance_mask(diff.shape, prediction.device, prediction.dtype)
        if mask is None:
            return diff.mean()

        weighted = diff.mean(dim=1, keepdim=True) * (1.0 + mask)
        return weighted.mean()

    def _latent_guidance_mask(
        self,
        latent_shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self._heatmap_mask is None:
            return None
        if self._latent_mask_cache is not None:
            return self._latent_mask_cache.to(device=device, dtype=dtype)

        mask = self._heatmap_mask
        if mask.dim() == 3:
            mask = mask.mean(dim=0, keepdim=True)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, size=latent_shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.squeeze(0)
        mask = mask.mean(dim=0, keepdim=True)
        mask = mask.clamp(0.0, 1.0)
        self._latent_mask_cache = mask.detach().to(self.config.device)
        return mask.to(device=device, dtype=dtype)

    def _prepare_initial_latents(self, num_samples: int) -> torch.Tensor | None:
        if not self._prior_latent_bank:
            return None

        device = self.config.device
        dtype = self.config.dtype
        sigma = getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0)
        latents = []
        for _ in range(num_samples):
            base = self._prior_latent_bank[np.random.randint(0, len(self._prior_latent_bank))].to(device, dtype)
            base = base.clone()
            if self.config.noise_offset > 0:
                base = base + torch.randn_like(base) * self.config.noise_offset
            latents.append(base * sigma)
        return torch.cat(latents, dim=0)

    def _make_latent_guidance_callback(self, num_samples: int):
        template = self._latent_prior_template
        if template is None:
            return None

        template = template.to(self.config.device, dtype=self.config.dtype)
        mask = self._latent_guidance_mask(
            torch.Size([num_samples, template.size(1), template.size(2), template.size(3)]),
            self.config.device,
            self.config.dtype,
        )
        if mask is None:
            return None

        weight = float(self.config.prior_blend_weight)

        def _callback(step: int, timestep: int, latents: torch.Tensor) -> None:
            target = template.to(latents.device, dtype=latents.dtype)
            guide_mask = mask.to(latents.device, dtype=latents.dtype)
            latents.mul_(1.0 - guide_mask * weight)
            latents.add_(target * guide_mask * weight)

        return _callback

    def _ensure_lora_layers(self, lora_cls) -> None:
        if hasattr(self, "_lora_layers"):
            return

        unet = self.pipeline.unet
        if not hasattr(unet, "attn_processors"):
            self._lora_layers = torch.nn.ParameterList()
            return
        lora_attn_procs = {}
        for name in unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                cross_attention_dim = None
            else:
                cross_attention_dim = unet.config.cross_attention_dim

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[::-1][block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            else:
                hidden_size = unet.config.block_out_channels[0]

            lora_attn_procs[name] = lora_cls(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=4,
            )

        unet.set_attn_processor(lora_attn_procs)
        params = []
        for proc in lora_attn_procs.values():
            for param in proc.parameters():
                param.requires_grad = True
                params.append(param)
        self._lora_layers = torch.nn.ParameterList(params)

    def _encode_prompt_embeddings(self, prompt: str, batch_size: int) -> torch.Tensor:
        if hasattr(self.pipeline, "_encode_prompt"):
            return self.pipeline._encode_prompt(
                prompt,
                self.config.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=False,
            )

        tokenizer = self.pipeline.tokenizer
        text_inputs = tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.config.device)
        attention_mask = text_inputs.attention_mask.to(self.config.device) if "attention_mask" in text_inputs else None
        outputs = self.pipeline.text_encoder(text_input_ids, attention_mask=attention_mask)
        if isinstance(outputs, tuple):
            text_embeddings = outputs[0]
        else:
            text_embeddings = outputs.last_hidden_state
        return text_embeddings


__all__ = [
    "DiffusionConfig",
    "DiffusionReconstructor",
]