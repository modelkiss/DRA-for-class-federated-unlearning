"""Diffusion-based reconstruction utilities for unlearning evaluation.
LoRA is optional and OFF by default to maximize compatibility across diffusers versions.
Enable by setting DiffusionConfig.use_lora=True.
"""
from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import logging

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

    # LoRA settings
    use_lora: bool = False           # OFF by default
    lora_rank: int = 4               # typical small rank
    lora_alpha: int = 4              # scaling
    lora_target: str = "all"         # 'all' | 'cross' | 'self'


class DiffusionReconstructor:
    """Sample reconstructions using a text-to-image diffusion pipeline."""

    def __init__(self, config: DiffusionConfig) -> None:
        try:
            from diffusers import StableDiffusionPipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Diffusion reconstruction requires the 'diffusers' package. Install it via 'pip install diffusers'."
            ) from exc

        self.logger = logging.getLogger(__name__)
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
        """Generate `num_samples` images in a single batched call to avoid shape mismatches."""
        label = class_label or str(target_class)
        prompt = self.config.prompt_template.format(label=label)
        if self._guided_prompt_suffix:
            prompt = f"{prompt}, {self._guided_prompt_suffix}"

        # Prepare a list of identical prompts to match batch size
        prompts = [prompt] * int(num_samples)
        negative_prompts = None
        if self.config.negative_prompt is not None:
            negative_prompts = [self.config.negative_prompt] * int(num_samples)

        kwargs = {
            "guidance_scale": self.config.guidance_scale,
            "num_inference_steps": self.config.num_inference_steps,
        }
        if self.config.height is not None:
            kwargs["height"] = self.config.height
        if self.config.width is not None:
            kwargs["width"] = self.config.width

        # Pre-init latents for the full batch (B, 4, H//8, W//8) if available
        latents = self._prepare_initial_latents(num_samples)

        # Build optional callback (support both old and new API).
        cb_legacy = None
        cb_on_step_end = None
        if self._latent_prior_template is not None and self._heatmap_mask is not None:
            def _legacy_cb(step: int, timestep: int, latents_tensor: torch.Tensor) -> None:
                # In-place prior-guidance on latents
                template = self._latent_prior_template.to(latents_tensor.device, dtype=latents_tensor.dtype)
                mask = self._latent_guidance_mask(latents_tensor.shape, latents_tensor.device, latents_tensor.dtype)
                if mask is None:
                    return
                weight = float(self.config.prior_blend_weight)
                latents_tensor.mul_(1.0 - mask * weight)
                latents_tensor.add_(template * mask * weight)

            def _on_step_end(pipeline, step: int, timestep: int, callback_kwargs: dict) -> dict:
                # Same logic but via callback_on_step_end API (signature includes pipeline/self)
                lat = callback_kwargs.get("latents")
                if lat is None:
                    return callback_kwargs
                _legacy_cb(step, timestep, lat)
                callback_kwargs["latents"] = lat
                return callback_kwargs

            cb_legacy = _legacy_cb
            cb_on_step_end = _on_step_end

        # Prefer new API if available
        call_sig = inspect.signature(self.pipeline.__call__)
        use_new_cb = "callback_on_step_end" in call_sig.parameters

        # Single batched generation
        result = self.pipeline(
            prompts,
            negative_prompt=negative_prompts,
            latents=latents,
            callback_on_step_end=cb_on_step_end if use_new_cb else None,
            callback=cb_legacy if not use_new_cb else None,
            **kwargs,
        )

        # Convert resulting images to a torch batch (B, 3, H, W)
        pil_images = result.images
        imgs = []
        for img in pil_images:
            arr = np.array(img, copy=False)
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            imgs.append(t)
        batch = torch.stack(imgs, dim=0)

        if self._heatmap_mask is not None:
            # Optional post-blend on image space (very light-touch)
            mask = self._heatmap_mask
            if mask.dim() == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.to(batch.device, dtype=batch.dtype)
            mask = F.interpolate(mask.unsqueeze(0), size=batch.shape[-2:], mode="bilinear", align_corners=False)
            mask = mask.squeeze(0).expand(batch.size(0), -1, -1)
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

        trainable_params = None

        if self.config.use_lora:
            ok = self._setup_lora_layers()
            if ok and hasattr(self, "_lora_layers"):
                trainable_params = list(self._lora_layers.parameters())
            else:
                logging.getLogger(__name__).warning("LoRA setup failed; falling back to UNet fine-tuning.")
                trainable_params = list(self.pipeline.unet.parameters())
        else:
            trainable_params = list(self.pipeline.unet.parameters())

        # Freeze all, then unfreeze trainable
        for p in self.pipeline.unet.parameters():
            p.requires_grad = False
        for p in trainable_params:
            p.requires_grad = True

        # If nothing collected (shouldn't happen), unfreeze a minimal head
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
        for p in self.pipeline.unet.parameters():
            p.requires_grad = False

        return total_steps

    # ---------------------- LoRA setup helpers ----------------------

    def _setup_lora_layers(self) -> bool:
        """Try multiple strategies to enable LoRA. Return True if success, else False."""
        try:
            unet = self.pipeline.unet
            # Strategy 1: diffusers built-in (preferred)
            if hasattr(unet, "add_attn_procs"):
                try:
                    unet.add_attn_procs(self.config.lora_rank)
                    params = []
                    for name, proc in unet.attn_processors.items():
                        # Heuristic: LoRA processors contain 'lora' parameters
                        has_lora = any("lora" in n.lower() for n, _ in proc.named_parameters(recurse=True))
                        if has_lora:
                            for p in proc.parameters():
                                p.requires_grad = True
                                params.append(p)
                    if not params:
                        logging.getLogger(__name__).warning("add_attn_procs() returned no LoRA params; will try manual construction.")
                    else:
                        self._lora_layers = torch.nn.ParameterList(params)
                        logging.getLogger(__name__).info("LoRA via add_attn_procs: %d params", sum(p.numel() for p in params))
                        return True
                except Exception as e:
                    logging.getLogger(__name__).warning("unet.add_attn_procs failed: %s", e)

            # Strategy 2: manual processor construction (2.0 or classic)
            lora_cls = None
            try:
                from diffusers.models.attention_processor import LoRAAttnProcessor2_0 as _LORA  # type: ignore
                lora_cls = _LORA
            except Exception:
                try:
                    from diffusers.models.attention_processor import LoRAAttnProcessor as _LORA  # type: ignore
                    lora_cls = _LORA
                except Exception:
                    lora_cls = None

            if lora_cls is None:
                logging.getLogger(__name__).warning("No LoRA processor class available in this diffusers version.")
                return False

            if not hasattr(unet, "attn_processors"):
                logging.getLogger(__name__).warning("UNet has no attn_processors; cannot set LoRA processors.")
                return False

            lora_attn_procs = {}
            for name in unet.attn_processors.keys():
                # cross/self attention dim
                if name.endswith("attn1.processor"):
                    cross_attention_dim = None  # self-attn
                else:
                    cross_attention_dim = unet.config.cross_attention_dim

                # hidden size per block
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

                # try kwargs then positional
                proc = None
                try:
                    sig = inspect.signature(lora_cls)
                except Exception:
                    sig = None
                try:
                    if sig is not None:
                        kw = {"hidden_size": hidden_size, "cross_attention_dim": cross_attention_dim}
                        if "rank" in sig.parameters:
                            kw["rank"] = self.config.lora_rank
                        if "network_alpha" in sig.parameters:
                            kw["network_alpha"] = self.config.lora_alpha
                        proc = lora_cls(**kw)
                except Exception:
                    proc = None
                if proc is None:
                    try:
                        args = [hidden_size]
                        if cross_attention_dim is not None:
                            args.append(cross_attention_dim)
                        proc = lora_cls(*args)
                    except Exception as e:
                        logging.getLogger(__name__).warning("LoRA proc init failed for %s: %s", name, e)
                        continue

                lora_attn_procs[name] = proc

            if not lora_attn_procs:
                logging.getLogger(__name__).warning("No LoRA processors constructed.")
                return False

            unet.set_attn_processor(lora_attn_procs)
            params = []
            for proc in lora_attn_procs.values():
                for p in proc.parameters():
                    p.requires_grad = True
                    params.append(p)
            if not params:
                logging.getLogger(__name__).warning("Constructed LoRA processors have no parameters.")
                return False

            self._lora_layers = torch.nn.ParameterList(params)
            logging.getLogger(__name__).info("LoRA via manual processors: %d params", sum(p.numel() for p in params))
            return True

        except Exception as e:
            logging.getLogger(__name__).warning("LoRA setup error: %s", e)
            return False

    # ---------------------- loss & utils ----------------------

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
        # Use UNet dtype to avoid mismatch
        dtype = self.pipeline.unet.dtype
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

    # ---------------------- text enc ----------------------

    def _encode_prompt_embeddings(self, prompt: str, batch_size: int) -> torch.Tensor:
        def _unwrap_prompt_embeds(result: torch.Tensor | tuple[torch.Tensor | None, ...]) -> torch.Tensor:
            if isinstance(result, tuple):
                for item in result:
                    if isinstance(item, torch.Tensor):
                        return item
                raise TypeError("Prompt encoding returned only None values.")
            return result

        if hasattr(self.pipeline, "encode_prompt"):
            result = self.pipeline.encode_prompt(
                prompt,
                device=self.config.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=False,
            )
            return _unwrap_prompt_embeds(result)

        if hasattr(self.pipeline, "_encode_prompt"):
            result = self.pipeline._encode_prompt(
                prompt,
                self.config.device,
                num_images_per_prompt=batch_size,
                do_classifier_free_guidance=False,
            )
            return _unwrap_prompt_embeds(result)

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
