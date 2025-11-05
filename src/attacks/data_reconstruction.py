"""Diffusion-based reconstruction utilities with contrastive guidance."""
from __future__ import annotations

import logging
from dataclasses import dataclass
import inspect
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .label_inference import SensitiveFeature
from src.utils.normalization import denormalize, get_normalization_stats


@dataclass
class DiffusionConfig:
    """Configuration for diffusion-based reconstruction."""

    model_id: str
    guidance_scale: float = 0.0
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
    strength: float = 0.7
    contrastive_step: float = 0.12
    contrastive_min_step: float = 0.02
    contrastive_frequency: int = 2
    contrastive_clip: float = 1.5
    contrastive_ema: float = 0.85

    # LoRA settings
    use_lora: bool = False           # OFF by default
    lora_rank: int = 4               # typical small rank
    lora_alpha: int = 4              # scaling
    lora_target: str = "all"         # 'all' | 'cross' | 'self'


class DiffusionReconstructor:
    """Sample reconstructions using a text-to-image diffusion pipeline."""

    def __init__(self, config: DiffusionConfig) -> None:
        try:
            from diffusers import StableDiffusionImg2ImgPipeline  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Diffusion reconstruction requires the 'diffusers' package. Install it via 'pip install diffusers'."
            ) from exc

        self.logger = logging.getLogger(__name__)
        self.config = config
        self.pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(config.model_id)
        target_dtype = config.dtype
        if config.device.type == "cpu":
            target_dtype = torch.float32
        self.pipeline.to(config.device, dtype=target_dtype)
        self.config.dtype = target_dtype
        self._base_guidance_scale = 0.0
        self.config.guidance_scale = 0.0
        self._guided_prompt_suffix: str = ""
        self._active_features: list[SensitiveFeature] = []
        self._guidance_hparams: dict[str, float] = {}
        self._heatmap_mask: torch.Tensor | None = None
        self._latent_mask_cache: torch.Tensor | None = None
        self._trained_steps: int = 0
        self._prior_image_bank: list[torch.Tensor] = []
        self._ema_latent_update: torch.Tensor | None = None

    def reset_guidance(self) -> None:
        """Reset any sensitive feature guidance applied to the pipeline."""

        self.config.guidance_scale = self._base_guidance_scale
        self._guided_prompt_suffix = ""
        self._active_features = []
        self._guidance_hparams: dict[str, float] = {}
        self._heatmap_mask = None
        self._latent_mask_cache = None
        self._trained_steps = 0
        self._prior_image_bank = []
        self._ema_latent_update = None

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
        """Record sensitive features without applying textual guidance."""

        self._active_features = list(features)
        self._guided_prompt_suffix = ""
        self.config.guidance_scale = 0.0

        mean_score = 0.0
        if features:
            scores = [abs(float(feature.score)) for feature in features]
            mean_score = float(np.mean(scores)) if scores else 0.0

        self._guidance_hparams = {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "mean_score": mean_score,
            "train_steps": 0.0,
        }
        self._trained_steps = 0
        self._ema_latent_update = None

        if images is not None and images.numel() > 0:
            clipped = images.detach().cpu().clamp(0.0, 1.0)
            self._prior_image_bank = [tensor.clone() for tensor in clipped]
            self._latent_mask_cache = None
        self.logger.info("文本提示已禁用：跳过扩散模型的提示语微调，仅保留空间掩模与梯度细化。")

    def ingest_priors(self, samples: torch.Tensor, dataset: str) -> None:
        """将真实样本转换为潜空间均值，作为采样先验。"""

        _ = dataset  # 参数保留用于接口兼容
        if samples is None or samples.numel() == 0:
            self._prior_image_bank = []
            self._latent_mask_cache = None
            return

        if samples.size(1) == 1:
            samples = samples.repeat(1, 3, 1, 1)

        try:
            stats = get_normalization_stats(dataset)
            samples = denormalize(samples, stats)
        except KeyError:
            pass

        samples = samples.detach().cpu().clamp(0.0, 1.0)
        self._prior_image_bank = [tensor.clone() for tensor in samples]
        self._latent_mask_cache = None

    def set_heatmap_guidance(self, mask: torch.Tensor | None) -> None:
        """设置热力图掩模用于后处理强化。"""

        if mask is None or mask.numel() == 0:
            self._heatmap_mask = None
            self._latent_mask_cache = None
            self._ema_latent_update = None
            return
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        self._heatmap_mask = mask.detach().to(self.config.device, dtype=torch.float32).clamp(0.0, 1.0)
        self._latent_mask_cache = None
        self._ema_latent_update = None

    def reconstruct(
        self,
        target_class: int,
        num_samples: int,
        *,
        class_label: str | None = None,
        init_images: torch.Tensor | None = None,
        classifier_before: torch.nn.Module | None = None,
        classifier_after: torch.nn.Module | None = None,
        dataset: str | None = None,
        guidance_frequency: int | None = None,
    ) -> torch.Tensor:
        """Generate reconstructions with optional contrastive classifier guidance."""

        prompts = [""] * int(num_samples)

        images = self._prepare_initial_images(init_images, num_samples)

        kwargs = {
            "guidance_scale": 0.0,
            "num_inference_steps": self.config.num_inference_steps,
            "strength": self.config.strength,
            "callback_steps": 1,
        }

        call_sig = inspect.signature(self.pipeline.__call__)
        if "image" not in call_sig.parameters:
            raise RuntimeError("当前扩散模型不支持图生图模式，请切换 diffusers pipeline。")

        pipeline_prompt: list[str] | str = prompts
        if num_samples == 1:
            pipeline_prompt = prompts[0]

        if classifier_before is not None and classifier_after is not None and dataset is not None:
            kwargs["callback"] = self._build_contrastive_callback(
                classifier_before,
                classifier_after,
                target_class=target_class,
                dataset=dataset,
                guidance_frequency=guidance_frequency,
            )

        result = self.pipeline(
            pipeline_prompt,
            image=images,
            output_type="pt",
            **kwargs,
        )

        if not hasattr(result, "images"):
            raise RuntimeError("扩散模型调用未返回图像。")

        batch = result.images
        if isinstance(batch, list):
            imgs: list[torch.Tensor] = []
            for img in batch:
                if isinstance(img, torch.Tensor):
                    tensor = img
                else:
                    arr = np.array(img, copy=False)
                    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
                imgs.append(tensor)
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
    # Contrastive classifier guidance
    # ------------------------------------------------------------------

    def _build_contrastive_callback(
        self,
        classifier_before: torch.nn.Module,
        classifier_after: torch.nn.Module,
        *,
        target_class: int,
        dataset: str,
        guidance_frequency: int | None,
    ):
        frequency = guidance_frequency if guidance_frequency is not None else self.config.contrastive_frequency
        frequency = max(1, int(frequency))

        scale = getattr(self.pipeline.vae.config, "scaling_factor", 0.18215)
        stats = get_normalization_stats(dataset)

        def _normalize_for_classifier(images: torch.Tensor) -> torch.Tensor:
            mean = torch.tensor(stats[0], device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
            std = torch.tensor(stats[1], device=images.device, dtype=images.dtype).view(1, -1, 1, 1)
            return (images - mean) / std

        def _decode_latents(latents: torch.Tensor) -> torch.Tensor:
            decoded = self.pipeline.vae.decode(latents / scale).sample
            return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        def _step_size(step: int, total: int) -> float:
            if total <= 1:
                return float(self.config.contrastive_min_step)
            ratio = step / float(total - 1)
            max_step = float(self.config.contrastive_step)
            min_step = float(self.config.contrastive_min_step)
            return float((1.0 - ratio) * max_step + ratio * min_step)

        def _apply_clip(update: torch.Tensor) -> torch.Tensor:
            clip_norm = float(max(0.0, self.config.contrastive_clip))
            if clip_norm == 0.0:
                return update
            flat = update.view(update.size(0), -1)
            norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-6)
            factors = (clip_norm / norms).clamp(max=1.0)
            return update * factors.view(-1, 1, 1, 1)

        def callback(step: int, timestep: int, latents: torch.Tensor) -> None:
            if step % frequency != 0:
                return
            if not latents.requires_grad:
                latents.requires_grad_(True)
            with torch.enable_grad():
                decoded = _decode_latents(latents)
                if decoded.dtype != torch.float32:
                    decoded = decoded.to(torch.float32)
                normed = _normalize_for_classifier(decoded)
                logits_before = classifier_before(normed)
                logits_after = classifier_after(normed)
                contrast = (logits_before[:, target_class] - logits_after[:, target_class]).sum()
                grad = torch.autograd.grad(contrast, latents)[0]

            mask = self._latent_guidance_mask(latents.shape, latents.device, latents.dtype)
            if mask is not None:
                grad = grad * (1.0 + mask)

            grad = grad / grad.view(grad.size(0), -1).norm(dim=1, keepdim=True).clamp(min=1e-6).view(-1, 1, 1, 1)
            grad = _apply_clip(grad)

            ema = float(max(0.0, min(0.999, self.config.contrastive_ema)))
            if ema > 0.0:
                if self._ema_latent_update is None or self._ema_latent_update.shape != grad.shape:
                    self._ema_latent_update = grad.detach()
                else:
                    self._ema_latent_update = (
                        ema * self._ema_latent_update + (1.0 - ema) * grad.detach()
                    )
                grad = self._ema_latent_update

            step_size = _step_size(step, int(self.config.num_inference_steps))
            update = grad * step_size

            with torch.no_grad():
                latents.add_(update)

        return callback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_initial_images(
        self,
        init_images: torch.Tensor | None,
        num_samples: int,
    ) -> list[Image.Image]:
        """Prepare initial images for img2img sampling, applying mask-based corruption."""

        if init_images is not None and init_images.numel() > 0:
            base = init_images.detach().clone()
        elif self._prior_image_bank:
            indices = [np.random.randint(0, len(self._prior_image_bank)) for _ in range(max(1, num_samples))]
            base = torch.stack([self._prior_image_bank[idx] for idx in indices], dim=0)
        else:
            # Fallback to random noise if no priors are available
            default_size = getattr(self.pipeline.unet.config, "sample_size", 64) * 8
            height = self.config.height or default_size
            width = self.config.width or default_size
            base = torch.rand(num_samples, 3, height, width)

        if base.dim() != 4:
            raise ValueError("初始图像张量形状必须为 (B,C,H,W)。")

        base = base.clamp(0.0, 1.0)
        if base.size(1) == 1:
            base = base.repeat(1, 3, 1, 1)

        if base.size(0) < num_samples:
            repeats = []
            for index in range(num_samples):
                repeats.append(base[index % base.size(0)])
            base = torch.stack(repeats, dim=0)
        elif base.size(0) > num_samples:
            base = base[:num_samples]

        mask = self._heatmap_mask
        if mask is not None:
            mask_tensor = mask.detach()
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.dim() == 3 and mask_tensor.size(0) != base.size(0):
                mask_tensor = mask_tensor.mean(dim=0, keepdim=True)
            mask_tensor = mask_tensor.to(base.device, dtype=base.dtype)
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=base.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            if mask_tensor.size(0) != base.size(0):
                mask_tensor = mask_tensor.expand(base.size(0), -1, -1)
            mask_tensor = mask_tensor.clamp(0.0, 1.0)
            complement = 1.0 - mask_tensor
            noise_strength = float(min(1.0, 0.3 + max(0.0, self.config.noise_offset)))
            noise = torch.rand_like(base)
            base = base * mask_tensor.unsqueeze(1) + (
                noise_strength * noise + (1.0 - noise_strength) * base
            ) * complement.unsqueeze(1)
            base = base.clamp(0.0, 1.0)

        target_height = self.config.height
        target_width = self.config.width
        if target_height is None or target_width is None:
            default_size = getattr(self.pipeline.unet.config, "sample_size", 64) * 8
            target_height = target_height or default_size
            target_width = target_width or default_size

        images: list[Image.Image] = []
        for tensor in base:
            tensor = tensor.clamp(0.0, 1.0)
            array = (tensor.permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            if array.shape[2] == 1:
                array = np.repeat(array, 3, axis=2)
            pil_image = Image.fromarray(array)
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            if (pil_image.height, pil_image.width) != (target_height, target_width):
                pil_image = pil_image.resize((target_width, target_height), Image.BILINEAR)
            images.append(pil_image)
        return images

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
