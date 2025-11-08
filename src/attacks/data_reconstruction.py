"""Adaptive diffusion-based data reconstruction pipeline.

This module stitches together three stages to reconstruct forgotten data:

1. ``BaseDiffusionTrainer`` performs a lightweight adaptation of a Stable
   Diffusion model (SD 1.5 by default) on samples that match the inferred
   forgotten label.  The implementation uses LoRA as the default adapter but
   keeps the API generic in case textual inversion or other adapters are
   desired later.
2. ``SensitiveFeatureFinetuner`` consumes the assets produced by
   :func:`src.attacks.sensitive_feature_inference.run_sensitive_feature_inference`
   (region masks, edge maps, texture tokens, ControlNet hints) to refine the
   base diffusion weights so that reconstructed images emphasise sensitive
   regions.
3. ``AdaptiveGenerationController`` repeatedly generates image batches and
   evaluates them with the forgotten model snapshots.  Batches that match the
   target forgetting behaviour are exported while the controller dynamically
   adjusts guidance / ControlNet scales if the criterion is not met.

The pipeline is intentionally conservative – it prioritises robustness and
transparent logging over raw training throughput.  Each stage can be skipped
through configuration should users only require subsets of the functionality.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
try:  # pragma: no cover - AMP is optional on CPU-only builds
    from torch.amp import GradScaler as _AmpGradScaler, autocast as _amp_autocast
except (ImportError, AttributeError):  # pragma: no cover
    _AmpGradScaler = None  # type: ignore
    _amp_autocast = None  # type: ignore

try:  # pragma: no cover - CUDA AMP may be unavailable
    from torch.cuda.amp import GradScaler as _CudaGradScaler, autocast as _cuda_autocast
except (ImportError, AttributeError):  # pragma: no cover
    _CudaGradScaler = None  # type: ignore
    _cuda_autocast = None  # type: ignore

autocast = _amp_autocast or _cuda_autocast  # type: ignore[assignment]
if _AmpGradScaler is not None:
    GradScaler = _AmpGradScaler  # type: ignore[assignment]
    _GRAD_SCALER_REQUIRES_DEVICE_ARG = True
elif _CudaGradScaler is not None:
    GradScaler = _CudaGradScaler  # type: ignore[assignment]
    _GRAD_SCALER_REQUIRES_DEVICE_ARG = False
else:
    GradScaler = None  # type: ignore
    _GRAD_SCALER_REQUIRES_DEVICE_ARG = False


class _NoOpGradScaler:
    """Fallback GradScaler that implements the minimal API we rely on."""

    def __init__(self, enabled: bool = False):
        self._enabled = False

    def is_enabled(self) -> bool:
        return False

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):  # pragma: no cover - no-op
        return None

    def step(self, optimizer):  # pragma: no cover - fallback step
        optimizer.step()

    def update(self):  # pragma: no cover - no-op
        return None


def _make_grad_scaler(enabled: bool, device_type: str | None):
    if not enabled or GradScaler is None:
        return _NoOpGradScaler()
    if _GRAD_SCALER_REQUIRES_DEVICE_ARG:
        if device_type is None:
            device_type = "cuda"
        try:
            return GradScaler(device_type, enabled=enabled)
        except TypeError:  # pragma: no cover - legacy torch.amp signature
            return GradScaler(device_type)
    return GradScaler(enabled=enabled)


def _autocast_context(device_type: str, dtype: torch.dtype):
    if autocast is None:
        return nullcontext()
    try:
        return autocast(device_type=device_type, dtype=dtype)
    except TypeError:  # pragma: no cover - torch.cuda.amp.autocast signature
        return autocast(dtype=dtype)

from .label_inference import LabelInferenceResult
from ..utils.metrics import accuracy
from ..utils.normalization import denormalize

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BaseDiffusionTrainingConfig:
    """Configuration for the initial diffusion fine-tuning stage."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    method: str = "lora"
    resolution: int = 512
    batch_size: int = 2
    gradient_accumulation_steps: int = 1
    max_train_steps: int = 1500
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    lora_rank: int = 8
    lora_alpha: float = 32.0
    prior_loss_weight: float = 1.0
    mixed_precision: str = "fp16"
    cache_latents: bool = True
    allow_tf32: bool = True
    tokenizer_truncation: bool = True
    prompt_prefix: str = "photo of"
    sample_limit: int | None = None
    early_stop_loss_threshold: float | None = 0.05


@dataclass
class SensitiveFeatureFinetuneConfig:
    """Configuration for the sensitive feature refinement stage."""

    enabled: bool = True
    controlnet_model_id: str = "lllyasviel/sd-controlnet-canny"
    learning_rate: float = 5e-5
    max_train_steps: int = 800
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    condition_scale: float = 1.0
    canny_condition_weight: float = 1.0
    edge_condition_weight: float = 0.5
    region_loss_weight: float = 1.0
    edge_loss_weight: float = 1.0
    lpips_weight: float = 0.5
    mixed_precision: str = "fp16"
    guidance_scale: float = 7.5
    num_inference_steps: int = 30
    prompt_suffix: str = "highly detailed"
    max_condition_images: int = 8


@dataclass
class AdaptiveGenerationConfig:
    """Configuration for adaptive batch generation and evaluation."""

    images_per_batch: int = 1024
    max_batches: int = 10
    batch_size: int = 32
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_scale: float = 1.0
    target_accuracy_min: float = 0
    target_accuracy_max: float = 0.6
    accuracy_margin: float = 0.05
    adjustment_step: float = 0.2
    step_increment: int = 5
    seed: int = 2024
    save_rejected: bool = False


@dataclass
class DataReconstructionConfig:
    """Combined configuration passed to :func:`run_data_reconstruction`."""

    base: BaseDiffusionTrainingConfig = field(default_factory=BaseDiffusionTrainingConfig)
    sensitive: SensitiveFeatureFinetuneConfig = field(default_factory=SensitiveFeatureFinetuneConfig)
    adaptive: AdaptiveGenerationConfig = field(default_factory=AdaptiveGenerationConfig)


@dataclass
class AdaptiveBatchRecord:
    """Metadata describing a generated batch evaluation."""

    batch_index: int
    before_accuracy: float
    after_accuracy: float
    guidance_scale: float
    controlnet_scale: float
    inference_steps: int
    accepted: bool
    output_dir: str | None


@dataclass
class DataReconstructionResult:
    """Aggregate result returned by :func:`run_data_reconstruction`."""

    predicted_class: int
    output_dir: str
    accepted_batches: list[AdaptiveBatchRecord]
    rejected_batches: list[AdaptiveBatchRecord]
    config: dict[str, object]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _tensor_to_pil(tensor: torch.Tensor) -> "Image.Image":
    from PIL import Image

    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.size(0) == 1:
        tensor = tensor.repeat(3, 1, 1)
    array = tensor.permute(1, 2, 0).clamp(0.0, 1.0).numpy()
    return Image.fromarray((array * 255.0).astype(np.uint8))


def _load_grayscale(path: Path, resolution: int) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("L")
    image = image.resize((resolution, resolution), Image.BICUBIC)
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0)
    return tensor


def _gather_label_samples(
    dataloader: DataLoader,
    label: int,
    *,
    normalization_stats,
    limit: int | None = None,
) -> list["Image.Image"]:
    """Collect denormalised PIL samples for ``label`` from ``dataloader``."""

    collected: list["Image.Image"] = []
    for inputs, targets in dataloader:
        mask = targets == label
        if not torch.any(mask):
            continue
        subset = inputs[mask]
        subset = denormalize(subset, normalization_stats).clamp(0.0, 1.0)
        for tensor in subset:
            collected.append(_tensor_to_pil(tensor))
            if limit is not None and len(collected) >= limit:
                return collected
    return collected


class _PromptedImageDataset(Dataset):
    """Dataset that pairs images with prompts for diffusion fine-tuning."""

    def __init__(self, images: Sequence["Image.Image"], prompts: Sequence[str], tokenizer, resolution: int) -> None:
        if not images:
            raise ValueError("At least one training image is required for diffusion fine-tuning")
        if len(prompts) == 1:
            prompts = list(prompts) * len(images)
        if len(images) != len(prompts):
            raise ValueError("Number of images and prompts must match")

        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.images = list(images)
        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.images[index]
        if image.mode != "RGB":
            image = image.convert("RGB")
        pixel_values = self.transform(image)
        prompt = self.prompts[index]
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
        }


class _ControlNetDataset(Dataset):
    """Dataset combining images, prompts and ControlNet conditioning maps."""

    def __init__(
        self,
        *,
        base_image: "Image.Image",
        control_images: list[torch.Tensor],
        prompts: Sequence[str],
        tokenizer,
        resolution: int,
    ) -> None:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.prompts = list(prompts)
        self.tokenizer = tokenizer
        self.control = [tensor for tensor in control_images if tensor.numel() > 0]
        if not self.control:
            raise ValueError("ControlNet dataset requires at least one conditioning map")

        if base_image.mode != "RGB":
            base_image = base_image.convert("RGB")
        self.image = base_image
        self.transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        prompt = self.prompts[index % len(self.prompts)]
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        control = self.control[index % len(self.control)]
        if control.dim() == 2:
            control = control.unsqueeze(0)
        if control.size(0) == 1:
            control = control.repeat(3, 1, 1)
        return {
            "pixel_values": self.transform(self.image),
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "control": control,
        }


class _GeneratedDataset(Dataset):
    """Dataset view over generated PIL images for classifier evaluation."""

    def __init__(self, images: Sequence["Image.Image"], *, normalization_stats, label: int) -> None:
        from torchvision import transforms

        self.images = list(images)
        self.label = int(label)
        mean, std = normalization_stats
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image)
        return tensor, self.label


# ---------------------------------------------------------------------------
# Diffusion training primitives
# ---------------------------------------------------------------------------


def _prepare_lora_layers(unet: nn.Module, rank: int, alpha: float) -> nn.Module:
    try:
        from diffusers.models.attention_processor import LoRAAttnProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Diffusers with LoRA support is required. Install diffusers>=0.20.0"
        ) from exc

    try:  # diffusers>=0.25.0 provides AttnProcsLayers, but older versions do not
        from diffusers.models.attention_processor import AttnProcsLayers  # type: ignore
    except ImportError:  # pragma: no cover - gracefully handle older versions
        AttnProcsLayers = None

    lora_attn_procs: dict[str, nn.Module] = {}

    if hasattr(unet, "attn_processors") and unet.attn_processors:
        # diffusers>=0.15 exposes attention processors directly through the UNet
        # configuration, which allows us to construct LoRA adapters without relying on
        # brittle module name inspection. The logic is adapted from the official
        # LoRA training example in the diffusers repository.
        for name in unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                cross_attention_dim = None
            else:
                cross_attention_dim = getattr(unet.config, "cross_attention_dim", None)

            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = list(unet.config.block_out_channels[::-1])[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name.split(".")[1])
                hidden_size = unet.config.block_out_channels[block_id]
            else:  # pragma: no cover - defensive branch for future architectures
                continue

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
                network_alpha=alpha,
            )
    else:
        for name, module in unet.named_modules():
            if module.__class__.__name__ != "CrossAttention":
                continue
            hidden_size = module.to_q.in_features
            cross_attention_dim = (
                module.context_dim if hasattr(module, "context_dim") else module.to_k.in_features
            )
            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=rank,
                network_alpha=alpha,
            )
    if not lora_attn_procs:
        raise RuntimeError("Failed to locate attention processors for LoRA training")
    unet.set_attn_processor(lora_attn_procs)
    if AttnProcsLayers is not None:
        trainable: nn.Module = AttnProcsLayers(unet.attn_processors)
    else:
        unique_processors: list[nn.Module] = []
        seen_ids: set[int] = set()
        for processor in unet.attn_processors.values():
            if id(processor) in seen_ids:
                continue
            unique_processors.append(processor)
            seen_ids.add(id(processor))
        trainable = nn.ModuleList(unique_processors)
    trainable.requires_grad_(True)
    return trainable


def _optimizer_and_scheduler(parameters: Iterable[nn.Parameter], config: BaseDiffusionTrainingConfig, total_steps: int):
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    try:
        from diffusers.optimization import get_scheduler
    except ImportError:  # pragma: no cover - fallback to torch scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
        return optimizer, scheduler

    scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _diffusion_loss(
    *,
    unet,
    noise_scheduler,
    vae,
    text_encoder,
    pixel_values,
    input_ids,
    attention_mask,
    weight_dtype,
    device,
) -> torch.Tensor:
    latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample() * 0.18215
    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device, dtype=torch.long)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    encoder_hidden_states = text_encoder(
        input_ids,
        attention_mask=attention_mask,
    )[0]
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:  # pragma: no cover - defensive branch
        target = noise
    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def _train_base_diffusion(
    *,
    config: BaseDiffusionTrainingConfig,
    samples: Sequence["Image.Image"],
    prompts: Sequence[str],
    output_dir: Path,
    device: torch.device,
):
    if config.method.lower() != "lora":
        LOGGER.warning("当前实现仅支持 LoRA 方式，已回退至 LoRA 训练。")

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Diffusers is required for reconstruction. Install diffusers>=0.20.0"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.model_id,
        safety_checker=None,
        requires_safety_checker=False,
    )
    weight_dtype = torch.float16 if config.mixed_precision == "fp16" and torch.cuda.is_available() else torch.float32
    pipeline.to(device)
    pipeline.unet.to(device, dtype=weight_dtype)
    pipeline.vae.to(device, dtype=weight_dtype)
    pipeline.text_encoder.to(device, dtype=weight_dtype)

    dataset = _PromptedImageDataset(samples, prompts, pipeline.tokenizer, config.resolution)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

    total_steps = max(config.max_train_steps, len(dataloader))
    trainable = _prepare_lora_layers(pipeline.unet, config.lora_rank, config.lora_alpha)
    optimizer, scheduler = _optimizer_and_scheduler(trainable.parameters(), config, total_steps)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    pipeline.unet.train()
    step = 0
    losses: deque[float] = deque(maxlen=50)
    noise_scheduler = pipeline.scheduler
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder

    for epoch in range(math.ceil(config.max_train_steps / max(len(dataloader), 1))):
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            loss = _diffusion_loss(
                unet=pipeline.unet,
                noise_scheduler=noise_scheduler,
                vae=vae,
                text_encoder=text_encoder,
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                weight_dtype=weight_dtype,
                device=device,
            )

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            losses.append(float(loss.item()))
            step += 1
            if step % 50 == 0:
                LOGGER.info("Base diffusion training step %d/%d, loss=%.4f", step, config.max_train_steps, np.mean(losses))
            if (
                config.early_stop_loss_threshold is not None
                and len(losses) == losses.maxlen
                and np.mean(losses) <= config.early_stop_loss_threshold
            ):
                LOGGER.info(
                    "Early stopping base diffusion training at step %d because running loss %.4f <= threshold %.4f",
                    step,
                    np.mean(losses),
                    config.early_stop_loss_threshold,
                )
                break
            if step >= config.max_train_steps:
                break
        if step >= config.max_train_steps:
            break
        if (
            config.early_stop_loss_threshold is not None
            and len(losses) == losses.maxlen
            and np.mean(losses) <= config.early_stop_loss_threshold
        ):
            break

    pipeline.unet.save_attn_procs(output_dir)
    pipeline.save_pretrained(output_dir)
    LOGGER.info("Base diffusion training completed after %d steps", step)
    return pipeline


# ---------------------------------------------------------------------------
# Sensitive feature fine-tuning
# ---------------------------------------------------------------------------


def _load_controlnet_pipeline(base_pipeline, controlnet_model_id: str, device: torch.device, dtype: torch.dtype):
    try:
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "ControlNet support requires diffusers>=0.21.0"
        ) from exc

    load_kwargs = {}
    if dtype == torch.float16:
        load_kwargs["torch_dtype"] = dtype
    controlnet = ControlNetModel.from_pretrained(controlnet_model_id, **load_kwargs)
    controlnet_pipeline = StableDiffusionControlNetPipeline(
        vae=base_pipeline.vae,
        text_encoder=base_pipeline.text_encoder,
        tokenizer=base_pipeline.tokenizer,
        unet=base_pipeline.unet,
        scheduler=base_pipeline.scheduler,
        safety_checker=None,
        feature_extractor=None,
        controlnet=controlnet,
    )
    controlnet_pipeline.to(device)
    if dtype == torch.float16 and device.type == "cuda":
        controlnet_pipeline.unet.to(device=device, dtype=dtype)
        controlnet_pipeline.vae.to(device=device, dtype=dtype)
        controlnet_pipeline.controlnet.to(device=device, dtype=dtype)
        # keep text encoder in full precision to avoid numerical issues
        controlnet_pipeline.text_encoder.to(device=device, dtype=torch.float32)
    else:
        controlnet_pipeline.text_encoder.to(device=device, dtype=torch.float32)
    return controlnet_pipeline


def _build_sensitive_prompts(base_prompt: str, texture_tokens: Sequence[str], suffix: str, count: int) -> list[str]:
    tokens = list(texture_tokens) if texture_tokens else []
    if not tokens:
        tokens = [suffix]
    prompts = []
    for index in range(count):
        token = tokens[index % len(tokens)]
        prompt = f"{base_prompt} {token}"
        if suffix and suffix not in prompt:
            prompt = f"{prompt}, {suffix}"
        prompts.append(prompt)
    return prompts


def _fine_tune_sensitive_features(
    *,
    base_pipeline,
    config: SensitiveFeatureFinetuneConfig,
    sfi_summary: dict[str, object] | None,
    output_dir: Path,
    base_prompt: str,
    device: torch.device,
):
    if not config.enabled:
        LOGGER.info("Sensitive feature finetuning disabled")
        return base_pipeline

    if sfi_summary is None:
        LOGGER.warning("Sensitive feature assets are unavailable; skipping finetuning stage")
        return base_pipeline

    weight_dtype = torch.float16 if config.mixed_precision == "fp16" and torch.cuda.is_available() else torch.float32

    classes = sfi_summary.get("classes", {})
    if not classes:
        LOGGER.warning("SFI summary does not contain class-level assets; skipping finetuning")
        return base_pipeline

    class_assets = next(iter(classes.values()))
    controlnet_assets = class_assets.get("controlnet", {})
    texture_tokens = class_assets.get("texture_tokens", [])

    region_mask_path = Path(controlnet_assets.get("region_mask", ""))
    canny_original_path = Path(controlnet_assets.get("canny_original", ""))
    edge_mask_path = Path(controlnet_assets.get("edge_mask", ""))

    if not region_mask_path.exists() or not canny_original_path.exists():
        LOGGER.warning("ControlNet conditioning files are missing; skipping finetuning")
        return base_pipeline

    control_images = []
    try:
        control_images.append(_load_grayscale(region_mask_path, base_pipeline.unet.config.sample_size * 8))
    except Exception as exc:  # pragma: no cover - fallback for legacy configs
        LOGGER.debug("Failed to load region mask (%s): %s", region_mask_path, exc)
    try:
        control_images.append(_load_grayscale(canny_original_path, base_pipeline.unet.config.sample_size * 8))
    except Exception as exc:
        LOGGER.debug("Failed to load canny map (%s): %s", canny_original_path, exc)
    if edge_mask_path.exists():
        try:
            control_images.append(_load_grayscale(edge_mask_path, base_pipeline.unet.config.sample_size * 8))
        except Exception as exc:
            LOGGER.debug("Failed to load edge mask (%s): %s", edge_mask_path, exc)

    control_images = [tensor for tensor in control_images if tensor.numel() > 0]
    if not control_images:
        LOGGER.warning("No valid control images found; skipping finetuning")
        return base_pipeline

    prompts = _build_sensitive_prompts(base_prompt, texture_tokens, config.prompt_suffix, min(len(control_images), config.max_condition_images))

    sample_image = base_pipeline("a person").images[0]
    dataset = _ControlNetDataset(
        base_image=sample_image,
        control_images=control_images,
        prompts=prompts,
        tokenizer=base_pipeline.tokenizer,
        resolution=base_pipeline.unet.config.sample_size * 8,
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=False)

    control_pipeline = _load_controlnet_pipeline(base_pipeline, config.controlnet_model_id, device, weight_dtype)
    control_pipeline.unet.train()

    current_dtype = weight_dtype
    use_autocast = current_dtype == torch.float16 and device.type == "cuda" and autocast is not None
    if current_dtype == torch.float16 and not use_autocast:
        LOGGER.warning(
            "AMP is unavailable on device %s; falling back to float32 precision for sensitive finetuning",
            device,
        )
        current_dtype = torch.float32
        control_pipeline.unet.to(device=device, dtype=current_dtype)
        control_pipeline.vae.to(device=device, dtype=current_dtype)
        control_pipeline.controlnet.to(device=device, dtype=current_dtype)
        control_pipeline.text_encoder.to(device=device, dtype=torch.float32)
    else:
        control_pipeline.text_encoder.to(device=device, dtype=torch.float32)

    scaler = _make_grad_scaler(use_autocast, device.type)
    optimizer = torch.optim.AdamW(control_pipeline.unet.parameters(), lr=config.learning_rate)

    step = 0
    for epoch in range(math.ceil(config.max_train_steps / max(len(dataloader), 1))):
        for batch in dataloader:
            optimizer.zero_grad(set_to_none=True)

            pixel_values = batch["pixel_values"].to(device, dtype=current_dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            control = batch["control"].to(device, dtype=current_dtype)

            if not torch.isfinite(pixel_values).all():
                LOGGER.warning("Skipping batch with non-finite pixel values during sensitive finetuning")
                continue
            if not torch.isfinite(control).all():
                LOGGER.warning("Skipping batch with non-finite control signal during sensitive finetuning")
                continue

            autocast_cm = _autocast_context(device.type, torch.float16) if use_autocast else nullcontext()
            with autocast_cm:
                latents = control_pipeline.vae.encode(pixel_values).latent_dist.sample() * 0.18215
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    control_pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
                noisy_latents = control_pipeline.scheduler.add_noise(latents, noise, timesteps)

                encoder_hidden_states = control_pipeline.text_encoder(input_ids, attention_mask=attention_mask)[0]

                controlnet_output = control_pipeline.controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=control,
                    conditioning_scale=config.condition_scale,
                    return_dict=True,
                )

                model_pred = control_pipeline.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    down_block_additional_residuals=controlnet_output.down_block_res_samples,
                    mid_block_additional_residual=controlnet_output.mid_block_res_sample,
                ).sample

                if control_pipeline.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif control_pipeline.scheduler.config.prediction_type == "v_prediction":
                    target = control_pipeline.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = noise

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            if not torch.isfinite(loss):
                LOGGER.error(
                    "Sensitive finetuning produced a non-finite loss at step %d; switching to float32 precision",
                    step + 1,
                )
                current_dtype = torch.float32
                use_autocast = False
                scaler = _make_grad_scaler(False, device.type)
                control_pipeline.unet.to(device=device, dtype=current_dtype)
                control_pipeline.vae.to(device=device, dtype=current_dtype)
                control_pipeline.controlnet.to(device=device, dtype=current_dtype)
                control_pipeline.text_encoder.to(device=device, dtype=torch.float32)
                new_lr = config.learning_rate * 0.5
                LOGGER.warning(
                    "Sensitive finetuning fallback: using float32 precision and reducing learning rate to %.2e",
                    new_lr,
                )
                optimizer = torch.optim.AdamW(control_pipeline.unet.parameters(), lr=new_lr)
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(control_pipeline.unet.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(control_pipeline.unet.parameters(), max_norm=1.0)
                optimizer.step()

            step += 1
            if step % 25 == 0:
                LOGGER.info("Sensitive finetuning step %d/%d, loss=%.4f", step, config.max_train_steps, float(loss.item()))
            if step >= config.max_train_steps:
                break
        if step >= config.max_train_steps:
            break

    control_pipeline.unet.save_attn_procs(output_dir / "sensitive")
    control_pipeline.save_pretrained(output_dir / "sensitive")
    LOGGER.info("Sensitive finetuning completed after %d steps", step)
    return control_pipeline


# ---------------------------------------------------------------------------
# Adaptive generation and evaluation
# ---------------------------------------------------------------------------


def _evaluate_generated(
    images: Sequence["Image.Image"],
    *,
    model_before: nn.Module,
    model_after: nn.Module,
    normalization_stats,
    target_class: int,
    device: torch.device,
    batch_size: int,
) -> tuple[float, float]:
    dataset = _GeneratedDataset(images, normalization_stats=normalization_stats, label=target_class)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    before = accuracy(model_before, dataloader, device)
    after = accuracy(model_after, dataloader, device)
    return before, after


def _generate_images(
    pipeline,
    *,
    prompt: str,
    batch_size: int,
    num_images: int,
    num_inference_steps: int,
    guidance_scale: float,
    controlnet_scale: float | None = None,
    control_image=None,
) -> list["Image.Image"]:
    images: list["Image.Image"] = []
    generator = torch.Generator(device=pipeline.device) if hasattr(pipeline, "device") else torch.Generator()
    while len(images) < num_images:
        current_batch = min(batch_size, num_images - len(images))
        kwargs = {
            "prompt": [prompt] * current_batch,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }
        if control_image is not None:
            kwargs["image"] = control_image
            if controlnet_scale is not None:
                kwargs["controlnet_conditioning_scale"] = controlnet_scale
        output = pipeline(**kwargs)
        images.extend(output.images)
    return images


def _adaptive_generation_loop(
    pipeline,
    *,
    config: AdaptiveGenerationConfig,
    prompt: str,
    model_before: nn.Module,
    model_after: nn.Module,
    normalization_stats,
    target_class: int,
    control_image=None,
    output_dir: Path,
    device: torch.device,
) -> tuple[list[AdaptiveBatchRecord], list[AdaptiveBatchRecord]]:
    rng = random.Random(config.seed)
    accepted: list[AdaptiveBatchRecord] = []
    rejected: list[AdaptiveBatchRecord] = []
    guidance = config.guidance_scale
    inference_steps = config.num_inference_steps
    control_scale = config.controlnet_scale

    for batch_index in range(config.max_batches):
        LOGGER.info(
            "Adaptive generation batch %d/%d with guidance=%.2f, steps=%d",
            batch_index + 1,
            config.max_batches,
            guidance,
            inference_steps,
        )

        images = _generate_images(
            pipeline,
            prompt=prompt,
            batch_size=config.batch_size,
            num_images=config.images_per_batch,
            num_inference_steps=inference_steps,
            guidance_scale=guidance,
            controlnet_scale=control_scale,
            control_image=control_image,
        )
        before_acc, after_acc = _evaluate_generated(
            images,
            model_before=model_before,
            model_after=model_after,
            normalization_stats=normalization_stats,
            target_class=target_class,
            device=device,
            batch_size=config.batch_size,
        )

        accepted_flag = (
            config.target_accuracy_min <= after_acc <= config.target_accuracy_max
            and before_acc >= after_acc + config.accuracy_margin
        )
        batch_dir: Path | None = None
        if accepted_flag:
            batch_dir = output_dir / f"batch_{batch_index:02d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            for idx, image in enumerate(images):
                image.save(batch_dir / f"{idx:05d}.png")
            LOGGER.info(
                "Batch %d accepted: before=%.4f, after=%.4f (saved to %s)",
                batch_index,
                before_acc,
                after_acc,
                batch_dir,
            )
        else:
            LOGGER.info(
                "Batch %d rejected: before=%.4f, after=%.4f (target range=%.2f-%.2f)",
                batch_index,
                before_acc,
                after_acc,
                config.target_accuracy_min,
                config.target_accuracy_max,
            )
            if config.save_rejected:
                batch_dir = output_dir / "rejected" / f"batch_{batch_index:02d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                for idx, image in enumerate(images):
                    image.save(batch_dir / f"{idx:05d}.png")

        record = AdaptiveBatchRecord(
            batch_index=batch_index,
            before_accuracy=before_acc,
            after_accuracy=after_acc,
            guidance_scale=guidance,
            controlnet_scale=control_scale,
            inference_steps=inference_steps,
            accepted=accepted_flag,
            output_dir=None if batch_dir is None else str(batch_dir),
        )
        if accepted_flag:
            accepted.append(record)
            break
        rejected.append(record)

        if after_acc > config.target_accuracy_max:
            guidance = max(1.0, guidance - config.adjustment_step)
            inference_steps = max(10, inference_steps - config.step_increment)
        elif after_acc < config.target_accuracy_min:
            guidance = guidance + config.adjustment_step
            inference_steps = inference_steps + config.step_increment
        else:
            # Criterion on margin failed; slightly perturb guidance.
            direction = -1.0 if before_acc - after_acc < config.accuracy_margin else 1.0
            guidance = max(1.0, guidance + direction * (config.adjustment_step / 2.0))
        if control_scale is not None:
            control_scale = max(0.1, min(2.0, control_scale + rng.uniform(-0.1, 0.1)))

    return accepted, rejected


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_data_reconstruction(
    inference: LabelInferenceResult,
    *,
    dataloader: DataLoader,
    normalization_stats,
    model_before: nn.Module,
    model_after: nn.Module,
    output_root: Path,
    device: torch.device,
    config: DataReconstructionConfig | None = None,
    sfi_summary: dict[str, object] | None = None,
    dataset_name: str | None = None,
) -> DataReconstructionResult:
    """Execute the full reconstruction workflow."""

    if config is None:
        config = DataReconstructionConfig()

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    predicted_class = int(inference.predicted_class)
    base_prompt = f"{config.base.prompt_prefix} class {predicted_class}"
    samples = _gather_label_samples(
        dataloader,
        predicted_class,
        normalization_stats=normalization_stats,
        limit=config.base.sample_limit,
    )
    if not samples:
        raise RuntimeError(
            f"No samples of class {predicted_class} were found in the evaluation dataloader"
        )

    prompts = [base_prompt for _ in range(len(samples))]
    base_dir = output_root / "base"
    LOGGER.info(
        "Starting base diffusion fine-tuning using %d samples for class %d", len(samples), predicted_class
    )
    base_pipeline = _train_base_diffusion(
        config=config.base,
        samples=samples,
        prompts=prompts,
        output_dir=base_dir,
        device=device,
    )

    sensitive_pipeline = _fine_tune_sensitive_features(
        base_pipeline=base_pipeline,
        config=config.sensitive,
        sfi_summary=sfi_summary,
        output_dir=output_root,
        base_prompt=base_prompt,
        device=device,
    )

    generator_pipeline = sensitive_pipeline if config.sensitive.enabled else base_pipeline

    control_image = None
    if sfi_summary is not None and config.sensitive.enabled:
        class_assets = next(iter(sfi_summary.get("classes", {}).values()), None)
        if class_assets is not None:
            controlnet_assets = class_assets.get("controlnet", {})
            canny_path = Path(controlnet_assets.get("canny_original", ""))
            if canny_path.exists():
                control_image = _load_grayscale(canny_path, generator_pipeline.unet.config.sample_size * 8)
                if control_image.dim() == 2:
                    control_image = control_image.unsqueeze(0).repeat(3, 1, 1)
                control_image = control_image.unsqueeze(0)

    accepted, rejected = _adaptive_generation_loop(
        generator_pipeline,
        config=config.adaptive,
        prompt=base_prompt,
        model_before=model_before,
        model_after=model_after,
        normalization_stats=normalization_stats,
        target_class=predicted_class,
        control_image=control_image,
        output_dir=output_root,
        device=device,
    )

    summary = DataReconstructionResult(
        predicted_class=predicted_class,
        output_dir=str(output_root),
        accepted_batches=accepted,
        rejected_batches=rejected,
        config={
            "dataset": dataset_name,
            "base": asdict(config.base),
            "sensitive": asdict(config.sensitive),
            "adaptive": asdict(config.adaptive),
        },
    )

    with (output_root / "reconstruction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "predicted_class": summary.predicted_class,
                "accepted": [asdict(record) for record in summary.accepted_batches],
                "rejected": [asdict(record) for record in summary.rejected_batches],
                "config": summary.config,
            },
            handle,
            indent=2,
        )

    LOGGER.info("Data reconstruction completed. Summary saved to %s", output_root / "reconstruction_summary.json")
    return summary


__all__ = [
    "BaseDiffusionTrainingConfig",
    "SensitiveFeatureFinetuneConfig",
    "AdaptiveGenerationConfig",
    "DataReconstructionConfig",
    "AdaptiveBatchRecord",
    "DataReconstructionResult",
    "run_data_reconstruction",
]

