"""Lightweight sensitive-region extraction for diffusion weighting."""
from __future__ import annotations

import torch

from .label_inference import LabelInferenceResult, SensitiveFeature, _normalize_heatmap


def _mean_map(cache: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    tensor = cache.get(key)
    if tensor is None or tensor.numel() == 0:
        raise ValueError(f"heatmap cache missing {key}")
    return tensor.float().mean(dim=0)


def infer_sensitive_features(
    inference: LabelInferenceResult,
    *_,
    device: torch.device | None = None,
    **__,
) -> tuple[list[SensitiveFeature], dict[str, torch.Tensor]]:
    """Collapse Grad-CAM / saliency deltas into a single weighting mask."""

    cls = int(inference.predicted_class)
    cache = inference.heatmap_cache.get(cls)
    if not cache:
        return [], {}

    try:
        gradcam_before = _mean_map(cache, "gradcam_before")
        gradcam_after = _mean_map(cache, "gradcam_after")
        saliency_before = _mean_map(cache, "saliency_before")
        saliency_after = _mean_map(cache, "saliency_after")
    except ValueError:
        return [], {}

    diff = torch.relu(gradcam_before - gradcam_after) + torch.relu(saliency_before - saliency_after)
    if diff.dim() != 2:
        diff = diff.mean(dim=0)
    diff = diff.clamp(min=0.0)

    mask = _normalize_heatmap(diff.unsqueeze(0))[0]
    threshold = float(mask.mean().item())
    weighted = torch.where(mask >= threshold, mask, torch.zeros_like(mask))
    weighted = _normalize_heatmap(weighted.unsqueeze(0))[0]

    if device is not None:
        weighted = weighted.to(device)

    feature = SensitiveFeature(
        name=f"contrastive_mask_cls{cls}",
        score=float(weighted.mean().item()),
        source="heatmap-weighted-mse",
    )

    return [feature], {feature.name: weighted.detach()}


__all__ = ["infer_sensitive_features"]
