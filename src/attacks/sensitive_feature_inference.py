"""Sensitive feature inference based on micro-structure localisation."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .label_inference import (
    LabelInferenceResult,
    SensitiveFeature,
    _compute_saliency_map,
    _edge_focus_metric,
    _min_max_normalize,
    _normalize_heatmap,
)


def _mean_map(cache: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    tensor = cache.get(key)
    if tensor is None or tensor.numel() == 0:
        raise ValueError(f"heatmap cache missing {key}")
    return tensor.float().mean(dim=0)


def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return _normalize_heatmap(tensor.unsqueeze(0))[0]
    if tensor.dim() == 3:
        return _normalize_heatmap(tensor)[0]
    raise ValueError("Expected a 2D or 3D tensor for heatmap normalisation")


def _build_diff_map(cache: dict[str, torch.Tensor]) -> torch.Tensor:
    cam_before = _mean_map(cache, "gradcam_before")
    cam_after = _mean_map(cache, "gradcam_after")
    saliency_before = _mean_map(cache, "saliency_before")
    saliency_after = _mean_map(cache, "saliency_after")

    cam_diff = torch.relu(cam_before - cam_after)
    saliency_diff = torch.relu(saliency_before - saliency_after)
    combined = cam_diff + saliency_diff
    return _normalize_map(combined)


def _pick_reference_maps(
    inference: LabelInferenceResult,
    *,
    exclude: int,
    limit: int = 3,
) -> list[torch.Tensor]:
    drops = inference.accuracy_drop.to(torch.float32)
    indices = torch.argsort(drops, descending=False).tolist()
    references: list[torch.Tensor] = []
    for cls in indices:
        if cls == exclude:
            continue
        cache = inference.heatmap_cache.get(int(cls))
        if not cache:
            continue
        try:
            references.append(_build_diff_map(cache))
        except ValueError:
            continue
        if len(references) >= limit:
            break
    return references


def _patch_proposals(
    heatmap: torch.Tensor,
    thresholds: Sequence[float],
    window: int,
) -> list[tuple[int, int, int, int]]:
    if window <= 0:
        raise ValueError("window size must be positive")

    window = max(3, window)
    stride = max(1, window // 2)
    masks = [(heatmap >= (heatmap.max() * float(th))).float() for th in thresholds]
    if not masks:
        masks = [(heatmap >= (heatmap.max() * 0.5)).float()]
    union = torch.stack(masks, dim=0).amax(dim=0)

    pooled = F.max_pool2d(
        heatmap.unsqueeze(0).unsqueeze(0),
        kernel_size=window,
        stride=stride,
        padding=window // 2,
    )
    upsampled = F.interpolate(pooled, size=heatmap.shape, mode="nearest")[0, 0]
    peaks = (heatmap >= upsampled) & (union > 0)
    ys, xs = torch.where(peaks)

    proposals: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    height, width = heatmap.shape
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = max(0, y - window // 2)
        x0 = max(0, x - window // 2)
        y1 = min(height, y0 + window)
        x1 = min(width, x0 + window)
        if y1 - y0 < 2 or x1 - x0 < 2:
            continue
        patch = (y0, y1, x0, x1)
        if patch in seen:
            continue
        seen.add(patch)
        proposals.append(patch)
    return proposals


def _combine_scores(vectors: Sequence[Sequence[float]]) -> list[torch.Tensor]:
    tensors = [torch.tensor(vector, dtype=torch.float32) for vector in vectors]
    return [_min_max_normalize(tensor) for tensor in tensors]


def _make_feature_name(y0: int, y1: int, x0: int, x1: int, height: int, width: int, rank: int) -> str:
    vertical = "top" if y0 < height // 3 else "bottom" if y1 > 2 * height // 3 else "center"
    horizontal = "-left" if x0 < width // 3 else "-right" if x1 > 2 * width // 3 else ""
    return f"patch{rank:02d}_{vertical}{horizontal}_{y1 - y0}x{x1 - x0}"


def _prepare_drift_map(
    inference: LabelInferenceResult,
    cls: int,
    before: nn.Module,
    after: nn.Module,
    *,
    device: torch.device,
    fallback: torch.Tensor,
) -> torch.Tensor:
    samples = inference.sample_bank.get(cls)
    if samples is not None and samples.numel() > 0:
        batch = samples[: min(4, samples.size(0))].to(device)
        targets = torch.full((batch.size(0),), cls, device=device, dtype=torch.long)
        before = before.to(device).eval()
        after = after.to(device).eval()
        with torch.enable_grad():
            saliency_before = _compute_saliency_map(before, batch, targets).detach()
            saliency_after = _compute_saliency_map(after, batch, targets).detach()
        drift = (saliency_before - saliency_after).abs().mean(dim=0)
        return drift.detach().cpu()

    cache = inference.heatmap_cache.get(cls)
    if cache:
        saliency_before = _mean_map(cache, "saliency_before")
        saliency_after = _mean_map(cache, "saliency_after")
        return (saliency_before - saliency_after).abs()
    return torch.zeros_like(fallback)


def infer_sensitive_features(
    inference: LabelInferenceResult,
    before: nn.Module,
    after: nn.Module,
    *,
    device: torch.device,
    k_patches: int = 8,
    thresholds: Tuple[float, ...] = (0.5, 0.7),
) -> tuple[list[SensitiveFeature], dict[str, torch.Tensor]]:
    """Infer sensitive micro-structure patches from cached heatmaps."""

    cls = int(inference.predicted_class)
    cache = inference.heatmap_cache.get(cls)
    if not cache:
        return [], {}

    try:
        diff_map = _build_diff_map(cache)
    except ValueError:
        return [], {}

    reference_maps = _pick_reference_maps(inference, exclude=cls)
    if reference_maps:
        reference = torch.stack(reference_maps, dim=0).mean(dim=0)
    else:
        reference = torch.zeros_like(diff_map)

    height, width = diff_map.shape
    window = max(8, min(height, width) // 12)
    proposals = _patch_proposals(diff_map, thresholds, window)
    if not proposals:
        return [], {}

    saliency_before = _mean_map(cache, "saliency_before")
    saliency_after = _mean_map(cache, "saliency_after")
    drift_map = _prepare_drift_map(inference, cls, before, after, device=device, fallback=diff_map)
    drift_map = drift_map.to(diff_map.dtype)
    drift_map = F.interpolate(
        drift_map.unsqueeze(0).unsqueeze(0),
        size=diff_map.shape,
        mode="bilinear",
        align_corners=False,
    )[0, 0]

    raw_uniqueness: list[float] = []
    raw_drift: list[float] = []
    raw_edge: list[float] = []
    raw_high_freq: list[float] = []
    entries: list[tuple[tuple[int, int, int, int], dict[str, float]]] = []

    border_ratio = 0.3

    for proposal in proposals:
        y0, y1, x0, x1 = proposal
        patch_map = diff_map[y0:y1, x0:x1]
        ref_patch = reference[y0:y1, x0:x1]
        drift_patch = drift_map[y0:y1, x0:x1]
        saliency_patch_before = saliency_before[y0:y1, x0:x1]
        saliency_patch_after = saliency_after[y0:y1, x0:x1]

        uniqueness = float((patch_map.mean() - ref_patch.mean()).clamp(min=0.0).item())
        drift_score = float(drift_patch.mean().item())

        edge_before = _edge_focus_metric(saliency_patch_before.unsqueeze(0), border_ratio=border_ratio)
        edge_after = _edge_focus_metric(saliency_patch_after.unsqueeze(0), border_ratio=border_ratio)
        edge_density = float((edge_before - edge_after).abs().mean().item())

        spectrum = torch.fft.fft2(patch_map)
        magnitude = torch.abs(spectrum)
        if magnitude.numel() == 0:
            high_freq = 0.0
        else:
            h, w = patch_map.shape
            cy, cx = h // 2, w // 2
            low_h = max(1, h // 6)
            low_w = max(1, w // 6)
            low_freq = magnitude[max(0, cy - low_h) : min(h, cy + low_h), max(0, cx - low_w) : min(w, cx + low_w)].sum()
            high_freq = float((magnitude.sum() - low_freq) / (magnitude.sum() + 1e-6))

        raw_uniqueness.append(uniqueness)
        raw_drift.append(drift_score)
        raw_edge.append(edge_density)
        raw_high_freq.append(high_freq)

        entries.append(
            (
                proposal,
                {
                    "uniqueness": uniqueness,
                    "drift": drift_score,
                    "edge_density": edge_density,
                    "high_freq": high_freq,
                },
            )
        )

    if not entries:
        return [], {}

    norm_uniqueness, norm_drift, norm_edge, norm_high = _combine_scores(
        [raw_uniqueness, raw_drift, raw_edge, raw_high_freq]
    )

    scores = []
    for idx, (_, stats) in enumerate(entries):
        score = (
            0.4 * norm_uniqueness[idx].item()
            + 0.3 * norm_drift[idx].item()
            + 0.2 * norm_edge[idx].item()
            + 0.1 * norm_high[idx].item()
        )
        stats["score"] = score
        scores.append(score)

    order = torch.tensor(scores).argsort(descending=True).tolist()
    if k_patches > 0:
        order = order[:k_patches]

    features: list[SensitiveFeature] = []
    masks: dict[str, torch.Tensor] = {}

    for rank, entry_idx in enumerate(order, start=1):
        (y0, y1, x0, x1), stats = entries[entry_idx]
        name = _make_feature_name(y0, y1, x0, x1, height, width, rank)
        features.append(
            SensitiveFeature(
                name=name,
                score=float(stats["score"]),
                source="heatmap+drift+reference",
            )
        )
        mask = torch.zeros_like(diff_map, dtype=torch.float32)
        mask[y0:y1, x0:x1] = 1.0
        masks[name] = mask

    return features, masks


__all__ = ["infer_sensitive_features"]

