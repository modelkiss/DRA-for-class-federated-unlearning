"""Sensitive feature inference utilities for data reconstruction guidance."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .label_inference import LabelInferenceResult
from ..utils.normalization import denormalize


@dataclass
class SensitiveFeatureConfig:
    """Configuration controlling the sensitive feature inference pipeline."""

    max_classes: int = 1
    mask_quantile: float = 0.8
    mask_min_threshold: float = 0.25
    edge_border_ratio: float = 0.2
    patch_size: int = 32
    num_patches: int = 32
    dct_components: int = 64
    gabor_sigmas: Sequence[float] = (1.5, 3.0, 4.5)
    gabor_frequencies: Sequence[float] = (0.08, 0.16, 0.24)
    gabor_orientations: int = 8
    num_prototypes: int = 8
    canny_low: int = 50
    canny_high: int = 150
    stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5"


def _normalize_map(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.numel() == 0:
        return tensor
    tensor = tensor - tensor.min()
    max_val = tensor.max()
    if torch.isclose(max_val, torch.zeros(1, dtype=tensor.dtype, device=tensor.device)):
        return torch.zeros_like(tensor)
    return tensor / max_val


def _quantile_threshold(values: torch.Tensor, quantile: float, minimum: float) -> float:
    flat = values.flatten()
    if flat.numel() == 0:
        return float(minimum)
    if not 0.0 < quantile < 1.0:
        return float(max(minimum, quantile))
    q = torch.quantile(flat, torch.tensor(quantile, device=flat.device))
    return float(max(q.item(), minimum))


def _save_grayscale_png(array: np.ndarray, path: Path) -> None:
    from PIL import Image

    clipped = np.clip(array, 0.0, 1.0)
    image = Image.fromarray((clipped * 255.0).astype(np.uint8))
    image.save(path)


def _save_binary_mask(mask: np.ndarray, path: Path) -> None:
    from PIL import Image

    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image.save(path)


def _build_edge_mask(shape: Sequence[int], border_ratio: float) -> torch.Tensor:
    height, width = shape
    border_h = max(1, int(height * border_ratio))
    border_w = max(1, int(width * border_ratio))
    mask = torch.zeros((height, width), dtype=torch.bool)
    mask[:border_h, :] = True
    mask[-border_h:, :] = True
    mask[:, :border_w] = True
    mask[:, -border_w:] = True
    return mask


def _compute_canny(image: np.ndarray, low: int, high: int) -> np.ndarray:
    image = np.clip(image, 0.0, 1.0)
    if image.ndim == 3 and image.shape[2] > 1:
        gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    else:
        gray = image[..., 0] if image.ndim == 3 else image
    gray_uint8 = (gray * 255.0).astype(np.uint8)

    try:  # Prefer OpenCV if available.
        import cv2  # type: ignore

        edges = cv2.Canny(gray_uint8, low, high)
        return edges.astype(np.float32) / 255.0
    except Exception:  # pragma: no cover - optional dependency fallback
        try:
            from skimage.feature import canny  # type: ignore

            edges = canny(gray.astype(np.float32), low_threshold=low / 255.0, high_threshold=high / 255.0)
            return edges.astype(np.float32)
        except Exception:
            # Final fallback: gradient magnitude thresholding.
            gy, gx = np.gradient(gray.astype(np.float32))
            magnitude = np.sqrt(gx**2 + gy**2)
            if magnitude.size == 0:
                return np.zeros_like(gray, dtype=np.float32)
            threshold = np.quantile(magnitude, 0.9)
            return (magnitude >= threshold).astype(np.float32)


def _build_gabor_kernel(sigma: float, frequency: float, theta: float) -> torch.Tensor:
    radius = max(3, int(math.ceil(3 * sigma)))
    size = radius * 2 + 1
    ys, xs = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    x_theta = xs * cos_theta + ys * sin_theta
    y_theta = -xs * sin_theta + ys * cos_theta
    gaussian = torch.exp(-0.5 * (x_theta**2 + y_theta**2) / (sigma**2))
    kernel = gaussian * torch.cos(2 * math.pi * frequency * x_theta)
    kernel -= kernel.mean()
    norm = torch.sqrt((kernel**2).sum())
    if torch.isclose(norm, torch.tensor(0.0)):
        return kernel
    return kernel / norm


def _compute_gabor_statistics(
    patch: torch.Tensor,
    sigmas: Sequence[float],
    frequencies: Sequence[float],
    orientations: int,
) -> List[dict]:
    if patch.ndim != 2:
        raise ValueError("Gabor statistics expect a 2D grayscale patch")

    patch_tensor = patch.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    device = patch_tensor.device

    entries: List[dict] = []
    for sigma in sigmas:
        for frequency in frequencies:
            for index in range(max(1, orientations)):
                theta = math.pi * index / max(1, orientations)
                kernel = _build_gabor_kernel(sigma, frequency, theta).to(device=device, dtype=patch_tensor.dtype)
                response = F.conv2d(patch_tensor, kernel.view(1, 1, *kernel.shape), padding=kernel.shape[-1] // 2)
                values = response[0, 0]
                mean_energy = float(values.pow(2).mean().item())
                variance = float(values.var(unbiased=False).item())
                entries.append(
                    {
                        "sigma": float(sigma),
                        "frequency": float(frequency),
                        "theta_deg": float(theta * 180.0 / math.pi),
                        "mean_energy": mean_energy,
                        "variance": variance,
                    }
                )
    return entries


def _compute_dct_spectrum(patch: np.ndarray, top_k: int) -> np.ndarray:
    if patch.ndim == 3 and patch.shape[2] > 1:
        gray = patch.mean(axis=2)
    else:
        gray = patch if patch.ndim == 2 else patch[..., 0]

    try:
        from scipy.fft import dct  # type: ignore

        coeff = dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")
    except Exception:  # pragma: no cover - optional dependency fallback
        coeff = np.fft.fft2(gray)
        coeff = np.real(coeff)

    flat = np.abs(coeff).flatten()
    if flat.size == 0:
        return np.zeros((top_k,), dtype=np.float32)

    order = np.argsort(flat)[::-1]
    top_indices = order[: min(top_k, flat.size)]
    selected = flat[top_indices].astype(np.float32)
    if selected.size < top_k:
        padding = np.zeros((top_k - selected.size,), dtype=np.float32)
        selected = np.concatenate([selected, padding], axis=0)
    return selected


def _simple_kmeans(data: np.ndarray, num_clusters: int, max_iter: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    if data.shape[0] == 0 or num_clusters <= 0:
        return np.empty((0, data.shape[1])), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed=42)
    num_clusters = min(num_clusters, data.shape[0])
    indices = rng.choice(data.shape[0], size=num_clusters, replace=False)
    centroids = data[indices]

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = []
        for cluster in range(num_clusters):
            members = data[labels == cluster]
            if members.size == 0:
                new_centroids.append(centroids[cluster])
            else:
                new_centroids.append(members.mean(axis=0))
        new_centroids_array = np.stack(new_centroids, axis=0)
        if np.allclose(new_centroids_array, centroids):
            break
        centroids = new_centroids_array
    return centroids, labels


def _denormalize_samples(samples: torch.Tensor, stats: Tuple[Tuple[float, ...], Tuple[float, ...]]) -> torch.Tensor:
    tensor = samples.clone()
    if stats is None:
        return tensor
    return denormalize(tensor, stats).clamp(0.0, 1.0)


def _prepare_candidate_classes(inference: LabelInferenceResult, limit: int) -> List[int]:
    """Return the primary class (predicted label) for sensitive feature analysis.

    If the predicted class lacks cached samples—for example due to data loading
    issues—the function gracefully falls back to other candidates to avoid
    breaking the downstream pipeline. The fallback still respects the provided
    ``limit``.
    """

    predicted = int(inference.predicted_class)
    samples = inference.sample_bank.get(predicted)
    if samples is not None and samples.numel() > 0:
        return [predicted]

    priority: List[int] = [predicted]
    priority.extend(int(cls) for cls in inference.second_stage_candidates)
    priority.extend(int(cls) for cls in inference.first_stage_candidates)

    unique: List[int] = []
    for cls in priority:
        if cls not in unique:
            unique.append(cls)
        if len(unique) >= limit:
            break
    return unique


def _extract_patch_bank(
    samples: torch.Tensor,
    mask: torch.Tensor,
    config: SensitiveFeatureConfig,
    *,
    stats: Tuple[Tuple[float, ...], Tuple[float, ...]] | None = None,
) -> Tuple[np.ndarray, List[dict]]:
    if samples.numel() == 0:
        return np.empty((0, config.patch_size, config.patch_size, max(1, samples.size(1)))), []

    denorm = _denormalize_samples(samples, stats)
    num_samples, channels, height, width = denorm.shape
    patch_size = min(config.patch_size, height, width)
    if patch_size <= 0:
        raise ValueError("Patch size must be positive")

    coords = torch.nonzero(mask > 0.0, as_tuple=False)
    if coords.numel() == 0:
        # Fallback to uniform sampling if the mask is empty.
        ys = torch.randint(0, height, (config.num_patches,))
        xs = torch.randint(0, width, (config.num_patches,))
        coords = torch.stack([ys, xs], dim=1)

    patches: List[np.ndarray] = []
    metadata: List[dict] = []

    for index in range(min(config.num_patches, coords.size(0))):
        center_y = int(coords[index, 0].item())
        center_x = int(coords[index, 1].item())
        sample_idx = index % num_samples

        top = max(0, min(center_y - patch_size // 2, height - patch_size))
        left = max(0, min(center_x - patch_size // 2, width - patch_size))
        patch_tensor = denorm[sample_idx, :, top : top + patch_size, left : left + patch_size]
        if patch_tensor.shape[1] != patch_size or patch_tensor.shape[2] != patch_size:
            continue

        patch_np = patch_tensor.permute(1, 2, 0).cpu().numpy()
        patches.append(patch_np)
        metadata.append({"sample_index": sample_idx, "top": top, "left": left})

    if not patches:
        return np.empty((0, patch_size, patch_size, channels)), []

    return np.stack(patches, axis=0), metadata


def _derive_texture_tokens(gabor_entries: List[List[dict]], dct_spectra: np.ndarray) -> List[str]:
    tokens: set[str] = set()
    for patch_index, entries in enumerate(gabor_entries):
        if not entries:
            continue
        dominant = max(entries, key=lambda item: item["mean_energy"])
        orientation = int(round(dominant["theta_deg"] / 15.0) * 15)
        token = f"orient_{orientation}_sigma_{dominant['sigma']:.2f}_freq_{dominant['frequency']:.2f}"
        tokens.add(token)
    if dct_spectra.size > 0:
        averaged = dct_spectra.mean(axis=0)
        if averaged.size > 0:
            peak_indices = np.argsort(averaged)[::-1][: min(5, averaged.size)]
            for idx in peak_indices:
                tokens.add(f"dct_peak_{int(idx)}")
    return sorted(tokens)


def run_sensitive_feature_inference(
    inference: LabelInferenceResult,
    *,
    output_root: Path,
    dataset: str,
    normalization_stats: Tuple[Tuple[float, ...], Tuple[float, ...]] | None,
    config: SensitiveFeatureConfig | None = None,
) -> Dict[str, object]:
    """Generate sensitive feature descriptors to guide diffusion-based reconstruction."""

    if config is None:
        config = SensitiveFeatureConfig()

    output_root.mkdir(parents=True, exist_ok=True)

    candidates = _prepare_candidate_classes(inference, config.max_classes)
    summary: Dict[str, object] = {
        "dataset": dataset,
        "stable_diffusion_model": config.stable_diffusion_model,
        "config": asdict(config),
        "classes": {},
    }

    for rank, cls in enumerate(candidates):
        cache = inference.heatmap_cache.get(cls)
        samples = inference.sample_bank.get(cls)
        if cache is None or samples is None or samples.numel() == 0:
            continue

        class_dir = output_root / f"CLASS_{cls:03d}"
        class_dir.mkdir(parents=True, exist_ok=True)

        saliency_before = cache["saliency_before"].float()
        saliency_after = cache["saliency_after"].float()
        gradcam_before = cache["gradcam_before"].float()
        gradcam_after = cache["gradcam_after"].float()

        delta_saliency = torch.clamp(saliency_before - saliency_after, min=0.0).mean(dim=0)
        delta_gradcam = torch.clamp(gradcam_before - gradcam_after, min=0.0).mean(dim=0)

        saliency_norm = _normalize_map(delta_saliency)
        gradcam_norm = _normalize_map(delta_gradcam)

        saliency_path = class_dir / "saliency_diff.png"
        gradcam_path = class_dir / "gradcam_diff.png"
        _save_grayscale_png(saliency_norm.cpu().numpy(), saliency_path)
        _save_grayscale_png(gradcam_norm.cpu().numpy(), gradcam_path)

        threshold = _quantile_threshold(saliency_norm, config.mask_quantile, config.mask_min_threshold)
        region_mask = (saliency_norm >= threshold).float()
        region_mask_path = class_dir / "region_mask.png"
        np.save(class_dir / "region_mask.npy", region_mask.cpu().numpy())
        _save_binary_mask(region_mask.cpu().numpy(), region_mask_path)

        edge_mask_bool = _build_edge_mask(region_mask.shape, config.edge_border_ratio)
        edge_candidate = torch.zeros_like(region_mask)
        edge_candidate[edge_mask_bool] = saliency_norm[edge_mask_bool]
        if edge_candidate.numel() > 0:
            edge_threshold = _quantile_threshold(edge_candidate, 0.9, config.mask_min_threshold)
        else:
            edge_threshold = config.mask_min_threshold
        edge_mask = (edge_candidate >= edge_threshold).float()
        edge_mask_path = class_dir / "edge_mask.png"
        np.save(class_dir / "edge_mask.npy", edge_mask.cpu().numpy())
        _save_binary_mask(edge_mask.cpu().numpy(), edge_mask_path)

        denorm_samples = _denormalize_samples(samples, normalization_stats)
        reference = denorm_samples[0].permute(1, 2, 0).cpu().numpy()
        canny_original = _compute_canny(reference, config.canny_low, config.canny_high)
        canny_mask = _compute_canny(region_mask.cpu().numpy(), config.canny_low, config.canny_high)

        canny_original_path = class_dir / "canny_original.png"
        canny_mask_path = class_dir / "canny_mask.png"
        _save_grayscale_png(canny_original, canny_original_path)
        _save_grayscale_png(canny_mask, canny_mask_path)

        hed_path = class_dir / "hed_candidate.png"
        _save_grayscale_png(edge_mask.cpu().numpy(), hed_path)

        patch_bank, patch_metadata = _extract_patch_bank(samples, region_mask, config, stats=normalization_stats)
        np.save(class_dir / "patch_bank.npy", patch_bank)

        dct_spectra: List[np.ndarray] = []
        gabor_entries: List[List[dict]] = []
        for index, patch in enumerate(patch_bank):
            dct_spectra.append(_compute_dct_spectrum(patch, config.dct_components))
            gray_patch = torch.from_numpy(patch.mean(axis=2)).float()
            gabor_entries.append(
                _compute_gabor_statistics(
                    gray_patch,
                    sigmas=config.gabor_sigmas,
                    frequencies=config.gabor_frequencies,
                    orientations=config.gabor_orientations,
                )
            )

        if dct_spectra:
            dct_array = np.stack(dct_spectra, axis=0)
        else:
            dct_array = np.empty((0, config.dct_components), dtype=np.float32)
        np.save(class_dir / "dct_spectra.npy", dct_array)

        gabor_path = class_dir / "gabor_stats.json"
        with gabor_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "patches": [
                        {
                            "index": index,
                            "metadata": patch_metadata[index],
                            "entries": gabor_entries[index],
                        }
                        for index in range(len(gabor_entries))
                    ],
                    "sigmas": list(map(float, config.gabor_sigmas)),
                    "frequencies": list(map(float, config.gabor_frequencies)),
                    "orientations": int(config.gabor_orientations),
                },
                handle,
                indent=2,
            )

        flattened_patches = patch_bank.reshape(patch_bank.shape[0], -1) if patch_bank.size else np.empty((0, 1))
        centroids, labels = _simple_kmeans(flattened_patches.astype(np.float32), config.num_prototypes)
        texture_tokens = _derive_texture_tokens(gabor_entries, dct_array)

        class_summary = {
            "rank": rank,
            "class_id": int(cls),
            "predicted": int(inference.predicted_class),
            "threshold": threshold,
            "edge_threshold": edge_threshold,
            "num_patches": int(patch_bank.shape[0]),
            "patch_size": int(config.patch_size),
            "prototype_count": int(centroids.shape[0]),
            "texture_tokens": texture_tokens,
            "controlnet": {
                "canny_original": str(canny_original_path),
                "canny_mask": str(canny_mask_path),
                "hed_candidate": str(hed_path),
                "region_mask": str(region_mask_path),
                "edge_mask": str(edge_mask_path),
            },
            "lora_materials": {
                "patch_bank": str(class_dir / "patch_bank.npy"),
                "dct_spectra": str(class_dir / "dct_spectra.npy"),
                "gabor_stats": str(gabor_path),
            },
            "saliency_diff": str(saliency_path),
            "gradcam_diff": str(gradcam_path),
            "kmeans_labels": labels.tolist(),
        }

        with (class_dir / "sfi_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(class_summary, handle, indent=2)

        summary["classes"][str(cls)] = class_summary

    with (output_root / "sfi_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


__all__ = ["SensitiveFeatureConfig", "run_sensitive_feature_inference"]
