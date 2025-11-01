"""Label inference attack leveraging differences between models."""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..utils.metrics import (
    accuracy_dict,
    confusion_matrix,
    per_class_accuracy,
)


@dataclass
class SensitiveFeature:
    """A sensitive feature inferred from forensic signals."""

    name: str
    score: float
    source: str

    def to_dict(self) -> dict[str, float | str]:
        return {"name": self.name, "score": float(self.score), "source": self.source}


@dataclass
class LabelInferenceResult:
    predicted_class: int
    score_vector: torch.Tensor
    per_class_before: torch.Tensor
    per_class_after: torch.Tensor
    confusion_before: torch.Tensor
    confusion_after: torch.Tensor
    positive_rate_before: torch.Tensor
    positive_rate_after: torch.Tensor
    positive_rate_drop: torch.Tensor
    first_stage_candidates: Sequence[int]
    second_stage_candidates: Sequence[int]
    heatmap_details: dict[int, dict[str, float]]
    similarity_scores: dict[int, float]
    heatmap_cache: dict[int, dict[str, torch.Tensor]]
    sample_bank: dict[int, torch.Tensor]
    ground_truth: Optional[int] = None
    gradient_before: OrderedDict[str, torch.Tensor] | None = None
    gradient_after: OrderedDict[str, torch.Tensor] | None = None
    gradient_delta: OrderedDict[str, torch.Tensor] | None = None
    accuracy_delta: torch.Tensor | None = None
    confusion_delta: torch.Tensor | None = None
    confidence_before: torch.Tensor | None = None
    confidence_after: torch.Tensor | None = None
    confidence_delta: torch.Tensor | None = None
    weight_delta: torch.Tensor | None = None
    bias_delta: torch.Tensor | None = None
    candidate_details: dict[int, dict[str, float]] | None = None
    sensitive_features: Sequence[SensitiveFeature] | None = None

    def to_dict(self) -> dict:
        return {
            "predicted_class": int(self.predicted_class),
            "score_vector": self.score_vector.tolist(),
            "per_class_before": accuracy_dict(self.per_class_before),
            "per_class_after": accuracy_dict(self.per_class_after),
            "ground_truth": None if self.ground_truth is None else int(self.ground_truth),
            "accuracy_delta": None
            if self.accuracy_delta is None
            else self.accuracy_delta.tolist(),
            "confusion_delta": None
            if self.confusion_delta is None
            else self.confusion_delta.tolist(),
            "confidence_before": None
            if self.confidence_before is None
            else self.confidence_before.tolist(),
            "confidence_after": None
            if self.confidence_after is None
            else self.confidence_after.tolist(),
            "confidence_delta": None
            if self.confidence_delta is None
            else self.confidence_delta.tolist(),
            "weight_delta": None
            if self.weight_delta is None
            else self.weight_delta.tolist(),
            "bias_delta": None
            if self.bias_delta is None
            else self.bias_delta.tolist(),
            "candidate_details": self.candidate_details,
            "sensitive_features": None
            if self.sensitive_features is None
            else [feature.to_dict() for feature in self.sensitive_features],
            "positive_rate_before": self.positive_rate_before.tolist(),
            "positive_rate_after": self.positive_rate_after.tolist(),
            "positive_rate_drop": self.positive_rate_drop.tolist(),
            "first_stage_candidates": [int(item) for item in self.first_stage_candidates],
            "second_stage_candidates": [int(item) for item in self.second_stage_candidates],
            "heatmap_details": {
                int(cls): {key: float(value) for key, value in details.items()}
                for cls, details in self.heatmap_details.items()
            },
            "similarity_scores": {int(cls): float(score) for cls, score in self.similarity_scores.items()},
        }


ScoreTransform = Callable[[torch.Tensor], torch.Tensor]

_SOBEL_X = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
_SOBEL_Y = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

LOGGER = logging.getLogger(__name__)


def _per_class_confidence(
    model: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return average softmax confidence for the ground-truth class per label."""

    model.eval()
    totals = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            for cls in range(num_classes):
                mask = targets == cls
                if mask.any():
                    totals[cls] += probs[mask, cls].sum()
                    counts[cls] += mask.sum()

    confidences = torch.zeros_like(totals)
    mask = counts > 0
    confidences[mask] = totals[mask] / counts[mask]
    return confidences


def _collect_class_samples(
    dataloader: DataLoader,
    candidates: Sequence[int],
    max_samples: int,
) -> dict[int, torch.Tensor]:
    """Collect up to ``max_samples`` 测试样本用于指定类别的热力图分析。"""

    remaining = {int(cls): max_samples for cls in candidates}
    storage: dict[int, list[torch.Tensor]] = {int(cls): [] for cls in candidates}

    if max_samples <= 0 or not remaining:
        return {cls: torch.empty(0) for cls in storage}

    for inputs, targets in dataloader:
        for cls in list(remaining.keys()):
            if remaining[cls] <= 0:
                continue
            mask = targets == cls
            if not torch.any(mask):
                continue
            selected = inputs[mask].detach().cpu()
            needed = min(remaining[cls], selected.size(0))
            storage[cls].append(selected[:needed])
            remaining[cls] -= needed
        if all(count <= 0 for count in remaining.values()):
            break

    return {
        cls: torch.cat(tensors, dim=0) if tensors else torch.empty(0)
        for cls, tensors in storage.items()
    }


def _find_last_conv_module(model: nn.Module) -> nn.Conv2d | None:
    """Heuristically locate the最后一层卷积用于 Grad-CAM。"""

    last_conv: nn.Conv2d | None = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _gradcam_heatmap(
    model: nn.Module,
    inputs: torch.Tensor,
    target_indices: torch.Tensor,
    conv_layer: nn.Conv2d,
) -> torch.Tensor:
    """Compute Grad-CAM heatmaps for ``inputs`` focusing on ``target_indices``."""

    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_: nn.Module, __, output: torch.Tensor) -> None:
        activations.append(output.detach())

    def backward_hook(_: nn.Module, __, grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0].detach())

    handle_fwd = conv_layer.register_forward_hook(forward_hook)
    handle_bwd = conv_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(inputs.requires_grad_(True))
        selected = logits.gather(1, target_indices.view(-1, 1))
        selected.sum().backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks未捕获到激活或梯度")

        acts = activations[0]
        grads = gradients[0]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=inputs.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        return _normalize_heatmap(cam)
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _center_distance_metric(heatmaps: torch.Tensor) -> torch.Tensor:
    """Measure attention中心到图像中心的归一化距离。"""

    batch, height, width = heatmaps.shape
    device = heatmaps.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=heatmaps.dtype),
        torch.arange(width, device=device, dtype=heatmaps.dtype),
        indexing="ij",
    )
    weights = heatmaps.clamp(min=0)
    totals = weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    center_y = (weights * ys).sum(dim=(1, 2), keepdim=True) / totals
    center_x = (weights * xs).sum(dim=(1, 2), keepdim=True) / totals
    target_y = (height - 1) / 2.0
    target_x = (width - 1) / 2.0
    distance = torch.sqrt((center_y - target_y) ** 2 + (center_x - target_x) ** 2).view(-1)
    max_distance = torch.sqrt(torch.tensor(height**2 + width**2, dtype=heatmaps.dtype, device=device))
    return (distance / max_distance).clamp(0, 1)


def _edge_focus_metric(heatmaps: torch.Tensor, border_ratio: float) -> torch.Tensor:
    """Estimate模型对边缘纹理的关注强度。"""

    if heatmaps.numel() == 0:
        return torch.zeros(0, device=heatmaps.device, dtype=heatmaps.dtype)

    kernel_x = _SOBEL_X.to(device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 3, 3)
    kernel_y = _SOBEL_Y.to(device=heatmaps.device, dtype=heatmaps.dtype).view(1, 1, 3, 3)
    gradients_x = F.conv2d(heatmaps.unsqueeze(1), kernel_x, padding=1)
    gradients_y = F.conv2d(heatmaps.unsqueeze(1), kernel_y, padding=1)
    magnitude = torch.sqrt(gradients_x.pow(2) + gradients_y.pow(2) + 1e-12)

    _, height, width = heatmaps.shape
    border_h = max(1, int(height * border_ratio))
    border_w = max(1, int(width * border_ratio))

    mask = torch.zeros_like(heatmaps, dtype=torch.bool)
    mask[:, :border_h, :] = True
    mask[:, -border_h:, :] = True
    mask[:, :, :border_w] = True
    mask[:, :, -border_w:] = True

    scores = torch.zeros(heatmaps.size(0), device=heatmaps.device, dtype=heatmaps.dtype)
    for idx in range(heatmaps.size(0)):
        selected = magnitude[idx][mask[idx]]
        if selected.numel() > 0:
            scores[idx] = selected.mean()
    return scores


def _min_max_normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize张量到 [0, 1] 范围，若范围为零则返回全零。"""

    if values.numel() == 0:
        return values
    min_val = values.min()
    max_val = values.max()
    if torch.isclose(max_val, min_val):
        return torch.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def _select_high_level_parameter(
    before: nn.Module,
    after: nn.Module,
) -> tuple[str | None, torch.Tensor | None]:
    """选择参数变化最大的高层权重 (维度≥2)。"""

    after_params = dict(after.named_parameters())
    chosen_name: str | None = None
    chosen_delta: torch.Tensor | None = None
    best_norm = -float("inf")

    for name, param_before in before.named_parameters():
        if name not in after_params:
            continue
        if param_before.dim() < 2:
            continue
        delta = param_before.detach() - after_params[name].detach().to(param_before.device)
        norm = delta.norm().item()
        if norm > best_norm:
            best_norm = norm
            chosen_name = name
            chosen_delta = delta

    return chosen_name, chosen_delta


def _average_parameter_gradient(
    model: nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    parameter_name: str,
) -> torch.Tensor:
    """Average梯度 w.r.t. ``parameter_name`` for指定类别。"""

    params = dict(model.named_parameters())
    if parameter_name not in params:
        raise KeyError(f"Parameter {parameter_name} not found in model")
    param = params[parameter_name]
    grads: list[torch.Tensor] = []
    for sample in inputs:
        sample = sample.unsqueeze(0)
        model.zero_grad(set_to_none=True)
        logits = model(sample)
        target = logits[:, target_class]
        grad = torch.autograd.grad(target.sum(), param, retain_graph=False, create_graph=False, allow_unused=True)[0]
        if grad is None:
            grad = torch.zeros_like(param)
        grads.append(grad.detach())
    return torch.stack(grads, dim=0).mean(dim=0)


def _normalize_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    """Scale saliency maps to the [0, 1] range per sample."""

    flat = tensor.view(tensor.size(0), -1)
    mins = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
    tensor = tensor - mins
    flat = tensor.view(tensor.size(0), -1)
    maxs = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
    safe_maxs = torch.where(maxs > 0, maxs, torch.ones_like(maxs))
    return tensor / safe_maxs


def _compute_saliency_map(
    model: nn.Module,
    inputs: torch.Tensor,
    target_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute gradient-based saliency maps for ``inputs``.

    如果 ``target_indices`` 提供，则对指定类别的对数值求导；否则默认选择模型预测类别。
    """

    model.zero_grad(set_to_none=True)
    cloned = inputs.detach().clone().requires_grad_(True)
    outputs = model(cloned)
    if target_indices is None:
        preds = outputs.argmax(dim=1)
    else:
        preds = target_indices.view(-1).to(outputs.device)
    selected = outputs.gather(1, preds.unsqueeze(1)).sum()
    grads = torch.autograd.grad(selected, cloned, retain_graph=False, create_graph=False)[0]
    saliency = grads.abs().amax(dim=1)
    return _normalize_heatmap(saliency)


def infer_forgotten_label(
    before: nn.Module,
    after: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    transform: Optional[ScoreTransform] = None,
    ground_truth: Optional[int] = None,
    *,
    heatmap_samples: int = 10,
    heatmap_border_ratio: float = 0.2,
) -> LabelInferenceResult:
    """按照多阶段法分析遗忘类别并输出详尽诊断信息。"""

    before = before.to(device)
    after = after.to(device)
    before.eval()
    after.eval()

    per_before = per_class_accuracy(before, dataloader, num_classes, device)
    per_after = per_class_accuracy(after, dataloader, num_classes, device)
    confusion_before = confusion_matrix(before, dataloader, num_classes, device)
    confusion_after = confusion_matrix(after, dataloader, num_classes, device)
    confusion_delta = confusion_before - confusion_after
    accuracy_delta = per_before - per_after

    positive_before = _per_class_confidence(before, dataloader, num_classes, device)
    positive_after = _per_class_confidence(after, dataloader, num_classes, device)
    positive_drop = positive_before - positive_after

    LOGGER.info("标签推理阶段：各类别阳性率统计：")
    for cls in range(num_classes):
        LOGGER.info(
            "  类别 %d -> 遗忘前阳性率 %.4f, 遗忘后阳性率 %.4f, 差值 %.4f",
            cls,
            positive_before[cls].item(),
            positive_after[cls].item(),
            positive_drop[cls].item(),
        )

    if float(positive_drop.abs().max().item()) == 0.0:
        raise RuntimeError("阳性率在遗忘前后完全一致，无法判断遗忘类别。")

    candidate_count = max(1, math.ceil(num_classes * 0.6))
    sorted_indices = torch.argsort(positive_drop, descending=True)
    first_stage_candidates = sorted_indices[:candidate_count].tolist()
    LOGGER.info(
        "第一阶段候选类别（取前60%%，共 %d 个）：%s",
        len(first_stage_candidates),
        first_stage_candidates,
    )

    sample_bank = _collect_class_samples(dataloader, first_stage_candidates, heatmap_samples)
    for cls in first_stage_candidates:
        LOGGER.info("  类别 %d -> 采样测试样本 %d 个用于热力图分析", cls, sample_bank.get(cls, torch.empty(0)).size(0))

    conv_before = _find_last_conv_module(before)
    conv_after = _find_last_conv_module(after)
    if conv_before is None or conv_after is None:
        LOGGER.warning("未检测到卷积层，Grad-CAM 将退化为基于输入梯度的热力图。")

    heatmap_details: dict[int, dict[str, float]] = {}
    heatmap_cache: dict[int, dict[str, torch.Tensor]] = {}
    candidate_details: dict[int, dict[str, float]] = {}

    normalized_positive_values = []
    normalized_center_values = []
    normalized_edge_values = []
    valid_classes: list[int] = []

    for cls in first_stage_candidates:
        samples = sample_bank.get(cls)
        if samples is None or samples.numel() == 0:
            LOGGER.warning("  类别 %d 在测试集中样本不足，跳过热力图差异分析。", cls)
            heatmap_details[cls] = {
                "samples": 0.0,
                "center_shift_mean": 0.0,
                "edge_drop_mean": 0.0,
                "positive_rate_drop": float(positive_drop[cls].item()),
            }
            continue

        inputs = samples.to(device)
        target_indices = torch.full((inputs.size(0),), cls, device=device, dtype=torch.long)

        with torch.enable_grad():
            if conv_before is not None and conv_after is not None:
                gradcam_before = _gradcam_heatmap(before, inputs.clone(), target_indices, conv_before)
                gradcam_after = _gradcam_heatmap(after, inputs.clone(), target_indices, conv_after)
            else:
                gradcam_before = _compute_saliency_map(before, inputs, target_indices)
                gradcam_after = _compute_saliency_map(after, inputs, target_indices)
            saliency_before = _compute_saliency_map(before, inputs, target_indices)
            saliency_after = _compute_saliency_map(after, inputs, target_indices)

        center_before = _center_distance_metric(gradcam_before)
        center_after = _center_distance_metric(gradcam_after)
        center_diff_mean = float((center_after - center_before).mean().item())
        center_shift_positive = max(0.0, center_diff_mean)

        edge_before = _edge_focus_metric(saliency_before, heatmap_border_ratio)
        edge_after = _edge_focus_metric(saliency_after, heatmap_border_ratio)
        edge_drop_mean = float((edge_before - edge_after).mean().item())
        edge_drop_positive = max(0.0, edge_drop_mean)

        positive_component = max(0.0, float(positive_drop[cls].item()))

        LOGGER.info(
            "  类别 %d -> 热力图中心偏移均值 %.4f, 边缘关注下降均值 %.4f, 阳性率下降 %.4f",
            cls,
            center_diff_mean,
            edge_drop_mean,
            positive_component,
        )

        heatmap_details[cls] = {
            "samples": float(inputs.size(0)),
            "center_distance_before": float(center_before.mean().item()),
            "center_distance_after": float(center_after.mean().item()),
            "center_shift_mean": center_diff_mean,
            "edge_focus_before": float(edge_before.mean().item()),
            "edge_focus_after": float(edge_after.mean().item()),
            "edge_drop_mean": edge_drop_mean,
            "positive_rate_drop": positive_component,
        }

        heatmap_cache[cls] = {
            "gradcam_before": gradcam_before.detach().cpu(),
            "gradcam_after": gradcam_after.detach().cpu(),
            "saliency_before": saliency_before.detach().cpu(),
            "saliency_after": saliency_after.detach().cpu(),
        }

        valid_classes.append(cls)
        normalized_positive_values.append(positive_component)
        normalized_center_values.append(center_shift_positive)
        normalized_edge_values.append(edge_drop_positive)

    if not valid_classes:
        raise RuntimeError("第一阶段候选类别均缺乏样本，无法继续标签推理。")

    positive_tensor = torch.tensor(normalized_positive_values, dtype=torch.float32)
    center_tensor = torch.tensor(normalized_center_values, dtype=torch.float32)
    edge_tensor = torch.tensor(normalized_edge_values, dtype=torch.float32)

    positive_norm = _min_max_normalize(positive_tensor)
    center_norm = _min_max_normalize(center_tensor)
    edge_norm = _min_max_normalize(edge_tensor)

    combined_tensor = positive_norm + 0.5 * center_norm + 0.5 * edge_norm

    combined_scores: dict[int, float] = {}
    for index, cls in enumerate(valid_classes):
        combined = float(combined_tensor[index].item())
        combined_scores[cls] = combined
        candidate_details[cls] = {
            **heatmap_details.get(cls, {}),
            "combined_score": combined,
        }
        LOGGER.info(
            "  类别 %d -> 归一化指标：阳性率 %.4f, 中心偏移 %.4f, 边缘关注 %.4f, 综合得分 %.4f",
            cls,
            positive_norm[index].item(),
            center_norm[index].item(),
            edge_norm[index].item(),
            combined,
        )

    second_stage_count = min(3, len(valid_classes))
    second_stage_candidates = sorted(
        valid_classes,
        key=lambda cls: combined_scores.get(cls, 0.0),
        reverse=True,
    )[:second_stage_count]

    LOGGER.info("第二阶段候选类别（综合得分最高的 %d 个）：%s", second_stage_count, second_stage_candidates)

    parameter_name, parameter_delta = _select_high_level_parameter(before, after)
    similarity_scores: dict[int, float] = {}

    score_vector = torch.full((num_classes,), fill_value=-1e9, device=device)

    if parameter_name is None or parameter_delta is None:
        LOGGER.warning("未找到高层参数差异，直接依据综合热力图得分进行判断。")
        for cls in valid_classes:
            score_vector[cls] = combined_scores.get(cls, 0.0)
            similarity_scores[cls] = combined_scores.get(cls, 0.0)
    else:
        LOGGER.info(
            "选择高层参数 %s 进行梯度相似度分析 (Δ范数=%.4f)",
            parameter_name,
            float(parameter_delta.norm().item()),
        )
        for cls in second_stage_candidates:
            samples = sample_bank.get(cls)
            if samples is None or samples.numel() == 0:
                LOGGER.warning("类别 %d 缺少样本，无法计算梯度相似度。", cls)
                similarity_scores[cls] = float("-inf")
                continue
            inputs = samples[:heatmap_samples].to(device)
            avg_before = _average_parameter_gradient(before, inputs, cls, parameter_name)
            avg_after = _average_parameter_gradient(after, inputs, cls, parameter_name)
            grad_delta = (avg_before - avg_after).to(parameter_delta.device)
            grad_flat = grad_delta.view(-1)
            delta_flat = parameter_delta.view(-1)
            if grad_flat.norm().item() == 0.0:
                similarity = 0.0
            else:
                similarity = float(
                    F.cosine_similarity(delta_flat.unsqueeze(0), grad_flat.unsqueeze(0), dim=1).item()
                )
            similarity_scores[cls] = similarity
            score_vector[cls] = similarity
            LOGGER.info("  类别 %d -> 参数梯度相似度 %.4f", cls, similarity)

        for cls in valid_classes:
            if score_vector[cls].item() == -1e9:
                # 对未进入第二阶段的类别保留综合热力图得分以备惩罚机制使用
                score_vector[cls] = combined_scores.get(cls, 0.0)
                similarity_scores.setdefault(cls, combined_scores.get(cls, 0.0))

    if transform is not None:
        score_vector = transform(score_vector)

    predicted = int(torch.argmax(score_vector).item())

    LOGGER.info(
        "标签推理综合得分: %s",
        {cls: float(score_vector[cls].item()) for cls in valid_classes},
    )

    # 构造敏感特征描述
    sensitive_features: list[SensitiveFeature] = []
    if predicted in heatmap_details:
        details = heatmap_details[predicted]
        sensitive_features.append(
            SensitiveFeature(
                name="positive_rate_drop",
                score=details.get("positive_rate_drop", 0.0),
                source="confidence",
            )
        )
        sensitive_features.append(
            SensitiveFeature(
                name="center_shift_mean",
                score=details.get("center_shift_mean", 0.0),
                source="heatmap_center",
            )
        )
        sensitive_features.append(
            SensitiveFeature(
                name="edge_drop_mean",
                score=details.get("edge_drop_mean", 0.0),
                source="heatmap_edge",
            )
        )
    if predicted in similarity_scores:
        sensitive_features.append(
            SensitiveFeature(
                name="parameter_similarity",
                score=similarity_scores[predicted],
                source="parameter",
            )
        )

    return LabelInferenceResult(
        predicted_class=predicted,
        score_vector=score_vector.detach().cpu(),
        per_class_before=per_before.detach().cpu(),
        per_class_after=per_after.detach().cpu(),
        confusion_before=confusion_before.detach().cpu(),
        confusion_after=confusion_after.detach().cpu(),
        positive_rate_before=positive_before.detach().cpu(),
        positive_rate_after=positive_after.detach().cpu(),
        positive_rate_drop=positive_drop.detach().cpu(),
        first_stage_candidates=tuple(first_stage_candidates),
        second_stage_candidates=tuple(second_stage_candidates),
        heatmap_details={cls: {key: float(value) for key, value in details.items()} for cls, details in heatmap_details.items()},
        similarity_scores={cls: float(score) for cls, score in similarity_scores.items()},
        heatmap_cache=heatmap_cache,
        sample_bank={cls: tensor.detach().cpu() for cls, tensor in sample_bank.items()},
        ground_truth=ground_truth,
        gradient_before=None,
        gradient_after=None,
        gradient_delta=None,
        accuracy_delta=accuracy_delta.detach().cpu(),
        confusion_delta=confusion_delta.detach().cpu(),
        confidence_before=positive_before.detach().cpu(),
        confidence_after=positive_after.detach().cpu(),
        confidence_delta=positive_drop.detach().cpu(),
        weight_delta=None,
        bias_delta=None,
        candidate_details={cls: {key: float(value) for key, value in details.items()} for cls, details in candidate_details.items()},
        sensitive_features=tuple(sensitive_features),
    )

__all__ = ["LabelInferenceResult", "SensitiveFeature", "infer_forgotten_label"]
