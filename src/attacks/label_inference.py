"""Label inference attack leveraging differences between models."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..utils.metrics import (
    accuracy_dict,
    average_parameter_gradients,
    confusion_matrix,
    gradient_delta_norms,
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
    gradient_class_scores: torch.Tensor | None = None
    saliency_delta: torch.Tensor | None = None
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
            "gradient_delta_norms": None
            if self.gradient_delta is None
            else gradient_delta_norms(self.gradient_before or {}, self.gradient_after or {}),
            "gradient_class_scores": None
            if self.gradient_class_scores is None
            else self.gradient_class_scores.tolist(),
            "saliency_delta": None
            if self.saliency_delta is None
            else self.saliency_delta.tolist(),
            "candidate_details": self.candidate_details,
            "sensitive_features": None
            if self.sensitive_features is None
            else [feature.to_dict() for feature in self.sensitive_features],
        }


ScoreTransform = Callable[[torch.Tensor], torch.Tensor]


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise ``tensor`` to the [-1, 1] range handling all-zero tensors."""

    max_abs = tensor.abs().max()
    if torch.isnan(max_abs) or max_abs == 0:
        return torch.zeros_like(tensor)
    return tensor / max_abs


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


def _find_classifier_parameters(
    model: nn.Module, num_classes: int
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return the final linear layer parameters heuristically."""

    weight: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    for name, param in model.named_parameters():
        if param.dim() == 2 and param.size(0) == num_classes:
            weight = param
        elif param.dim() == 1 and param.size(0) == num_classes and name.endswith("bias"):
            bias = param
    return weight, bias


def _classifier_norm_drop(
    before: nn.Module,
    after: nn.Module,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return L2 norm decreases for classifier weights/biases per class."""

    weight_before, bias_before = _find_classifier_parameters(before, num_classes)
    weight_after, bias_after = _find_classifier_parameters(after, num_classes)
    if weight_before is None or weight_after is None:
        return torch.zeros(num_classes, device=device), None

    with torch.no_grad():
        weight_before = weight_before.detach().to(device)
        weight_after = weight_after.detach().to(device)
        before_norm = torch.linalg.vector_norm(weight_before, dim=1)
        after_norm = torch.linalg.vector_norm(weight_after, dim=1)
        weight_delta = before_norm - after_norm

        bias_delta: torch.Tensor | None = None
        if bias_before is not None and bias_after is not None:
            bias_before = bias_before.detach().to(device)
            bias_after = bias_after.detach().to(device)
            bias_delta = bias_before.abs() - bias_after.abs()

    return weight_delta, bias_delta


def _gradient_class_scores(
    gradient_delta: OrderedDict[str, torch.Tensor] | None,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Aggregate gradient delta norms for classifier-aligned parameters."""

    if gradient_delta is None:
        return None

    scores = torch.zeros(num_classes, device=device)
    for tensor in gradient_delta.values():
        grad = tensor.to(device)
        if grad.dim() >= 1 and grad.size(0) == num_classes:
            if grad.dim() == 1:
                scores += grad.abs()
            else:
                dims = tuple(range(1, grad.dim()))
                scores += grad.norm(p=2, dim=dims)
    return scores


def _normalize_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    """Scale saliency maps to the [0, 1] range per sample."""

    flat = tensor.view(tensor.size(0), -1)
    mins = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
    tensor = tensor - mins
    flat = tensor.view(tensor.size(0), -1)
    maxs = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
    safe_maxs = torch.where(maxs > 0, maxs, torch.ones_like(maxs))
    return tensor / safe_maxs


def _compute_saliency_map(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Compute gradient-based saliency maps for ``inputs``."""

    model.zero_grad(set_to_none=True)
    cloned = inputs.detach().clone().requires_grad_(True)
    outputs = model(cloned)
    preds = outputs.argmax(dim=1)
    selected = outputs.gather(1, preds.unsqueeze(1)).sum()
    grads = torch.autograd.grad(selected, cloned, retain_graph=False, create_graph=False)[0]
    saliency = grads.abs().amax(dim=1)
    return _normalize_heatmap(saliency)


def _saliency_delta_scores(
    before: nn.Module,
    after: nn.Module,
    dataloader: DataLoader,
    candidates: Sequence[int],
    num_classes: int,
    device: torch.device,
    *,
    max_batches: int | None = 1,
    max_samples: int | None = 64,
) -> torch.Tensor | None:
    """Aggregate saliency difference per class for selected candidates."""

    if not candidates:
        return None

    candidate_tensor = torch.tensor(sorted(set(int(cls) for cls in candidates)), device=device)
    scores = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)

    before.eval()
    after.eval()

    processed_batches = 0
    processed_samples = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = torch.isin(targets, candidate_tensor)
        if not torch.any(mask):
            continue

        subset_inputs = inputs[mask]
        subset_targets = targets[mask]

        with torch.enable_grad():
            before_maps = _compute_saliency_map(before, subset_inputs)
            after_maps = _compute_saliency_map(after, subset_inputs)

        diff = (before_maps - after_maps).abs().view(subset_inputs.size(0), -1).mean(dim=1)
        for index, cls in enumerate(subset_targets):
            scores[cls] += diff[index]
            counts[cls] += 1

        processed_batches += 1
        processed_samples += subset_inputs.size(0)
        if max_batches is not None and processed_batches >= max_batches:
            break
        if max_samples is not None and processed_samples >= max_samples:
            break

    mask = counts > 0
    if not torch.any(mask):
        return None
    scores[mask] = scores[mask] / counts[mask]
    return scores


def infer_forgotten_label(
    before: nn.Module,
    after: nn.Module,
    dataloader: DataLoader,
    num_classes: int,
    device: torch.device,
    transform: Optional[ScoreTransform] = None,
    ground_truth: Optional[int] = None,
    *,
    gradient_batches: int | None = 2,
    gradient_filter: Optional[Sequence[str]] = None,
    saliency_batches: int | None = 1,
    saliency_max_samples: int | None = 64,
) -> LabelInferenceResult:
    """Predict the removed class using multi-stage forensic heuristics."""
    before = before.to(device)
    after = after.to(device)

    per_before = per_class_accuracy(before, dataloader, num_classes, device)
    per_after = per_class_accuracy(after, dataloader, num_classes, device)
    confusion_before = confusion_matrix(before, dataloader, num_classes, device)
    confusion_after = confusion_matrix(after, dataloader, num_classes, device)
    confusion_delta = confusion_before - confusion_after

    accuracy_delta = per_before - per_after

    confidence_before = _per_class_confidence(before, dataloader, num_classes, device)
    confidence_after = _per_class_confidence(after, dataloader, num_classes, device)
    confidence_delta = confidence_before - confidence_after

    weight_delta, bias_delta = _classifier_norm_drop(before, after, num_classes, device)
    if bias_delta is None:
        bias_delta = torch.zeros_like(weight_delta)

    candidate_count = max(1, math.ceil(num_classes * 0.5))
    topk = torch.topk(accuracy_delta, k=candidate_count)
    accuracy_threshold = 0.10
    max_drop_value, max_drop_index = torch.max(accuracy_delta, dim=0)
    candidate_indices = set(topk.indices.tolist())
    if max_drop_value >= accuracy_threshold:
        candidate_indices.add(int(max_drop_index.item()))
    candidates = sorted(candidate_indices)

    row_totals_before = confusion_before.sum(dim=1).clamp_min(1.0)
    row_totals_after = confusion_after.sum(dim=1).clamp_min(1.0)
    diag_before = torch.diagonal(confusion_before, 0)
    diag_after = torch.diagonal(confusion_after, 0)
    diag_drop = (diag_before / row_totals_before) - (diag_after / row_totals_after)

    off_before = confusion_before - torch.diag_embed(diag_before)
    off_after = confusion_after - torch.diag_embed(diag_after)
    off_increase = (off_after.sum(dim=1) / row_totals_after) - (
            off_before.sum(dim=1) / row_totals_before
    )

    gradient_delta = None
    if gradient_batches is not None and gradient_batches > 0:
        predicate: Callable[[str, nn.Parameter], bool] | None = None
        if gradient_filter:
            filters = tuple(gradient_filter)

        def predicate(name: str, _: nn.Parameter) -> bool:
            return any(key in name for key in filters)

        gradients_before = average_parameter_gradients(
            before,
            dataloader,
            device,
            max_batches=gradient_batches,
            predicate=predicate,
        )
        gradients_after = average_parameter_gradients(
            after,
            dataloader,
            device,
            max_batches=gradient_batches,
            predicate=predicate,
        )

        gradient_delta = OrderedDict()
        for name, before_grad in gradients_before.items():
            if name not in gradients_after:
                continue
            gradient_delta[name] = before_grad - gradients_after[name]
    else:
        gradients_before = None
        gradients_after = None

    gradient_scores = _gradient_class_scores(gradient_delta, num_classes, device)

    saliency_scores = (
        _saliency_delta_scores(
            before,
            after,
            dataloader,
            candidates,
            num_classes,
            device,
            max_batches=saliency_batches,
            max_samples=saliency_max_samples,
        )
        if saliency_batches is not None and saliency_batches > 0
        else None
    )

    components: OrderedDict[str, torch.Tensor] = OrderedDict()
    components["accuracy"] = accuracy_delta.clamp(min=0)
    components["confusion_diag"] = diag_drop.clamp(min=0)
    components["confusion_off"] = off_increase.clamp(min=0)
    weight_component = weight_delta.clamp(min=0) + bias_delta.clamp(min=0)
    components["classifier_weight"] = weight_component
    components["confidence"] = confidence_delta.clamp(min=0)
    if gradient_scores is not None:
        components["gradient"] = gradient_scores.clamp(min=0)
    if saliency_scores is not None:
        components["saliency"] = saliency_scores.clamp(min=0)

    normalized_components = OrderedDict(
        (name, _normalize_tensor(tensor)) for name, tensor in components.items()
    )

    score_vector = torch.full_like(accuracy_delta, fill_value=-1e9)
    if candidates:
        candidate_tensor = torch.tensor(candidates, device=accuracy_delta.device, dtype=torch.long)
        stacked = torch.stack(list(normalized_components.values())) if normalized_components else None
        if stacked is not None and stacked.numel() > 0:
            fused = stacked[:, candidate_tensor].mean(dim=0)
            score_vector[candidate_tensor] = fused

    if transform is not None:
        score_vector = transform(score_vector)

    predicted = int(torch.argmax(score_vector).item())

    candidate_details: dict[int, dict[str, float]] | None = None
    if candidates:
        candidate_details = {}
        for cls in candidates:
            details = {
                "fusion_score": float(score_vector[cls].item()),
                "accuracy_delta": float(accuracy_delta[cls].item()),
                "confusion_diag_drop": float(diag_drop[cls].item()),
                "confusion_off_increase": float(off_increase[cls].item()),
                "weight_delta": float(weight_delta[cls].item()),
                "bias_delta": float(bias_delta[cls].item()),
                "confidence_delta": float(confidence_delta[cls].item()),
            }
            if gradient_scores is not None:
                details["gradient_delta"] = float(gradient_scores[cls].item())
            if saliency_scores is not None:
                details["saliency_delta"] = float(saliency_scores[cls].item())
            candidate_details[int(cls)] = details

    return LabelInferenceResult(
        predicted_class=predicted,
        score_vector=score_vector.detach().cpu(),
        per_class_before=per_before.detach().cpu(),
        per_class_after=per_after.detach().cpu(),
        confusion_before=confusion_before.detach().cpu(),
        confusion_after=confusion_after.detach().cpu(),
        ground_truth=ground_truth,
        gradient_before=gradients_before,
        gradient_after=gradients_after,
        gradient_delta=gradient_delta,
        accuracy_delta=accuracy_delta.detach().cpu(),
        confusion_delta=confusion_delta.detach().cpu(),
        confidence_before=confidence_before.detach().cpu(),
        confidence_after=confidence_after.detach().cpu(),
        confidence_delta=confidence_delta.detach().cpu(),
        weight_delta=weight_delta.detach().cpu(),
        bias_delta=bias_delta.detach().cpu(),
        gradient_class_scores=None if gradient_scores is None else gradient_scores.detach().cpu(),
        saliency_delta=None if saliency_scores is None else saliency_scores.detach().cpu(),
        candidate_details=candidate_details,
        sensitive_features=_infer_sensitive_features(
            predicted=predicted,
            accuracy_delta=accuracy_delta,
            confidence_delta=confidence_delta,
            weight_delta=weight_delta,
            bias_delta=bias_delta,
            gradient_delta=gradient_delta,
            gradient_scores=gradient_scores,
            saliency_scores=saliency_scores,
        ),
    )


def _infer_sensitive_features(
        *,
        predicted: int,
        accuracy_delta: torch.Tensor,
        confidence_delta: torch.Tensor,
        weight_delta: torch.Tensor,
        bias_delta: torch.Tensor,
        gradient_delta: OrderedDict[str, torch.Tensor] | None,
        gradient_scores: torch.Tensor | None,
        saliency_scores: torch.Tensor | None,
        top_gradient_features: int = 5,
) -> Sequence[SensitiveFeature]:
    """Derive sensitive features describing the forgotten class."""

    features: list[SensitiveFeature] = []

    def _safe_item(tensor: torch.Tensor) -> float:
        value = float(tensor.detach().cpu().float().item())
        if math.isnan(value):
            return 0.0
        return value

    features.append(
        SensitiveFeature(
            name="accuracy_delta",
            score=_safe_item(accuracy_delta[predicted]),
            source="accuracy",
        )
    )
    features.append(
        SensitiveFeature(
            name="confidence_delta",
            score=_safe_item(confidence_delta[predicted]),
            source="confidence",
        )
    )
    features.append(
        SensitiveFeature(
            name="classifier_weight_drop",
            score=_safe_item(weight_delta[predicted]),
            source="classifier",
        )
    )
    features.append(
        SensitiveFeature(
            name="classifier_bias_drop",
            score=_safe_item(bias_delta[predicted]),
            source="classifier",
        )
    )

    if gradient_scores is not None:
        features.append(
            SensitiveFeature(
                name="gradient_class_score",
                score=_safe_item(gradient_scores[predicted]),
                source="gradient",
            )
        )

    if saliency_scores is not None:
        features.append(
            SensitiveFeature(
                name="saliency_delta",
                score=_safe_item(saliency_scores[predicted]),
                source="saliency",
            )
        )

    if gradient_delta is not None:
        gradient_importance: list[tuple[str, float]] = []
        for name, tensor in gradient_delta.items():
            grad = tensor.detach()
            if grad.dim() == 0:
                score = float(grad.abs().cpu().item())
            elif grad.size(0) > predicted:
                slice_tensor = grad[predicted]
                score = float(slice_tensor.abs().mean().cpu().item())
            else:
                score = float(grad.abs().mean().cpu().item())
            if not math.isnan(score):
                gradient_importance.append((name, score))

        gradient_importance.sort(key=lambda item: item[1], reverse=True)
        for name, score in gradient_importance[:top_gradient_features]:
            features.append(SensitiveFeature(name=name, score=score, source="gradient"))

    return tuple(features)


__all__ = ["LabelInferenceResult", "SensitiveFeature", "infer_forgotten_label"]