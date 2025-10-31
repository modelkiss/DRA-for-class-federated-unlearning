"""End-to-end experiment pipeline for class-level federated unlearning."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.attacks.data_reconstruction import DiffusionConfig, DiffusionReconstructor
from src.attacks.label_inference import LabelInferenceResult, SensitiveFeature, infer_forgotten_label
from src.data.datasets import FederatedDataConfig, FederatedDataset, create_federated_dataloaders
from src.defenses.differential_privacy import DifferentialPrivacyConfig
from src.federated.aggregation import AggregationConfig
from src.federated.client import Client, ClientConfig
from src.federated.fedavg import FederatedServer, ServerConfig
from src.forgetting.class_forgetting import (
    ForgettingResult,
    FedEraserConfig,
    FedAFConfig,
    OneShotClassUnlearningConfig,
    forget_class,
)
from src.models.nets import build_model
from src.utils.logging import setup_logging
from src.utils.metrics import accuracy
from src.utils.normalization import denormalize, get_normalization_stats

LOGGER = logging.getLogger(__name__)


INPUT_SHAPES: Dict[str, Sequence[int]] = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
}


@dataclass
class ResolvedAggregation:
    mechanism: str
    parameters: dict[str, object]
    learning_rate: float
    local_epochs: int
    batch_size: int
    client_fraction: float
    proximal_mu: float | None = None

def _count_targets_in_dataset(dataset, target_class: int) -> int:
    """Count how many samples in ``dataset`` belong to ``target_class``."""

    if isinstance(dataset, Subset):
        parent = dataset.dataset
        indices = dataset.indices
        return sum(1 for idx in indices if int(parent[idx][1]) == target_class)

    count = 0
    for index in range(len(dataset)):
        _, label = dataset[index]
        if int(label) == target_class:
            count += 1
    return count


def _normalize_heatmap(tensor: torch.Tensor) -> torch.Tensor:
    """Scale saliency maps to the [0, 1] range per sample."""

    flat = tensor.view(tensor.size(0), -1)
    mins = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1)
    tensor = tensor - mins
    flat = tensor.view(tensor.size(0), -1)
    maxs = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1)
    safe_maxs = torch.where(maxs > 0, maxs, torch.ones_like(maxs))
    return tensor / safe_maxs


def _compute_saliency_maps(model: torch.nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gradient-based saliency maps and predicted classes."""

    model.zero_grad(set_to_none=True)
    inputs = inputs.clone().detach().requires_grad_(True)
    outputs = model(inputs)
    preds = outputs.argmax(dim=1)
    selected = outputs.gather(1, preds.unsqueeze(1)).sum()
    grads = torch.autograd.grad(selected, inputs, retain_graph=False)[0]
    saliency = grads.abs().amax(dim=1, keepdim=False)
    saliency = _normalize_heatmap(saliency)
    return saliency.detach(), preds.detach()


def _normalize_to_dataset(images: torch.Tensor, dataset: str) -> torch.Tensor:
    """Normalise reconstructed images according to dataset statistics."""

    mean, std = get_normalization_stats(dataset)
    device = images.device
    dtype = images.dtype
    mean_tensor = torch.tensor(mean, device=device, dtype=dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=dtype).view(1, -1, 1, 1)
    return (images - mean_tensor) / std_tensor


def _evaluate_reconstruction_accuracy(
    model: torch.nn.Module,
    reconstructions: torch.Tensor,
    target_class: int,
    dataset: str,
    device: torch.device,
) -> float:
    """Classify reconstructions and compute accuracy for ``target_class``."""

    if reconstructions.numel() == 0:
        return 0.0

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        images = reconstructions.to(device)
        images = images.clamp(0.0, 1.0)
        images = _normalize_to_dataset(images, dataset)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        matches = predictions == target_class
        return float(matches.float().mean().item())


def _build_penalty_transform(penalties: dict[int, float]):
    if not penalties:
        return None

    def transform(scores: torch.Tensor) -> torch.Tensor:
        adjusted = scores.clone()
        for cls, factor in penalties.items():
            adjusted[cls] = adjusted[cls] * factor
        return adjusted

    return transform


def _log_sensitive_features(features: Sequence[SensitiveFeature]) -> None:
    if not features:
        LOGGER.info("敏感特征推理：未检测到显著特征，使用基础扩散提示语。")
        return

    LOGGER.info("敏感特征推理：")
    for feature in features:
        LOGGER.info("  来源=%s, 特征=%s, 重要性=%.4f", feature.source, feature.name, feature.score)


def _parse_level_thresholds(raw: str | None) -> list[float] | None:
    if raw is None:
        return None
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return None
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:  # pragma: no cover - defensive programming
        raise ValueError("AHSecAgg level thresholds must be numeric values") from exc


def _resolve_aggregation(args: argparse.Namespace) -> ResolvedAggregation:
    mechanism = args.aggregation.lower()
    if mechanism in {"pairwise", "pairwise_masking"}:
        mechanism = "fastsecagg"
    base = ResolvedAggregation(
        mechanism=mechanism,
        parameters={},
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        client_fraction=args.fraction,
    )

    if mechanism == "fedavg":
        return ResolvedAggregation(
            mechanism="fedavg",
            parameters={"weighting": args.fedavg_weight_strategy},
            learning_rate=args.fedavg_lr if args.fedavg_lr is not None else base.learning_rate,
            local_epochs=args.fedavg_local_epochs if args.fedavg_local_epochs is not None else base.local_epochs,
            batch_size=args.fedavg_batch_size if args.fedavg_batch_size is not None else base.batch_size,
            client_fraction=(
                args.fedavg_client_fraction if args.fedavg_client_fraction is not None else base.client_fraction
            ),
            proximal_mu=None,
        )
    if mechanism == "fedprox":
        return ResolvedAggregation(
            mechanism="fedprox",
            parameters={"weighting": args.fedprox_weight_strategy},
            learning_rate=args.fedprox_lr if args.fedprox_lr is not None else base.learning_rate,
            local_epochs=args.fedprox_local_epochs if args.fedprox_local_epochs is not None else base.local_epochs,
            batch_size=args.fedprox_batch_size if args.fedprox_batch_size is not None else base.batch_size,
            client_fraction=(
                args.fedprox_client_fraction if args.fedprox_client_fraction is not None else base.client_fraction
            ),
            proximal_mu=args.fedprox_mu,
        )
    if mechanism == "secagg":
        return ResolvedAggregation(
            mechanism="secagg",
            parameters={
                "mask_seed": args.secagg_seed,
                "threshold": args.secagg_threshold,
                "dropout_tolerance": args.secagg_dropout_tolerance,
                "retransmissions": args.secagg_retransmissions,
                "key_refresh_interval": args.secagg_key_refresh,
            },
            learning_rate=base.learning_rate,
            local_epochs=base.local_epochs,
            batch_size=base.batch_size,
            client_fraction=base.client_fraction,
        )
    if mechanism == "ahsecagg":
        return ResolvedAggregation(
            mechanism="ahsecagg",
            parameters={
                "cluster_size": args.ahsecagg_cluster_size,
                "levels": args.ahsecagg_levels,
                "level_thresholds": _parse_level_thresholds(args.ahsecagg_level_thresholds),
                "dropout_rate": args.ahsecagg_dropout,
                "mask_reuse": args.ahsecagg_mask_reuse,
            },
            learning_rate=base.learning_rate,
            local_epochs=base.local_epochs,
            batch_size=base.batch_size,
            client_fraction=base.client_fraction,
        )
    if mechanism == "fastsecagg":
        return ResolvedAggregation(
            mechanism="fastsecagg",
            parameters={
                "group_count": args.fastsecagg_groups,
                "key_agreement": args.fastsecagg_key_agreement,
                "mask_update_frequency": args.fastsecagg_mask_update,
                "timeout": args.fastsecagg_timeout,
                "encryption": args.fastsecagg_encryption,
            },
            learning_rate=base.learning_rate,
            local_epochs=base.local_epochs,
            batch_size=base.batch_size,
            client_fraction=base.client_fraction,
        )
    raise ValueError(f"Unsupported aggregation mechanism: {args.aggregation}")


def _export_sample_heatmaps(
    *,
    dataloader,
    model_before: torch.nn.Module,
    model_after: torch.nn.Module,
    output_dir: Path,
    dataset: str,
    cmap: str,
    device: torch.device,
    plt_module,
) -> Path:
    """Generate per-sample visualisations comparing saliency maps before/after forgetting."""

    stats = get_normalization_stats(dataset)
    sample_dir = output_dir / f"sample_heatmaps_{dataset}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    model_before = model_before.to(device).eval()
    model_after = model_after.to(device).eval()

    total = 0
    for batch_index, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        with torch.enable_grad():
            before_maps, before_preds = _compute_saliency_maps(model_before, inputs)
            after_maps, after_preds = _compute_saliency_maps(model_after, inputs)

        diff_maps = before_maps - after_maps
        originals = denormalize(inputs.detach().cpu(), stats).clamp(0.0, 1.0)
        before_maps = before_maps.cpu()
        after_maps = after_maps.cpu()
        diff_maps = diff_maps.cpu()
        before_preds = before_preds.cpu()
        after_preds = after_preds.cpu()
        targets_cpu = targets.detach().cpu()

        for offset in range(inputs.size(0)):
            figure_path = sample_dir / f"sample_{total:05d}.png"
            fig, axes = plt_module.subplots(1, 4, figsize=(12, 3))
            image = originals[offset].permute(1, 2, 0).numpy()
            if image.shape[2] == 1:
                axes[0].imshow(image[:, :, 0], cmap="gray", vmin=0.0, vmax=1.0)
            else:
                axes[0].imshow(image, vmin=0.0, vmax=1.0)
            axes[0].set_title(f"原图\n标签={int(targets_cpu[offset])}")

            axes[1].imshow(before_maps[offset].numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
            axes[1].set_title(f"遗忘前热力图\n预测={int(before_preds[offset])}")

            axes[2].imshow(after_maps[offset].numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
            axes[2].set_title(f"遗忘后热力图\n预测={int(after_preds[offset])}")

            axes[3].imshow(diff_maps[offset].numpy(), cmap=cmap, vmin=-1.0, vmax=1.0)
            axes[3].set_title("热力图差异")

            for ax in axes:
                ax.axis("off")

            fig.suptitle(f"样本 {total}")
            fig.tight_layout()
            fig.savefig(figure_path, dpi=200)
            plt_module.close(fig)
            total += 1

        LOGGER.debug(
            "Processed batch %d/%d for per-sample热力图 (累计样本=%d)",
            batch_index + 1,
            len(dataloader),
            total,
        )

    LOGGER.info("已为测试集生成 %d 张热力图对比图，保存在 %s", total, sample_dir)
    return sample_dir



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated class forgetting pipeline")
    parser.add_argument("--dataset", default="cifar10", choices=INPUT_SHAPES.keys())
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5, help="Number of pre-forgetting federated rounds")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iid", action="store_true", help="Use IID data partitioning")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=1.0, help="Client sampling fraction per round")
    parser.add_argument(
        "--aggregation",
        default="fedavg",
        choices=["fedavg", "fedprox", "secagg", "ahsecagg", "fastsecagg", "pairwise", "pairwise_masking"],
        help="Federated aggregation mechanism to coordinate client updates.",
    )
    parser.add_argument("--fedavg-lr", type=float, default=None, help="FedAvg: local client learning rate override.")
    parser.add_argument(
        "--fedavg-local-epochs",
        type=int,
        default=None,
        help="FedAvg: number of local epochs performed by each client.",
    )
    parser.add_argument(
        "--fedavg-client-fraction",
        type=float,
        default=None,
        help="FedAvg: fraction of clients sampled per round.",
    )
    parser.add_argument(
        "--fedavg-batch-size",
        type=int,
        default=None,
        help="FedAvg: mini-batch size used for local training.",
    )
    parser.add_argument(
        "--fedavg-weight-strategy",
        default="samples",
        choices=["samples", "uniform"],
        help="FedAvg: aggregation weighting strategy (by samples or uniform).",
    )
    parser.add_argument("--fedprox-mu", type=float, default=0.01,
                        help="FedProx: proximal regularisation coefficient μ.")
    parser.add_argument("--fedprox-lr", type=float, default=None, help="FedProx: client learning rate override.")
    parser.add_argument(
        "--fedprox-local-epochs",
        type=int,
        default=None,
        help="FedProx: number of local epochs per client.",
    )
    parser.add_argument(
        "--fedprox-batch-size",
        type=int,
        default=None,
        help="FedProx: mini-batch size used for local optimisation.",
    )
    parser.add_argument(
        "--fedprox-client-fraction",
        type=float,
        default=None,
        help="FedProx: fraction of clients sampled each round.",
    )
    parser.add_argument(
        "--fedprox-weight-strategy",
        default="samples",
        choices=["samples", "uniform"],
        help="FedProx: aggregation weighting rule.",
    )
    parser.add_argument("--secagg-seed", type=int, default=42, help="SecAgg: random seed for mask generation.")
    parser.add_argument(
        "--secagg-threshold",
        type=int,
        default=2,
        help="SecAgg: minimum number of clients required for decryption.",
    )
    parser.add_argument(
        "--secagg-dropout-tolerance",
        type=float,
        default=0.1,
        help="SecAgg: tolerated client dropout ratio before aborting aggregation.",
    )
    parser.add_argument(
        "--secagg-retransmissions",
        type=int,
        default=1,
        help="SecAgg: communication retry attempts for masked shares.",
    )
    parser.add_argument(
        "--secagg-key-refresh",
        type=int,
        default=10,
        help="SecAgg: key refresh period measured in federated rounds.",
    )
    parser.add_argument(
        "--ahsecagg-cluster-size",
        type=int,
        default=4,
        help="AHSecAgg: number of clients per adaptive cluster.",
    )
    parser.add_argument(
        "--ahsecagg-levels",
        type=int,
        default=2,
        help="AHSecAgg: depth of the secure aggregation hierarchy.",
    )
    parser.add_argument(
        "--ahsecagg-level-thresholds",
        default=None,
        help="AHSecAgg: comma-separated decryption thresholds for each hierarchy level.",
    )
    parser.add_argument(
        "--ahsecagg-dropout",
        type=float,
        default=0.1,
        help="AHSecAgg: tolerated dropout ratio within each cluster.",
    )
    parser.add_argument(
        "--ahsecagg-mask-reuse",
        default="never",
        choices=["never", "round", "epoch"],
        help="AHSecAgg: mask reuse policy across rounds.",
    )
    parser.add_argument(
        "--fastsecagg-groups",
        type=int,
        default=2,
        help="FastSecAgg: number of client groups for pairwise masking.",
    )
    parser.add_argument(
        "--fastsecagg-key-agreement",
        default="static",
        choices=["static", "dynamic"],
        help="FastSecAgg: key agreement mode between clients.",
    )
    parser.add_argument(
        "--fastsecagg-mask-update",
        type=int,
        default=1,
        help="FastSecAgg: frequency of mask regeneration in rounds.",
    )
    parser.add_argument(
        "--fastsecagg-timeout",
        type=float,
        default=30.0,
        help="FastSecAgg: fault-tolerance timeout window in seconds.",
    )
    parser.add_argument(
        "--fastsecagg-encryption",
        default="paillier",
        choices=["paillier", "modular"],
        help="FastSecAgg: encryption primitive applied to masked sums.",
    )
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--reconstructions", type=int, default=8)
    parser.add_argument(
        "--reconstruction-tolerance",
        type=float,
        default=0.05,
        help="允许生成样本分类准确率相对测试集差异的最大绝对值 (单位: 比例).",
    )
    parser.add_argument(
        "--reinference-penalty",
        type=float,
        default=0.8,
        help="重推理时对上一轮预测类别的融合得分缩放系数 (0-1).",
    )
    parser.add_argument(
        "--max-reinference",
        type=int,
        default=5,
        help="允许触发的最大重推理次数，超过即判定重建失败。",
    )
    parser.add_argument(
        "--forgetting-method",
        default="fed_eraser",
        choices=["fed_eraser", "fedaf", "one_shot"],
        help="Class forgetting strategy applied after removing the target class.",
    )
    parser.add_argument(
        "--fed-eraser-history",
        type=int,
        default=5,
        help="FedEraser: number of recent calibration states retained for averaging.",
    )
    parser.add_argument(
        "--fed-eraser-calibration-rounds",
        type=int,
        default=3,
        help="FedEraser: number of calibration rounds performed after class removal.",
    )
    parser.add_argument(
        "--fed-eraser-lr",
        type=float,
        default=5e-3,
        help="FedEraser: calibration learning rate for participating clients.",
    )
    parser.add_argument(
        "--fed-eraser-client-fraction",
        type=float,
        default=0.5,
        help="FedEraser: fraction of clients sampled during calibration rounds.",
    )
    parser.add_argument(
        "--fed-eraser-weight-decay",
        type=float,
        default=1e-4,
        help="FedEraser: weight decay applied during calibration updates.",
    )
    parser.add_argument(
        "--fedaf-lambda",
        type=float,
        default=0.25,
        help="FedAF: forgetting regularisation weight λ.",
    )
    parser.add_argument(
        "--fedaf-beta",
        type=float,
        default=0.2,
        help="FedAF: retained knowledge constraint strength β.",
    )
    parser.add_argument(
        "--fedaf-lr",
        type=float,
        default=1e-2,
        help="FedAF: learning rate used during adaptive forgetting rounds.",
    )
    parser.add_argument(
        "--fedaf-mask-ratio",
        type=float,
        default=0.3,
        help="FedAF: ratio of target-class weights masked each round.",
    )
    parser.add_argument(
        "--fedaf-stop-threshold",
        type=float,
        default=0.4,
        help="FedAF: stop once the target-class weight norm drops by this ratio.",
    )
    parser.add_argument(
        "--fedaf-rounds",
        type=int,
        default=3,
        help="FedAF: maximum number of adaptive forgetting rounds.",
    )
    parser.add_argument(
        "--oneshot-projection-dim",
        type=int,
        default=16,
        help="One-shot unlearning: projection subspace dimensionality.",
    )
    parser.add_argument(
        "--oneshot-replacement-strength",
        type=float,
        default=0.5,
        help="One-shot unlearning: noise strength injected into the target classifier weights.",
    )
    parser.add_argument(
        "--oneshot-freeze-ratio",
        type=float,
        default=0.4,
        help="One-shot unlearning: fraction of non-classifier parameters frozen during tuning.",
    )
    parser.add_argument(
        "--oneshot-local-epochs",
        type=int,
        default=1,
        help="One-shot unlearning: local fine-tuning epochs executed on each client.",
    )
    parser.add_argument(
        "--oneshot-reconstruction-threshold",
        type=float,
        default=0.2,
        help="One-shot unlearning: trigger an extra stabilisation round if below this reconstruction gap.",
    )
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--no-heatmaps", action="store_true", help="Disable exporting heatmap visualisations")
    parser.add_argument(
        "--heatmap-cmap",
        default="coolwarm",
        help="Matplotlib colormap name used when rendering heatmaps",
    )
    parser.add_argument(
        "--dp-method",
        default="none",
        choices=["none", "dp-sgd", "ldp-fl", "adaptive-dp-fl", "rdp-fl"],
        help="Differential privacy strategy to enable during training.",
    )
    parser.add_argument(
        "--dp-sgd-epsilon",
        type=float,
        default=1.0,
        help="Target privacy budget ε for DP-SGD (central DP).",
    )
    parser.add_argument(
        "--dp-sgd-delta",
        type=float,
        default=1e-5,
        help="Confidence parameter δ for DP-SGD.",
    )
    parser.add_argument(
        "--dp-sgd-noise",
        type=float,
        default=1.0,
        help="Noise multiplier σ applied to aggregated updates in DP-SGD.",
    )
    parser.add_argument(
        "--dp-sgd-clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold C used by DP-SGD.",
    )
    parser.add_argument(
        "--dp-sgd-sampling-prob",
        type=float,
        default=1.0,
        help="Global sampling proportion q for DP-SGD.",
    )
    parser.add_argument(
        "--ldp-noise-multiplier",
        type=float,
        default=1.0,
        help="Per-client noise multiplier for LDP-FL.",
    )
    parser.add_argument(
        "--ldp-clip",
        type=float,
        default=1.0,
        help="Clipping norm applied to client updates under LDP-FL.",
    )
    parser.add_argument(
        "--ldp-seed-consistency",
        dest="ldp_seed_consistency",
        action="store_true",
        help="Use a consistent random seed across LDP-FL rounds.",
    )
    parser.set_defaults(ldp_seed_consistency=True)
    parser.add_argument(
        "--no-ldp-seed-consistency",
        dest="ldp_seed_consistency",
        action="store_false",
        help="Disable deterministic noise for LDP-FL.",
    )
    parser.add_argument(
        "--ldp-local-sampling-rate",
        type=float,
        default=1.0,
        help="Client-side sampling rate for LDP-FL.",
    )
    parser.add_argument(
        "--ldp-epsilon",
        type=float,
        default=1.0,
        help="Local privacy budget ε for LDP-FL.",
    )
    parser.add_argument(
        "--ldp-delta",
        type=float,
        default=1e-5,
        help="Local privacy confidence δ for LDP-FL.",
    )
    parser.add_argument(
        "--adaptive-dp-epsilon-fn",
        default="1.0",
        help="Expression describing adaptive ε allocation as a function of 'round' and 'total_rounds'.",
    )
    parser.add_argument(
        "--adaptive-dp-clipping",
        default="linear",
        choices=["none", "linear", "exponential", "cosine"],
        help="Adaptive clipping strategy for Adaptive DP-FL.",
    )
    parser.add_argument(
        "--adaptive-dp-noise-growth",
        type=float,
        default=0.5,
        help="Noise growth rate controlling σ progression in Adaptive DP-FL.",
    )
    parser.add_argument(
        "--adaptive-dp-max-rounds",
        type=int,
        default=100,
        help="Maximum rounds permitted under Adaptive DP-FL before disabling noise.",
    )
    parser.add_argument(
        "--adaptive-dp-budget-decay",
        type=float,
        default=0.1,
        help="Budget decay coefficient applied to Adaptive DP-FL clipping/noise.",
    )
    parser.add_argument(
        "--rdp-order",
        type=float,
        default=2.0,
        help="Rényi order α for RDP-FL accounting.",
    )
    parser.add_argument(
        "--rdp-epsilon-rule",
        default="additive",
        choices=["additive", "max", "window"],
        help="Accumulation rule for ε under RDP-FL.",
    )
    parser.add_argument(
        "--rdp-noise-multiplier",
        type=float,
        default=1.0,
        help="Noise multiplier applied each round in RDP-FL.",
    )
    parser.add_argument(
        "--rdp-clip",
        type=float,
        default=1.0,
        help="Clipping threshold for RDP-FL updates.",
    )
    parser.add_argument(
        "--rdp-accounting-window",
        type=int,
        default=10,
        help="Number of recent rounds considered in windowed RDP-FL accounting.",
    )
    parser.add_argument(
        "--gradient-batches",
        type=int,
        default=2,
        help="Number of batches for gradient-difference estimation (0 to disable).",
    )
    parser.add_argument(
        "--gradient-params",
        default="classifier,fc",
        help="Comma separated substrings for selecting parameters whose gradients are inspected.",
    )
    parser.add_argument(
        "--diffusion-model-id",
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained diffusion pipeline identifier (diffusers).",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=40,
        help="Number of inference steps for diffusion reconstruction.",
    )
    parser.add_argument(
        "--diffusion-guidance",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale for diffusion reconstruction.",
    )
    parser.add_argument(
        "--diffusion-negative-prompt",
        default=None,
        help="Optional negative prompt for diffusion sampling.",
    )
    parser.add_argument(
        "--diffusion-guidance-epochs",
        type=int,
        default=1,
        help="敏感特征指导扩散模型时的迭代轮次 (启发式设置).",
    )
    parser.add_argument(
        "--diffusion-guidance-lr",
        type=float,
        default=1e-4,
        help="敏感特征指导扩散模型时的学习率 (仅用于记录).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device %s", device)

    aggregation = _resolve_aggregation(args)

    federated_config = FederatedDataConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        batch_size=aggregation.batch_size,
        iid=args.iid,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    federated_dataset = create_federated_dataloaders(federated_config)

    target_sample_count = sum(
        _count_targets_in_dataset(loader.dataset, args.target_class)
        for loader in federated_dataset.train_loaders.values()
    )
    LOGGER.info(
        "目标类别 %d 在遗忘前共有 %d 个样本（涵盖所有客户端）",
        args.target_class,
        target_sample_count,
    )

    model = build_model(args.dataset, federated_dataset.num_classes)

    client_config = ClientConfig(
        learning_rate=aggregation.learning_rate,
        local_epochs=aggregation.local_epochs,
        device=device,
        proximal_mu=aggregation.proximal_mu,
    )
    clients = [
        Client(client_id=i, dataloader=loader, config=client_config)
        for i, loader in federated_dataset.train_loaders.items()
    ]

    dp_parameters: dict[str, float | int | str | bool] = {}
    if args.dp_method == "dp-sgd":
        dp_parameters = {
            "epsilon": args.dp_sgd_epsilon,
            "delta": args.dp_sgd_delta,
            "noise_multiplier": args.dp_sgd_noise,
            "clip": args.dp_sgd_clip,
            "sampling_probability": args.dp_sgd_sampling_prob,
        }
    elif args.dp_method == "ldp-fl":
        dp_parameters = {
            "per_client_noise_multiplier": args.ldp_noise_multiplier,
            "clipping_norm": args.ldp_clip,
            "seed_consistency": args.ldp_seed_consistency,
            "local_sampling_rate": args.ldp_local_sampling_rate,
            "epsilon": args.ldp_epsilon,
            "delta": args.ldp_delta,
        }
    elif args.dp_method == "adaptive-dp-fl":
        dp_parameters = {
            "epsilon_expression": args.adaptive_dp_epsilon_fn,
            "adaptive_clipping": args.adaptive_dp_clipping,
            "noise_growth_rate": args.adaptive_dp_noise_growth,
            "max_rounds": args.adaptive_dp_max_rounds,
            "budget_decay": args.adaptive_dp_budget_decay,
        }
    elif args.dp_method == "rdp-fl":
        dp_parameters = {
            "renyi_order": args.rdp_order,
            "epsilon_accumulation": args.rdp_epsilon_rule,
            "noise_multiplier": args.rdp_noise_multiplier,
            "clip": args.rdp_clip,
            "accounting_window": args.rdp_accounting_window,
        }

    dp_config = DifferentialPrivacyConfig(method=args.dp_method, parameters=dp_parameters)

    server_config = ServerConfig(
        device=device,
        fraction=aggregation.client_fraction,
        dp_config=dp_config,
        aggregation=AggregationConfig(
            mechanism=aggregation.mechanism,
            parameters=aggregation.parameters,
        ),
    )
    server = FederatedServer(model=model, clients=clients, config=server_config)

    LOGGER.info("Starting federated pre-training for %d rounds", args.rounds)
    server.train(args.rounds)
    pre_forgetting_model = copy.deepcopy(server.global_model)

    baseline_accuracy = accuracy(pre_forgetting_model.to(device), federated_dataset.test_loader, device)
    LOGGER.info("Baseline accuracy before forgetting: %.4f", baseline_accuracy)

    if args.forgetting_method == "fed_eraser":
        method_config = FedEraserConfig(
            history_window=args.fed_eraser_history,
            calibration_rounds=args.fed_eraser_calibration_rounds,
            calibration_lr=args.fed_eraser_lr,
            calibration_client_fraction=args.fed_eraser_client_fraction,
            update_weight_decay=args.fed_eraser_weight_decay,
        )
    elif args.forgetting_method == "fedaf":
        method_config = FedAFConfig(
            forgetting_regularization=args.fedaf_lambda,
            retention_strength=args.fedaf_beta,
            learning_rate=args.fedaf_lr,
            class_mask_ratio=args.fedaf_mask_ratio,
            stop_threshold=args.fedaf_stop_threshold,
            optimisation_rounds=args.fedaf_rounds,
        )
    else:
        method_config = OneShotClassUnlearningConfig(
            projection_dim=args.oneshot_projection_dim,
            replacement_strength=args.oneshot_replacement_strength,
            freeze_ratio=args.oneshot_freeze_ratio,
            local_tuning_epochs=args.oneshot_local_epochs,
            reconstruction_threshold=args.oneshot_reconstruction_threshold,
        )

    forgetting_result = perform_forgetting(
        server=server,
        dataset=federated_dataset,
        client_config=client_config,
        target_class=args.target_class,
        method=args.forgetting_method,
        input_shape=INPUT_SHAPES[args.dataset],
        method_config=method_config,
    )

    post_forgetting_model = copy.deepcopy(server.global_model)
    post_accuracy = accuracy(post_forgetting_model.to(device), federated_dataset.test_loader, device)
    LOGGER.info("Accuracy after forgetting: %.4f", post_accuracy)

    gradient_filter = None
    if args.gradient_params:
        tokens = [token.strip() for token in args.gradient_params.split(",") if token.strip()]
        gradient_filter = tokens or None

    diffusion_config = DiffusionConfig(
        model_id=args.diffusion_model_id,
        guidance_scale=args.diffusion_guidance,
        num_inference_steps=args.diffusion_steps,
        device=device,
        negative_prompt=args.diffusion_negative_prompt,
    )
    try:
        diffusion = DiffusionReconstructor(diffusion_config)
    except RuntimeError as exc:
        LOGGER.error("Diffusion reconstruction failed: %s", exc)
        raise SystemExit("Diffusion-based reconstruction requires the 'diffusers' package") from exc

    penalties: dict[int, float] = {}
    reinference_count = 0
    successful_reconstruction_accuracy: float | None = None
    class_label: str | None = None

    while True:
        transform = _build_penalty_transform(penalties)
        inference = infer_forgotten_label(
            before=pre_forgetting_model,
            after=post_forgetting_model,
            dataloader=federated_dataset.test_loader,
            num_classes=federated_dataset.num_classes,
            device=device,
            ground_truth=args.target_class,
            gradient_batches=args.gradient_batches if args.gradient_batches > 0 else None,
            gradient_filter=gradient_filter,
            transform=transform,
        )

        LOGGER.info(
            "Label inference attempt %d -> 预测类别 %d (真实类别 %d)",
            reinference_count + 1,
            inference.predicted_class,
            args.target_class,
        )
        if inference.candidate_details:
            LOGGER.info("候选类别多指标融合得分：")
            for cls in sorted(inference.candidate_details):
                metrics = inference.candidate_details[cls]
                LOGGER.info(
                    "  类别 %d -> 融合得分 %.4f, ΔAcc %.2f%%, ΔConf %.2f%%, ΔW %.4f, ΔGrad %s, ΔSal %.4f",
                    cls,
                    metrics.get("fusion_score", float("nan")),
                    metrics.get("accuracy_delta", 0.0) * 100,
                    metrics.get("confidence_delta", 0.0) * 100,
                    metrics.get("weight_delta", 0.0),
                    "{:.4f}".format(metrics["gradient_delta"]) if metrics.get("gradient_delta") is not None else "N/A",
                    metrics.get("saliency_delta", 0.0) if metrics.get("saliency_delta") is not None else 0.0,
                )
        if inference.gradient_delta is not None:
            gradient_norms = {name: tensor.norm().item() for name, tensor in inference.gradient_delta.items()}
            LOGGER.debug("Gradient delta norms: %s", gradient_norms)

        if federated_dataset.class_names and inference.predicted_class < len(federated_dataset.class_names):
            class_label = str(federated_dataset.class_names[inference.predicted_class])
        else:
            class_label = None

        sensitive_features = list(inference.sensitive_features or [])
        _log_sensitive_features(sensitive_features)

        diffusion.fine_tune_with_guidance(
            sensitive_features,
            epochs=args.diffusion_guidance_epochs,
            learning_rate=args.diffusion_guidance_lr,
        )

        candidate_reconstructions = diffusion.reconstruct(
            target_class=inference.predicted_class,
            num_samples=args.reconstructions,
            class_label=class_label,
        )

        expected_shape = INPUT_SHAPES[args.dataset]
        if candidate_reconstructions.shape[1:] != expected_shape:
            LOGGER.debug(
                "Resizing reconstructions from %s to expected shape %s",
                candidate_reconstructions.shape[1:],
                expected_shape,
            )
            candidate_reconstructions = F.interpolate(
                candidate_reconstructions,
                size=expected_shape[1:],
                mode="bilinear",
                align_corners=False,
            )
            if expected_shape[0] == 1 and candidate_reconstructions.shape[1] == 3:
                candidate_reconstructions = candidate_reconstructions.mean(dim=1, keepdim=True)

        baseline_class_accuracy = float(inference.per_class_before[inference.predicted_class].item())
        reconstruction_accuracy = _evaluate_reconstruction_accuracy(
            pre_forgetting_model,
            candidate_reconstructions,
            inference.predicted_class,
            args.dataset,
            device,
        )
        accuracy_gap = abs(reconstruction_accuracy - baseline_class_accuracy)
        LOGGER.info(
            "重构评估: 生成样本准确率=%.4f, 测试集同类准确率=%.4f, 差值=%.4f (阈值=%.4f)",
            reconstruction_accuracy,
            baseline_class_accuracy,
            accuracy_gap,
            args.reconstruction_tolerance,
        )

        if accuracy_gap <= args.reconstruction_tolerance:
            reconstructions = candidate_reconstructions
            successful_reconstruction_accuracy = reconstruction_accuracy
            break

        reinference_count += 1
        LOGGER.error(
            "重构准确率偏差 %.4f 超出允许范围，记录错误推理 %d 次，准备重新推理标签。",
            accuracy_gap,
            reinference_count,
        )
        if reinference_count > args.max_reinference:
            LOGGER.error("超过最大重推理次数 %d，宣布重建失败。", args.max_reinference)
            raise SystemExit("Reconstruction failed: exceeded maximum reinference attempts")

        penalties[inference.predicted_class] = penalties.get(inference.predicted_class, 1.0) * args.reinference_penalty

    inference_result = inference

    if successful_reconstruction_accuracy is None:
        successful_reconstruction_accuracy = 0.0

    per_class_records = []
    for cls in range(federated_dataset.num_classes):
        before_acc = float(inference_result.per_class_before[cls].item())
        after_acc = float(inference_result.per_class_after[cls].item())
        delta = before_acc - after_acc
        LOGGER.info(
            "类别 %d: 遗忘前 %.2f%%, 遗忘后 %.2f%%, 差值 %.2f%%",
            cls,
            before_acc * 100,
            after_acc * 100,
            delta * 100,
        )
        per_class_records.append(
            {
                "class": cls,
                "before": before_acc,
                "after": after_acc,
                "delta": delta,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(reconstructions, args.output / "reconstructed.pt")
    torch.save(pre_forgetting_model.state_dict(), args.output / "model_before.pt")
    torch.save(post_forgetting_model.state_dict(), args.output / "model_after.pt")
    torch.save(
        {
            "original": forgetting_result.original_state,
            "forgotten": forgetting_result.forgotten_state,
        },
        args.output / "models.pt",
    )
    if inference_result.gradient_delta is not None:
        torch.save(
            {
                "before": inference_result.gradient_before,
                "after": inference_result.gradient_after,
                "delta": inference_result.gradient_delta,
            },
            args.output / "gradient_deltas.pt",
        )
    with (args.output / "inference.json").open("w", encoding="utf-8") as handle:
        json.dump(inference_result.to_dict(), handle, indent=2)
    metadata = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "fedaf_rounds": args.fedaf_rounds if args.forgetting_method == "fedaf" else None,
        "baseline_accuracy": baseline_accuracy,
        "post_accuracy": post_accuracy,
        "target_class": args.target_class,
        "target_class_sample_count": target_sample_count,
        "predicted_class": inference_result.predicted_class,
        "attack_success": inference_result.predicted_class == args.target_class,
        "dp_method": args.dp_method,
        "dp_parameters": dp_parameters,
        "aggregation": {
            "mechanism": aggregation.mechanism,
            "parameters": aggregation.parameters,
            "is_secure": server.aggregator.is_secure,
        },
        "class_label": class_label,
        "gradient_batches": args.gradient_batches,
        "gradient_params": gradient_filter,
        "diffusion_model_id": args.diffusion_model_id,
        "diffusion_guidance": {
            "final_scale": diffusion.config.guidance_scale,
            "epochs": args.diffusion_guidance_epochs,
            "learning_rate": args.diffusion_guidance_lr,
            "metadata": getattr(diffusion, "_guidance_hparams", {}),
        },
        "reconstruction_accuracy": successful_reconstruction_accuracy,
        "reconstruction_tolerance": args.reconstruction_tolerance,
        "reinference_count": reinference_count,
        "baseline_class_accuracy": float(
            inference_result.per_class_before[inference_result.predicted_class].item()
        ),
        "sensitive_features": [
            feature.to_dict() for feature in (inference_result.sensitive_features or [])
        ],
        "heatmaps": {
            "enabled": not args.no_heatmaps,
            "cmap": args.heatmap_cmap,
        },
        "per_class_accuracy": per_class_records,
    }

    if not args.no_heatmaps:
        heatmap_artifacts = export_heatmaps(
            inference_result,
            output_dir=args.output,
            cmap=args.heatmap_cmap,
            dataset=args.dataset,
            gradient_filter=gradient_filter,
            dataloader=federated_dataset.test_loader,
            model_before=pre_forgetting_model,
            model_after=post_forgetting_model,
            device=device,
        )
        metadata["heatmaps"].update(
            {key: str(path) if path is not None else None for key, path in heatmap_artifacts.items()})
    else:
        heatmap_artifacts = {}

    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    LOGGER.info(
        "Saved %d reconstructed samples and inference metadata to %s (重推理次数=%d)",
        len(reconstructions),
        args.output,
        reinference_count,
    )


def perform_forgetting(
    server: FederatedServer,
    dataset: FederatedDataset,
    client_config: ClientConfig,
    target_class: int,
    method: str,
    input_shape: Sequence[int],
    method_config: FedEraserConfig | FedAFConfig | OneShotClassUnlearningConfig,
) -> ForgettingResult:
    return forget_class(
        server=server,
        dataset=dataset,
        client_config=client_config,
        target_class=target_class,
        method=method,
        input_shape=input_shape,
        method_config=method_config,
    )


def export_heatmaps(
    inference: LabelInferenceResult,
    output_dir: Path,
    cmap: str,
    dataset: str,
    *,
    gradient_filter: Sequence[str] | None = None,
 dataloader=None,
    model_before: torch.nn.Module | None = None,
    model_after: torch.nn.Module | None = None,
    device: torch.device | None = None,
) -> dict[str, Path | None]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Matplotlib is not available, skipping heatmap export: %s", exc)
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = inference.per_class_before.numel()
    class_indices = list(range(num_classes))
    results: dict[str, Path | None] = {"per_class": None, "confusion": None, "gradient": None, "samples": None}

    # Per-class accuracy differences
    acc_diff = (
        inference.per_class_before - inference.per_class_after
    ).unsqueeze(0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), 2.5))
    im = ax.imshow(acc_diff, cmap=cmap, aspect="auto")
    ax.set_title(f"Per-class accuracy delta ({dataset})")
    ax.set_yticks([0])
    ax.set_yticklabels(["Δ Acc"])

    per_class_path = output_dir / f"heatmap_accuracy_delta_{dataset}.png"
    fig.savefig(per_class_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved per-class heatmap to %s using cmap=%s", per_class_path, cmap)
    results["per_class"] = per_class_path

    # Confusion matrix differences
    conf_diff = (
        inference.confusion_before - inference.confusion_after
    ).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), max(6, num_classes * 0.6)))
    im = ax.imshow(conf_diff, cmap=cmap)
    ax.set_title(f"Confusion matrix delta ({dataset})")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(class_indices)
    ax.set_xticklabels(class_indices, rotation=90)
    ax.set_yticks(class_indices)
    ax.set_yticklabels(class_indices)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    confusion_path = output_dir / f"heatmap_confusion_delta_{dataset}.png"
    fig.savefig(confusion_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved confusion heatmap to %s using cmap=%s", confusion_path, cmap)
    results["confusion"] = confusion_path

    if inference.gradient_delta is not None:
        names = list(inference.gradient_delta.keys())
        norms = [tensor.norm().item() for tensor in inference.gradient_delta.values()]
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.6), 3))
        ax.imshow(torch.tensor(norms).unsqueeze(0), cmap=cmap, aspect="auto")
        ax.set_title("Gradient delta norms")
        ax.set_yticks([0])
        label = "Δ‖∇‖"
        if gradient_filter:
            label += f" ({', '.join(gradient_filter)})"
        ax.set_yticklabels([label])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        fig.tight_layout()
        gradient_path = output_dir / f"heatmap_gradient_delta_{dataset}.png"
        fig.savefig(gradient_path, dpi=200)
        plt.close(fig)
        LOGGER.info("Saved gradient delta heatmap to %s using cmap=%s", gradient_path, cmap)
        results["gradient"] = gradient_path

    if dataloader is not None and model_before is not None and model_after is not None and device is not None:
        results["samples"] = _export_sample_heatmaps(
            dataloader=dataloader,
            model_before=model_before,
            model_after=model_after,
            output_dir=output_dir,
            dataset=dataset,
            cmap=cmap,
            device=device,
            plt_module=plt,
        )

    return results


if __name__ == "__main__":
    main()