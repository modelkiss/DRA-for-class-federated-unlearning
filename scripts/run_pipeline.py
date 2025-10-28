"""End-to-end experiment pipeline for class-level federated unlearning."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from src.attacks.data_reconstruction import (
    DiffusionConfig,
    DiffusionReconstructor,
    GradientReconstructor,
    ReconstructionConfig,
)
from src.attacks.label_inference import LabelInferenceResult, infer_forgotten_label
from src.data.datasets import FederatedDataConfig, FederatedDataset, create_federated_dataloaders
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
    parser.add_argument("--forget-rounds", type=int, default=3, help="Rounds after removing the target class")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iid", action="store_true", help="Use IID data partitioning")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=1.0, help="Client sampling fraction per round")
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--dp-sigma", type=float, default=0.0)
    parser.add_argument("--dp-clip", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--reconstructions", type=int, default=8)
    parser.add_argument("--reconstruction-steps", type=int, default=400)
    parser.add_argument("--reconstruction-lr", type=float, default=0.1)
    parser.add_argument("--reconstruction-lambda", type=float, default=1.0)
    parser.add_argument("--reconstruction-tv", type=float, default=1e-4)
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
    parser.add_argument(
        "--reconstruction-method",
        default="gradient",
        choices=["gradient", "diffusion"],
    )
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--no-heatmaps", action="store_true", help="Disable exporting heatmap visualisations")
    parser.add_argument(
        "--heatmap-cmap",
        default="coolwarm",
        help="Matplotlib colormap name used when rendering heatmaps",
    )
    parser.add_argument(
        "--secure-aggregation",
        default="none",
        choices=["none", "bonawitz2017", "shamir", "homomorphic"],
        help="Secure aggregation protocol to simulate.",
    )
    parser.add_argument(
        "--dp-mechanism",
        default="gaussian",
        choices=["gaussian", "laplace", "student"],
        help="Differential privacy noise distribution",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(getattr(logging, args.log_level.upper()))

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device %s", device)

    federated_config = FederatedDataConfig(
        dataset=args.dataset,
        num_clients=args.num_clients,
        batch_size=args.batch_size,
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
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        device=device,
    )
    clients = [
        Client(client_id=i, dataloader=loader, config=client_config)
        for i, loader in federated_dataset.train_loaders.items()
    ]

    server_config = ServerConfig(
        device=device,
        fraction=args.fraction,
        dp_sigma=args.dp_sigma if args.dp_sigma > 0 else None,
        dp_clip=args.dp_clip,
        secure_aggregation=args.secure_aggregation,
        dp_mechanism=args.dp_mechanism,
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
        rounds=args.forget_rounds,
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

    inference = infer_forgotten_label(
        before=pre_forgetting_model,
        after=post_forgetting_model,
        dataloader=federated_dataset.test_loader,
        num_classes=federated_dataset.num_classes,
        device=device,
        ground_truth=args.target_class,
        gradient_batches=args.gradient_batches if args.gradient_batches > 0 else None,
        gradient_filter=gradient_filter,
    )
    LOGGER.info("Label inference prediction: %d (ground truth: %d)", inference.predicted_class, args.target_class)
    if inference.gradient_delta is not None:
        gradient_norms = {name: tensor.norm().item() for name, tensor in inference.gradient_delta.items()}
        LOGGER.debug("Gradient delta norms: %s", gradient_norms)

    per_class_records = []
    for cls in range(federated_dataset.num_classes):
        before_acc = float(inference.per_class_before[cls].item())
        after_acc = float(inference.per_class_after[cls].item())
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

    class_label = None
    if federated_dataset.class_names and inference.predicted_class < len(federated_dataset.class_names):
        class_label = str(federated_dataset.class_names[inference.predicted_class])

    if args.reconstruction_method == "gradient":
        reconstructor = GradientReconstructor(
            ReconstructionConfig(
                steps=args.reconstruction_steps,
                lr=args.reconstruction_lr,
                lambda_after=args.reconstruction_lambda,
                total_variation=args.reconstruction_tv,
                device=device,
            )
        )
        reconstructions = reconstructor.reconstruct(
            pre_forgetting_model,
            post_forgetting_model,
            target_class=inference.predicted_class,
            num_samples=args.reconstructions,
            input_shape=INPUT_SHAPES[args.dataset],
        )
    else:
        diffusion_config = DiffusionConfig(
            model_id=args.diffusion_model_id,
            guidance_scale=args.diffusion_guidance,
            num_inference_steps=args.diffusion_steps,
            device=device,
            negative_prompt=args.diffusion_negative_prompt,
        )
        try:
            diffusion = DiffusionReconstructor(diffusion_config)
            reconstructions = diffusion.reconstruct(
                target_class=inference.predicted_class,
                num_samples=args.reconstructions,
                class_label=class_label,
            )
        except RuntimeError as exc:
            LOGGER.error("Diffusion reconstruction failed: %s", exc)
            LOGGER.info("Falling back to gradient-based reconstruction")
            reconstructor = GradientReconstructor(
                ReconstructionConfig(
                    steps=args.reconstruction_steps,
                    lr=args.reconstruction_lr,
                    lambda_after=args.reconstruction_lambda,
                    total_variation=args.reconstruction_tv,
                    device=device,
                )
            )
            reconstructions = reconstructor.reconstruct(
                pre_forgetting_model,
                post_forgetting_model,
                target_class=inference.predicted_class,
                num_samples=args.reconstructions,
                input_shape=INPUT_SHAPES[args.dataset],
            )

    # Ensure reconstruction tensor matches dataset channel/size when using diffusion models.
    expected_shape = INPUT_SHAPES[args.dataset]
    if reconstructions.shape[1:] != expected_shape:
        LOGGER.debug(
            "Resizing reconstructions from %s to expected shape %s", reconstructions.shape[1:], expected_shape
        )
        reconstructions = F.interpolate(
            reconstructions,
            size=expected_shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        if expected_shape[0] == 1 and reconstructions.shape[1] == 3:
            reconstructions = reconstructions.mean(dim=1, keepdim=True)

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(reconstructions, args.output / "reconstructed.pt")
    torch.save(
        {
            "original": forgetting_result.original_state,
            "forgotten": forgetting_result.forgotten_state,
        },
        args.output / "models.pt",
    )
    if inference.gradient_delta is not None:
        torch.save(
            {
                "before": inference.gradient_before,
                "after": inference.gradient_after,
                "delta": inference.gradient_delta,
            },
            args.output / "gradient_deltas.pt",
        )
    with (args.output / "inference.json").open("w", encoding="utf-8") as handle:
        json.dump(inference.to_dict(), handle, indent=2)
    metadata = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "forget_rounds": args.forget_rounds,
        "baseline_accuracy": baseline_accuracy,
        "post_accuracy": post_accuracy,
        "target_class": args.target_class,
        "target_class_sample_count": target_sample_count,
        "predicted_class": inference.predicted_class,
        "attack_success": inference.predicted_class == args.target_class,
        "dp_sigma": args.dp_sigma,
        "dp_clip": args.dp_clip,
        "dp_mechanism": args.dp_mechanism,
        "secure_aggregation": args.secure_aggregation,
        "reconstruction_method": args.reconstruction_method,
        "class_label": class_label,
        "gradient_batches": args.gradient_batches,
        "gradient_params": gradient_filter,
        "diffusion_model_id": args.diffusion_model_id if args.reconstruction_method == "diffusion" else None,
        "heatmaps": {
            "enabled": not args.no_heatmaps,
            "cmap": args.heatmap_cmap,
        },
        "per_class_accuracy": per_class_records,
    }

    if not args.no_heatmaps:
        heatmap_artifacts = export_heatmaps(
            inference,
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

    LOGGER.info("Saved %d reconstructed samples and inference metadata to %s", len(reconstructions), args.output)


def perform_forgetting(
    server: FederatedServer,
    dataset: FederatedDataset,
    client_config: ClientConfig,
    target_class: int,
    rounds: int,
    method: str,
    input_shape: Sequence[int],
    method_config: FedEraserConfig | FedAFConfig | OneShotClassUnlearningConfig,
) -> ForgettingResult:
    return forget_class(
        server=server,
        dataset=dataset,
        client_config=client_config,
        target_class=target_class,
        rounds=rounds,
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