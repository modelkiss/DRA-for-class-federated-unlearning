"""End-to-end experiment pipeline for class-level federated unlearning."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import torch

from src.attacks.data_reconstruction import GradientReconstructor, ReconstructionConfig
from src.attacks.label_inference import LabelInferenceResult, infer_forgotten_label
from src.data.datasets import FederatedDataConfig, FederatedDataset, create_federated_dataloaders
from src.federated.client import Client, ClientConfig
from src.federated.fedavg import FederatedServer, ServerConfig
from src.forgetting.class_forgetting import ForgettingResult, forget_class
from src.models.nets import build_model
from src.utils.logging import setup_logging
from src.utils.metrics import accuracy

LOGGER = logging.getLogger(__name__)


INPUT_SHAPES: Dict[str, Sequence[int]] = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
}


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
    parser.add_argument("--secure-aggregation", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--reconstructions", type=int, default=8)
    parser.add_argument("--reconstruction-steps", type=int, default=400)
    parser.add_argument("--reconstruction-lr", type=float, default=0.1)
    parser.add_argument("--reconstruction-lambda", type=float, default=1.0)
    parser.add_argument("--reconstruction-tv", type=float, default=1e-4)
    parser.add_argument("--forgetting-method", default="fine_tune", choices=["fine_tune", "logit_suppression"])
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument("--no-heatmaps", action="store_true", help="Disable exporting heatmap visualisations")
    parser.add_argument(
        "--heatmap-cmap",
        default="coolwarm",
        help="Matplotlib colormap name used when rendering heatmaps",
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

    model = build_model(args.dataset, federated_dataset.num_classes)

    client_config = ClientConfig(
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        device=device,
    )
    clients = [Client(client_id=i, dataloader=loader, config=client_config) for i, loader in federated_dataset.train_loaders.items()]

    server_config = ServerConfig(
        device=device,
        fraction=args.fraction,
        dp_sigma=args.dp_sigma if args.dp_sigma > 0 else None,
        dp_clip=args.dp_clip,
        secure_aggregation=args.secure_aggregation,
    )
    server = FederatedServer(model=model, clients=clients, config=server_config)

    LOGGER.info("Starting federated pre-training for %d rounds", args.rounds)
    server.train(args.rounds)
    pre_forgetting_model = copy.deepcopy(server.global_model)

    baseline_accuracy = accuracy(pre_forgetting_model.to(device), federated_dataset.test_loader, device)
    LOGGER.info("Baseline accuracy before forgetting: %.4f", baseline_accuracy)

    forgetting_result = perform_forgetting(
        server=server,
        dataset=federated_dataset,
        client_config=client_config,
        target_class=args.target_class,
        rounds=args.forget_rounds,
        method=args.forgetting_method,
        input_shape=INPUT_SHAPES[args.dataset],
    )

    post_forgetting_model = copy.deepcopy(server.global_model)
    post_accuracy = accuracy(post_forgetting_model.to(device), federated_dataset.test_loader, device)
    LOGGER.info("Accuracy after forgetting: %.4f", post_accuracy)

    inference = infer_forgotten_label(
        before=pre_forgetting_model,
        after=post_forgetting_model,
        dataloader=federated_dataset.test_loader,
        num_classes=federated_dataset.num_classes,
        device=device,
        ground_truth=args.target_class,
    )
    LOGGER.info("Label inference prediction: %d (ground truth: %d)", inference.predicted_class, args.target_class)

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

    args.output.mkdir(parents=True, exist_ok=True)
    torch.save(reconstructions, args.output / "reconstructed.pt")
    torch.save(
        {
            "original": forgetting_result.original_state,
            "forgotten": forgetting_result.forgotten_state,
        },
        args.output / "models.pt",
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
        "predicted_class": inference.predicted_class,
        "attack_success": inference.predicted_class == args.target_class,
        "dp_sigma": args.dp_sigma,
        "dp_clip": args.dp_clip,
        "secure_aggregation": args.secure_aggregation,
        "heatmaps": {
            "enabled": not args.no_heatmaps,
            "cmap": args.heatmap_cmap,
        },
    }
    with (args.output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    if not args.no_heatmaps:
        export_heatmaps(
            inference,
            output_dir=args.output,
            cmap=args.heatmap_cmap,
            dataset=args.dataset,
        )

    LOGGER.info("Saved %d reconstructed samples and inference metadata to %s", len(reconstructions), args.output)


def perform_forgetting(
    server: FederatedServer,
    dataset: FederatedDataset,
    client_config: ClientConfig,
    target_class: int,
    rounds: int,
    method: str,
    input_shape: Sequence[int],
) -> ForgettingResult:
    return forget_class(
        server=server,
        dataset=dataset,
        client_config=client_config,
        target_class=target_class,
        rounds=rounds,
        method=method,
        input_shape=input_shape,
    )


def export_heatmaps(
    inference: LabelInferenceResult,
    output_dir: Path,
    cmap: str,
    dataset: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        LOGGER.warning("Matplotlib is not available, skipping heatmap export: %s", exc)
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_classes = inference.per_class_before.numel()
    class_indices = list(range(num_classes))

    # Per-class accuracy differences
    acc_diff = (
        inference.per_class_before - inference.per_class_after
    ).unsqueeze(0).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.6), 2.5))
    im = ax.imshow(acc_diff, cmap=cmap, aspect="auto")
    ax.set_title(f"Per-class accuracy delta ({dataset})")
    ax.set_yticks([0])
    ax.set_yticklabels(["Δ Acc"])
    ax.set_xticks(class_indices)
    ax.set_xticklabels(class_indices, rotation=45)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    per_class_path = output_dir / f"heatmap_accuracy_delta_{dataset}.png"
    fig.savefig(per_class_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Saved per-class heatmap to %s using cmap=%s", per_class_path, cmap)

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


if __name__ == "__main__":
    main()
