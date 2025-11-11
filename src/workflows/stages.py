"""Decomposed stage implementations for the class-level unlearning workflow."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Sequence

import torch

from src.attacks.label_inference import LabelInferenceResult, infer_forgotten_label
from src.attacks.sensitive_feature_inference import (
    SensitiveFeatureConfig,
    run_sensitive_feature_inference,
)
from src.attacks.data_reconstruction import (
    AdaptiveGenerationConfig,
    DataReconstructionConfig,
    DiffusionTrainingSummary,
    run_diffusion_generation,
    run_diffusion_training,
)
from src.data.datasets import FederatedDataConfig, create_federated_dataloaders
from src.defenses.differential_privacy import DifferentialPrivacyConfig
from src.federated.aggregation import AggregationConfig
from src.federated.client import Client, ClientConfig
from src.federated.fedavg import FederatedServer, ServerConfig
from src.forgetting.class_forgetting import (
    FedAFConfig,
    FedEraserConfig,
    OneShotClassUnlearningConfig,
    forget_class,
)
from src.models.nets import build_model
from src.utils.metrics import accuracy, per_class_accuracy
from src.utils.normalization import get_normalization_stats

INPUT_SHAPES = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "mnist": (1, 28, 28),
    "fashionmnist": (1, 28, 28),
    "megaface": (3, 128, 128),
}

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_model(dataset: str, num_classes: int, weights: Path, device: torch.device) -> torch.nn.Module:
    model = build_model(dataset, num_classes)
    state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    return model


# ---------------------------------------------------------------------------
# Stage 1 – Model training
# ---------------------------------------------------------------------------


@dataclass
class TrainingStageConfig:
    dataset: str = "cifar10"
    num_clients: int = 10
    iid: bool = True
    dirichlet_alpha: float = 0.5
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    client_learning_rate: float = 0.01
    local_epochs: int = 1
    client_batch_size: int = 64
    client_fraction: float = 1.0
    max_rounds: int = 50
    target_class_accuracy: float = 0.8
    dp_config: DifferentialPrivacyConfig = field(default_factory=DifferentialPrivacyConfig)
    device: str | None = None
    output_dir: Path = Path("outputs/stages/training")


@dataclass
class TrainingStageResult:
    model_path: Path
    aggregator_steps: int
    per_class_accuracy: dict[int, float]
    overall_accuracy: float
    config: dict

    def to_dict(self) -> dict:
        return {
            "model_path": str(self.model_path),
            "aggregator_steps": self.aggregator_steps,
            "per_class_accuracy": {int(k): float(v) for k, v in self.per_class_accuracy.items()},
            "overall_accuracy": float(self.overall_accuracy),
            "config": self.config,
        }


def run_model_training_stage(config: TrainingStageConfig) -> TrainingStageResult:
    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Training stage on device %s", device)

    aggregation_batch = config.client_batch_size
    federated_config = FederatedDataConfig(
        dataset=config.dataset,
        num_clients=config.num_clients,
        batch_size=aggregation_batch,
        iid=config.iid,
        dirichlet_alpha=config.dirichlet_alpha,
        split_seed=config.num_clients + int(config.iid) * 17,
    )
    dataset = create_federated_dataloaders(federated_config)

    model = build_model(config.dataset, dataset.num_classes)
    client_config = ClientConfig(
        learning_rate=config.client_learning_rate,
        local_epochs=config.local_epochs,
        device=device,
        proximal_mu=None,
    )

    clients = [Client(client_id=i, dataloader=loader, config=client_config) for i, loader in dataset.train_loaders.items()]

    server_config = ServerConfig(
        device=device,
        fraction=config.client_fraction,
        dp_config=config.dp_config,
        aggregation=config.aggregation,
    )
    server = FederatedServer(model=model, clients=clients, config=server_config)

    best_per_class = None
    best_overall = 0.0
    aggregator_steps = 0

    target_threshold = torch.tensor(config.target_class_accuracy, device=device)
    max_rounds = max(1, config.max_rounds)

    while aggregator_steps < max_rounds:
        LOGGER.info("Starting aggregation step %d", aggregator_steps + 1)
        server.run_round(round_index=aggregator_steps, total_rounds=max_rounds)
        aggregator_steps += 1

        eval_model = server.global_model.to(device)
        per_class = per_class_accuracy(eval_model, dataset.test_loader, dataset.num_classes, device)
        overall = accuracy(eval_model, dataset.test_loader, device)
        LOGGER.info("Evaluation after %d aggregations: overall=%.4f", aggregator_steps, overall)
        LOGGER.debug("Per-class accuracies: %s", per_class)

        if best_per_class is None or overall > best_overall:
            best_per_class = per_class.detach().clone()
            best_overall = overall

        if torch.all(per_class >= target_threshold):
            LOGGER.info("Target per-class accuracy %.2f reached for all classes after %d aggregations", config.target_class_accuracy, aggregator_steps)
            best_per_class = per_class.detach().clone()
            best_overall = overall
            break
    else:
        raise RuntimeError(
            f"Per-class accuracy target {config.target_class_accuracy:.2f} not reached within {config.max_rounds} aggregation steps"
        )

    output_dir = _ensure_directory(Path(config.output_dir))
    model_path = output_dir / "federated_model.pt"
    torch.save(server.global_model.state_dict(), model_path)

    per_class_dict = {int(idx): float(value) for idx, value in enumerate(best_per_class.cpu())}
    config_dict = {
        "dataset": config.dataset,
        "num_clients": config.num_clients,
        "iid": config.iid,
        "dirichlet_alpha": config.dirichlet_alpha,
        "client_batch_size": config.client_batch_size,
        "client_learning_rate": config.client_learning_rate,
        "local_epochs": config.local_epochs,
        "client_fraction": config.client_fraction,
        "aggregation": asdict(config.aggregation),
        "dp": {"method": config.dp_config.method, "parameters": config.dp_config.parameters},
    }
    result = TrainingStageResult(
        model_path=model_path,
        aggregator_steps=aggregator_steps,
        per_class_accuracy=per_class_dict,
        overall_accuracy=float(best_overall),
        config=config_dict,
    )

    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)

    LOGGER.info("Training stage completed with %d aggregations; model saved to %s", aggregator_steps, model_path)
    return result


# ---------------------------------------------------------------------------
# Stage 2 – Model forgetting
# ---------------------------------------------------------------------------


@dataclass
class ForgettingStageConfig:
    dataset: str | None
    target_class: int
    training_summary: Path
    method: str = "oneshot"
    output_dir: Path = Path("outputs/stages/forgetting")
    client_learning_rate: float | None = None
    local_epochs: int | None = None
    client_batch_size: int | None = None
    client_fraction: float | None = None
    iid: bool | None = None
    dirichlet_alpha: float | None = None
    num_clients: int | None = None
    aggregation: AggregationConfig | None = None
    dp_config: DifferentialPrivacyConfig | None = None
    device: str | None = None
    fedaf_rounds: int = 5


@dataclass
class ForgettingStageResult:
    pre_model_path: Path
    post_model_path: Path
    forgetting_summary: dict

    def to_dict(self) -> dict:
        return {
            "pre_model_path": str(self.pre_model_path),
            "post_model_path": str(self.post_model_path),
            "forgetting_summary": self.forgetting_summary,
        }


def run_model_forgetting_stage(config: ForgettingStageConfig) -> ForgettingStageResult:
    training_metadata = json.loads(Path(config.training_summary).read_text(encoding="utf-8"))
    trained_model_path = Path(training_metadata["model_path"])
    training_config = training_metadata.get("config", {})

    dataset_name = config.dataset or training_config.get("dataset")
    if dataset_name is None:
        raise ValueError("Dataset must be provided either via ForgettingStageConfig or training summary")

    num_clients = config.num_clients if config.num_clients is not None else training_config.get("num_clients", 1)
    client_batch_size = config.client_batch_size if config.client_batch_size is not None else training_config.get("client_batch_size", 64)
    iid = config.iid if config.iid is not None else training_config.get("iid", True)
    dirichlet_alpha = config.dirichlet_alpha if config.dirichlet_alpha is not None else training_config.get("dirichlet_alpha", 0.5)
    client_fraction = config.client_fraction if config.client_fraction is not None else training_config.get("client_fraction", 1.0)
    local_epochs = config.local_epochs if config.local_epochs is not None else training_config.get("local_epochs", 1)
    client_lr = config.client_learning_rate if config.client_learning_rate is not None else training_config.get("client_learning_rate", 0.01)

    aggregation_dict = training_config.get("aggregation", {})
    aggregation_cfg = config.aggregation if config.aggregation is not None else AggregationConfig(**aggregation_dict)

    dp_dict = training_config.get("dp", {"method": "none", "parameters": {}})
    dp_cfg = config.dp_config if config.dp_config is not None else DifferentialPrivacyConfig(
        method=dp_dict.get("method", "none"), parameters=dp_dict.get("parameters", {})
    )

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    federated_config = FederatedDataConfig(
        dataset=dataset_name,
        num_clients=num_clients,
        batch_size=client_batch_size,
        iid=iid,
        dirichlet_alpha=dirichlet_alpha,
        split_seed=num_clients + config.target_class,
    )
    dataset = create_federated_dataloaders(federated_config)

    base_model = build_model(dataset_name, dataset.num_classes)
    base_model.load_state_dict(torch.load(trained_model_path, map_location=device))

    client_cfg = ClientConfig(
        learning_rate=client_lr,
        local_epochs=local_epochs,
        device=device,
        proximal_mu=None,
    )
    clients = [Client(client_id=i, dataloader=loader, config=client_cfg) for i, loader in dataset.train_loaders.items()]
    server_cfg = ServerConfig(
        device=device,
        fraction=client_fraction,
        dp_config=dp_cfg,
        aggregation=aggregation_cfg,
    )
    server = FederatedServer(model=base_model, clients=clients, config=server_cfg)

    if config.method == "fed_eraser":
        method_cfg = FedEraserConfig()
    elif config.method == "fedaf":
        method_cfg = FedAFConfig(optimisation_rounds=config.fedaf_rounds)
    else:
        method_cfg = OneShotClassUnlearningConfig()

    forgetting = forget_class(
        server=server,
        dataset=dataset,
        client_config=client_cfg,
        target_class=config.target_class,
        method=config.method,
        input_shape=INPUT_SHAPES[dataset_name],
        method_config=method_cfg,
    )

    output_dir = _ensure_directory(Path(config.output_dir))
    pre_model_path = output_dir / "model_before.pt"
    post_model_path = output_dir / "model_after.pt"

    torch.save(forgetting.original_state, pre_model_path)
    torch.save(forgetting.forgotten_state, post_model_path)

    post_model = _load_model(dataset_name, dataset.num_classes, post_model_path, device)
    baseline_model = _load_model(dataset_name, dataset.num_classes, pre_model_path, device)
    baseline_acc = accuracy(baseline_model.to(device), dataset.test_loader, device)
    post_acc = accuracy(post_model.to(device), dataset.test_loader, device)

    summary = {
        "target_class": int(config.target_class),
        "method": config.method,
        "baseline_accuracy": baseline_acc,
        "post_accuracy": post_acc,
        "dataset": dataset_name,
    }

    with (output_dir / "forgetting_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                **summary,
                "pre_model_path": str(pre_model_path),
                "post_model_path": str(post_model_path),
                "training_summary": training_metadata,
            },
            handle,
            indent=2,
        )

    LOGGER.info("Forgetting stage completed: baseline=%.4f, post=%.4f", baseline_acc, post_acc)
    return ForgettingStageResult(
        pre_model_path=pre_model_path,
        post_model_path=post_model_path,
        forgetting_summary=summary,
    )


# ---------------------------------------------------------------------------
# Stage 3 & 4 – Label inference
# ---------------------------------------------------------------------------


@dataclass
class LabelInferenceStageConfig:
    dataset: str | None
    forgetting_summary: Path | None
    output_dir: Path = Path("outputs/stages/label_inference")
    device: str | None = None


@dataclass
class LabelInferenceStageOneResult:
    inference_path: Path
    candidate_labels: list[int]
    accuracy_after: float

    def to_dict(self) -> dict:
        return {
            "inference_path": str(self.inference_path),
            "candidate_labels": [int(label) for label in self.candidate_labels],
            "accuracy_after": float(self.accuracy_after),
        }


@dataclass
class LabelInferenceStageTwoResult:
    predicted_label: int
    score: float

    def to_dict(self) -> dict:
        return {"predicted_label": int(self.predicted_label), "score": float(self.score)}


def run_label_inference_stage_one(config: LabelInferenceStageConfig) -> LabelInferenceStageOneResult:
    if config.forgetting_summary is None:
        raise ValueError("Forgetting summary path is required for label inference stage one")
    forgetting_data = json.loads(Path(config.forgetting_summary).read_text(encoding="utf-8"))
    training_summary = forgetting_data["training_summary"] if "training_summary" in forgetting_data else None
    if training_summary is None:
        raise ValueError("Forgetting summary must contain embedded training metadata")

    training_config = training_summary.get("config", {})
    dataset_name = config.dataset or forgetting_data.get("dataset") or training_config.get("dataset")
    if dataset_name is None:
        raise ValueError("Dataset must be provided for label inference stage")
    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    federated_config = FederatedDataConfig(
        dataset=dataset_name,
        num_clients=training_config.get("num_clients", 1),
        batch_size=training_config.get("client_batch_size", 64),
        iid=training_config.get("iid", True),
        dirichlet_alpha=training_config.get("dirichlet_alpha", 0.5),
        split_seed=training_summary.get("aggregator_steps", 0),
    )
    dataset = create_federated_dataloaders(federated_config)

    pre_model_path = Path(forgetting_data["pre_model_path"]) if "pre_model_path" in forgetting_data else Path(config.output_dir) / "model_before.pt"
    post_model_path = Path(forgetting_data["post_model_path"]) if "post_model_path" in forgetting_data else Path(config.output_dir) / "model_after.pt"

    pre_model = _load_model(dataset_name, dataset.num_classes, pre_model_path, device)
    post_model = _load_model(dataset_name, dataset.num_classes, post_model_path, device)

    inference = infer_forgotten_label(
        before=pre_model.to(device),
        after=post_model.to(device),
        dataloader=dataset.test_loader,
        num_classes=dataset.num_classes,
        device=device,
        ground_truth=forgetting_data.get("target_class"),
        heatmap_samples=0,
        heatmap_border_ratio=0.0,
        transform=None,
    )

    _, indices = torch.sort(inference.score_vector, descending=True)
    num_candidates = max(1, math.ceil(dataset.num_classes * 0.6))
    candidate_indices = indices[:num_candidates].tolist()

    post_accuracy = accuracy(post_model.to(device), dataset.test_loader, device)

    inference_dir = _ensure_directory(Path(config.output_dir))
    inference_path = inference_dir / "label_inference.pt"
    torch.save(inference, inference_path)

    summary = LabelInferenceStageOneResult(
        inference_path=inference_path,
        candidate_labels=[int(label) for label in candidate_indices],
        accuracy_after=float(post_accuracy),
    )

    with (inference_dir / "label_inference_stage1.json").open("w", encoding="utf-8") as handle:
        json.dump(summary.to_dict(), handle, indent=2)

    return summary


def run_label_inference_stage_two(config: LabelInferenceStageConfig) -> LabelInferenceStageTwoResult:
    stage1_path = Path(config.output_dir) / "label_inference.pt"
    if not stage1_path.exists():
        raise FileNotFoundError("Stage one inference artefact not found; run stage one first")
    inference: LabelInferenceResult = torch.load(stage1_path, map_location="cpu")
    score_vector = inference.score_vector
    predicted_index = int(torch.argmax(score_vector).item())
    score = float(score_vector[predicted_index].item())
    result = LabelInferenceStageTwoResult(predicted_label=predicted_index, score=score)

    with (Path(config.output_dir) / "label_inference_stage2.json").open("w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)
    return result


# ---------------------------------------------------------------------------
# Stage 5 – Sensitive feature inference
# ---------------------------------------------------------------------------


@dataclass
class SensitiveFeatureStageConfig:
    dataset: str
    inference_dir: Path
    output_dir: Path = Path("outputs/stages/sensitive_features")
    device: str | None = None
    max_classes: int = 1
    mask_quantile: float = 0.8
    mask_min_threshold: float = 0.2
    patch_size: int = 16
    num_patches: int = 8
    dct_components: int = 32
    num_prototypes: int = 4


@dataclass
class SensitiveFeatureStageResult:
    summary_path: Path


def run_sensitive_feature_stage(config: SensitiveFeatureStageConfig) -> SensitiveFeatureStageResult:
    inference_path = Path(config.inference_dir) / "label_inference.pt"
    if not inference_path.exists():
        raise FileNotFoundError("Label inference artefact missing. Run label inference first.")

    inference: LabelInferenceResult = torch.load(inference_path, map_location="cpu")

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = get_normalization_stats(config.dataset)

    cfg = SensitiveFeatureConfig(
        max_classes=config.max_classes,
        mask_quantile=config.mask_quantile,
        mask_min_threshold=config.mask_min_threshold,
        patch_size=config.patch_size,
        num_patches=config.num_patches,
        dct_components=config.dct_components,
        num_prototypes=config.num_prototypes,
    )

    output_dir = _ensure_directory(Path(config.output_dir))
    summary = run_sensitive_feature_inference(
        inference,
        output_root=output_dir,
        dataset=config.dataset,
        normalization_stats=stats,
        config=cfg,
    )

    summary_path = output_dir / "sensitive_feature_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return SensitiveFeatureStageResult(summary_path=summary_path)


# ---------------------------------------------------------------------------
# Stage 6 & 7 – Diffusion training and generation
# ---------------------------------------------------------------------------


@dataclass
class DiffusionTrainingStageConfig:
    dataset: str | None
    forgetting_summary: Path
    inference_dir: Path
    sensitive_summary: Path | None = None
    output_dir: Path = Path("outputs/stages/diffusion_training")
    device: str | None = None
    config: DataReconstructionConfig = field(default_factory=DataReconstructionConfig)


@dataclass
class DiffusionTrainingStageResult:
    summary_path: Path


@dataclass
class DiffusionGenerationStageConfig:
    training_dir: Path
    inference_dir: Path
    forgetting_summary: Path
    sensitive_summary: Path | None = None
    output_dir: Path = Path("outputs/stages/diffusion_generation")
    device: str | None = None
    adaptive: AdaptiveGenerationConfig | None = None


@dataclass
class DiffusionGenerationStageResult:
    summary_path: Path


def run_diffusion_training_stage(config: DiffusionTrainingStageConfig) -> DiffusionTrainingStageResult:
    inference_path = Path(config.inference_dir) / "label_inference.pt"
    if not inference_path.exists():
        raise FileNotFoundError("Label inference artefact missing. Run label inference first.")

    inference: LabelInferenceResult = torch.load(inference_path, map_location="cpu")

    forgetting_data = json.loads(Path(config.forgetting_summary).read_text(encoding="utf-8"))
    training_summary = forgetting_data.get("training_summary", {})
    training_config = training_summary.get("config", {})

    dataset_name = config.dataset or forgetting_data.get("dataset") or training_config.get("dataset")
    if dataset_name is None:
        raise ValueError("Dataset must be specified for diffusion training stage")

    num_clients = training_config.get("num_clients", 1)
    batch_size = training_config.get("client_batch_size", 64)
    iid = training_config.get("iid", True)
    dirichlet_alpha = training_config.get("dirichlet_alpha", 0.5)
    split_seed = training_summary.get("aggregator_steps", 0) + (1 if iid else 0) * 17

    dataset = create_federated_dataloaders(
        FederatedDataConfig(
            dataset=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid,
            dirichlet_alpha=dirichlet_alpha,
            split_seed=split_seed,
        )
    )

    stats = get_normalization_stats(dataset_name)
    output_dir = _ensure_directory(Path(config.output_dir))
    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sfi_summary = None
    if config.sensitive_summary is not None and Path(config.sensitive_summary).exists():
        sfi_summary = json.loads(Path(config.sensitive_summary).read_text(encoding="utf-8"))

    summary = run_diffusion_training(
        inference,
        dataloader=dataset.test_loader,
        normalization_stats=stats,
        output_root=output_dir,
        device=device,
        config=config.config,
        sensitive_summary=sfi_summary,
        dataset_name=dataset_name,
    )

    summary_path = output_dir / "diffusion_training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(summary), handle, indent=2)

    return DiffusionTrainingStageResult(summary_path=summary_path)


def run_diffusion_generation_stage(config: DiffusionGenerationStageConfig) -> DiffusionGenerationStageResult:
    inference_path = Path(config.inference_dir) / "label_inference.pt"
    if not inference_path.exists():
        raise FileNotFoundError("Label inference artefact missing. Run label inference first.")
    inference: LabelInferenceResult = torch.load(inference_path, map_location="cpu")

    training_summary_path = Path(config.training_dir) / "diffusion_training_summary.json"
    if not training_summary_path.exists():
        raise FileNotFoundError("Diffusion training summary missing. Run diffusion training stage first.")
    training_summary = DiffusionTrainingSummary(**json.loads(training_summary_path.read_text(encoding="utf-8")))

    forgetting_data = json.loads(Path(config.forgetting_summary).read_text(encoding="utf-8"))
    training_meta = forgetting_data.get("training_summary", {})
    training_config = training_meta.get("config", {})
    dataset_name = training_summary.config.get("dataset") or forgetting_data.get("dataset") or training_config.get("dataset")
    if dataset_name is None:
        raise ValueError("Dataset must be specified for diffusion generation stage")

    num_clients = training_config.get("num_clients", 1)
    batch_size = training_config.get("client_batch_size", 64)
    iid = training_config.get("iid", True)
    dirichlet_alpha = training_config.get("dirichlet_alpha", 0.5)
    split_seed = training_meta.get("aggregator_steps", 0) + (1 if iid else 0) * 17

    dataset = create_federated_dataloaders(
        FederatedDataConfig(
            dataset=dataset_name,
            num_clients=num_clients,
            batch_size=batch_size,
            iid=iid,
            dirichlet_alpha=dirichlet_alpha,
            split_seed=split_seed,
        )
    )

    device = torch.device(config.device) if config.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = _ensure_directory(Path(config.output_dir))

    pre_model = _load_model(dataset_name, dataset.num_classes, Path(forgetting_data["pre_model_path"]), device)
    post_model = _load_model(dataset_name, dataset.num_classes, Path(forgetting_data["post_model_path"]), device)

    stats = get_normalization_stats(dataset_name)

    sfi_summary = None
    if config.sensitive_summary is not None and Path(config.sensitive_summary).exists():
        sfi_summary = json.loads(Path(config.sensitive_summary).read_text(encoding="utf-8"))

    adaptive_cfg = config.adaptive if config.adaptive is not None else training_summary.config.get("adaptive")
    result = run_diffusion_generation(
        inference,
        training_summary=training_summary,
        model_before=pre_model.to(device).eval(),
        model_after=post_model.to(device).eval(),
        normalization_stats=stats,
        output_root=output_dir,
        device=device,
        adaptive_config=adaptive_cfg,
        sensitive_summary=sfi_summary,
    )

    summary_path = output_dir / "diffusion_generation_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)

    return DiffusionGenerationStageResult(summary_path=summary_path)
