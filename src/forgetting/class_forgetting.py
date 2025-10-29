"""Mechanisms for class-level forgetting in federated settings."""
from __future__ import annotations

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, fields
from typing import Dict, Iterable, List, MutableMapping, Sequence

import torch
from torch import nn

from ..data.datasets import FederatedDataset
from ..federated.client import Client, ClientConfig
from ..federated.fedavg import FederatedServer

LOGGER = logging.getLogger(__name__)


@dataclass
class ForgettingResult:
    original_state: Dict[str, torch.Tensor]
    forgotten_state: Dict[str, torch.Tensor]
    target_class: int


@dataclass
class FedEraserConfig:
    history_window: int = 5
    calibration_rounds: int = 3
    calibration_lr: float = 5e-3
    calibration_client_fraction: float = 0.5
    update_weight_decay: float = 1e-4


@dataclass
class FedAFConfig:
    forgetting_regularization: float = 0.25
    retention_strength: float = 0.2
    learning_rate: float = 1e-2
    class_mask_ratio: float = 0.3
    stop_threshold: float = 0.4
    optimisation_rounds: int = 3


@dataclass
class OneShotClassUnlearningConfig:
    projection_dim: int = 16
    replacement_strength: float = 0.5
    freeze_ratio: float = 0.4
    local_tuning_epochs: int = 1
    reconstruction_threshold: float = 0.2


MethodConfig = (
    FedEraserConfig | FedAFConfig | OneShotClassUnlearningConfig | Dict[str, float] | None
)


def reinitialize_clients(
    dataset: FederatedDataset,
    client_config: ClientConfig,
) -> List[Client]:
    clients: List[Client] = []
    for client_id, loader in dataset.train_loaders.items():
        clients.append(Client(client_id=client_id, dataloader=loader, config=client_config))
    return clients


def forget_class(
    server: FederatedServer,
    dataset: FederatedDataset,
    client_config: ClientConfig,
    target_class: int,
    method: str = "fed_eraser",
    input_shape: Sequence[int] | None = None,
    method_config: MethodConfig = None,
) -> ForgettingResult:
    """Perform class forgetting by removing data and retraining."""
    original_state = copy.deepcopy(server.state_dict())

    LOGGER.info("Removing class %d from all client datasets", target_class)
    dataset.remove_class(target_class)
    server.clients = reinitialize_clients(dataset, client_config)

    config = _coerce_config(method, method_config)

    if method == "fed_eraser":
        _apply_fed_eraser(server, dataset, client_config, config)  # type: ignore[arg-type]
    elif method == "fedaf":
        _apply_fedaf(
            server,
            dataset,
            client_config,
            target_class=target_class,
            original_state=original_state,
            config=config,  # type: ignore[arg-type]
        )
    elif method == "one_shot":
        _apply_one_shot_class_unlearning(
            server,
            dataset,
            client_config,
            target_class=target_class,
            original_state=original_state,
            config=config,  # type: ignore[arg-type]
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unknown forgetting method: {method}")

    forgotten_state = copy.deepcopy(server.state_dict())
    return ForgettingResult(original_state=original_state, forgotten_state=forgotten_state, target_class=target_class)


def _coerce_config(method: str, overrides: MethodConfig):
    if method == "fed_eraser":
        base_cls = FedEraserConfig
    elif method == "fedaf":
        base_cls = FedAFConfig
    elif method == "one_shot":
        base_cls = OneShotClassUnlearningConfig
    else:
        raise ValueError(f"Unsupported forgetting method: {method}")

    if isinstance(overrides, base_cls):
        return overrides
    if isinstance(overrides, dict):
        valid_fields = {field.name for field in fields(base_cls)}
        filtered = {key: overrides[key] for key in overrides.keys() & valid_fields}
        return base_cls(**filtered)
    return base_cls()


def _average_state_dicts(states: Iterable[MutableMapping[str, torch.Tensor]]) -> OrderedDict[str, torch.Tensor]:
    states = list(states)
    if not states:
        raise ValueError("At least one state dictionary is required for averaging")
    averaged: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in states[0].keys():
        stacked = torch.stack([state[key] for state in states])
        if stacked.dtype in (torch.float16, torch.float32, torch.float64):
            averaged[key] = stacked.mean(dim=0)
        else:
            averaged[key] = stacked[0]
    return averaged


def _locate_classifier(model: nn.Module) -> tuple[str, nn.Linear]:
    classifier: nn.Linear | None = None
    classifier_name = ""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            classifier = module
            classifier_name = name
    if classifier is None:
        raise RuntimeError("Unable to locate a linear classifier head in the model")
    return classifier_name, classifier


def _apply_fed_eraser(
    server: FederatedServer,
    dataset: FederatedDataset,
    base_client_config: ClientConfig,
    config: FedEraserConfig,
) -> None:
    LOGGER.info("Applying FedEraser with config: %s", config)
    history: List[Dict[str, torch.Tensor]] = []

    calibration_config = ClientConfig(
        learning_rate=config.calibration_lr,
        momentum=base_client_config.momentum,
        weight_decay=config.update_weight_decay,
        local_epochs=base_client_config.local_epochs,
        device=base_client_config.device,
    )

    original_fraction = server.config.fraction
    server.config.fraction = max(0.0, min(1.0, config.calibration_client_fraction))
    server.clients = reinitialize_clients(dataset, calibration_config)

    for current_round in range(max(0, config.calibration_rounds)):
        LOGGER.debug("FedEraser calibration round %d/%d", current_round + 1, config.calibration_rounds)
        server.train(1)
        history.append(copy.deepcopy(server.state_dict()))
        if len(history) > max(1, config.history_window):
            history.pop(0)

    if history:
        averaged = _average_state_dicts(history)
        server.load_state_dict(averaged)

    server.config.fraction = original_fraction
    server.clients = reinitialize_clients(dataset, base_client_config)


def _apply_fedaf(
        server: FederatedServer,
        dataset: FederatedDataset,
        base_client_config: ClientConfig,
        *,
        target_class: int,
        original_state: Dict[str, torch.Tensor],
        config: FedAFConfig,
) -> None:
    LOGGER.info("Applying FedAF with config: %s", config)
    training_config = ClientConfig(
        learning_rate=config.learning_rate,
        momentum=base_client_config.momentum,
        weight_decay=base_client_config.weight_decay,
        local_epochs=base_client_config.local_epochs,
        device=base_client_config.device,
    )
    server.clients = reinitialize_clients(dataset, training_config)

    classifier_name, _ = _locate_classifier(server.global_model)
    weight_key = f"{classifier_name}.weight"
    bias_key = f"{classifier_name}.bias"
    baseline_weights = original_state[weight_key].clone()
    baseline_bias = original_state.get(bias_key)
    baseline_norm = float(baseline_weights[target_class].norm().item())

    max_rounds = max(1, config.optimisation_rounds)
    stop_ratio = max(0.0, min(1.0, config.stop_threshold))
    mask_ratio = max(0.0, min(1.0, config.class_mask_ratio))
    retention = max(0.0, min(1.0, config.retention_strength))
    reg = max(0.0, config.forgetting_regularization)

    for current_round in range(max_rounds):
        LOGGER.debug("FedAF optimisation round %d/%d", current_round + 1, max_rounds)
        server.train(1)
        state = copy.deepcopy(server.state_dict())
        weights = state[weight_key]
        bias = state.get(bias_key)

        weights[target_class] *= 1.0 - reg
        if bias is not None:
            bias[target_class] *= 1.0 - reg

        if mask_ratio > 0:
            num_features = weights.shape[1]
            mask_count = max(1, int(mask_ratio * num_features))
            _, indices = torch.topk(weights[target_class].abs(), mask_count)
            weights[target_class, indices] = 0.0

        if retention > 0:
            reference_weights = baseline_weights.to(weights.device)
            for cls in range(weights.shape[0]):
                if cls == target_class:
                    continue
                weights[cls] = torch.lerp(weights[cls], reference_weights[cls], retention)
            if bias is not None and baseline_bias is not None:
                reference_bias = baseline_bias.to(bias.device)
                for cls in range(bias.shape[0]):
                    if cls == target_class:
                        continue
                    bias[cls] = torch.lerp(bias[cls], reference_bias[cls], retention)

        state[weight_key] = weights
        if bias is not None:
            state[bias_key] = bias
        server.load_state_dict(state)

        current_norm = float(weights[target_class].norm().item())
        if baseline_norm > 0 and current_norm <= baseline_norm * (1.0 - stop_ratio):
            LOGGER.info("FedAF early stopped after %d rounds (target norm %.4f)", current_round + 1, current_norm)
            break

    server.clients = reinitialize_clients(dataset, base_client_config)


def _apply_one_shot_class_unlearning(
        server: FederatedServer,
        dataset: FederatedDataset,
        base_client_config: ClientConfig,
        *,
        target_class: int,
        original_state: Dict[str, torch.Tensor],
        config: OneShotClassUnlearningConfig,
        input_shape: Sequence[int] | None,
) -> None:
    LOGGER.info("Applying one-shot class unlearning with config: %s", config)
    classifier_name, classifier = _locate_classifier(server.global_model)
    weight_key = f"{classifier_name}.weight"
    bias_key = f"{classifier_name}.bias"
    state = copy.deepcopy(server.state_dict())
    weights = state[weight_key]
    bias = state.get(bias_key)

    in_features = weights.shape[1]
    projection_dim = max(0, min(in_features, config.projection_dim))
    if projection_dim > 0:
        basis = torch.randn(in_features, projection_dim, device=weights.device, dtype=weights.dtype)
        q, _ = torch.linalg.qr(basis, mode="reduced")
        projection = q @ q.transpose(0, 1)
        weights[target_class] = weights[target_class] - weights[target_class] @ projection

    if config.replacement_strength > 0:
        strength = torch.as_tensor(config.replacement_strength, device=weights.device, dtype=weights.dtype)
        weights[target_class] = weights[target_class] + torch.randn_like(weights[target_class]) * strength
        if bias is not None:
            bias[target_class] = bias[target_class] + torch.randn_like(bias[target_class]) * strength

    state[weight_key] = weights
    if bias is not None:
        state[bias_key] = bias
    server.load_state_dict(state)

    trainable_params = [
        (name, param)
        for name, param in server.global_model.named_parameters()
        if param.requires_grad and not name.startswith(weight_key) and not name.startswith(bias_key)
    ]
    freeze_ratio = max(0.0, min(1.0, config.freeze_ratio))
    freeze_count = int(len(trainable_params) * freeze_ratio)
    frozen: List[tuple[nn.Parameter, bool]] = []
    for index, (_, parameter) in enumerate(trainable_params):
        if index < freeze_count:
            frozen.append((parameter, parameter.requires_grad))
            parameter.requires_grad = False

    tuning_config = ClientConfig(
        learning_rate=base_client_config.learning_rate,
        momentum=base_client_config.momentum,
        weight_decay=base_client_config.weight_decay,
        local_epochs=max(1, config.local_tuning_epochs),
        device=base_client_config.device,
    )
    server.clients = reinitialize_clients(dataset, tuning_config)

    LOGGER.debug("Running one-shot local fine-tuning")
    server.train(1)

    reference = original_state[weight_key].to(weights.device)
    updated_state = server.state_dict()
    updated_weights = updated_state[weight_key]
    diff = torch.norm(updated_weights[target_class] - reference[target_class]).item()
    if diff < config.reconstruction_threshold:
        LOGGER.info(
            "Reconstruction gap %.4f below threshold %.4f; performing an additional stabilisation round",
            diff,
            config.reconstruction_threshold,
        )
        server.train(1)

        for parameter, status in frozen:
            parameter.requires_grad = status

        server.clients = reinitialize_clients(dataset, base_client_config)


__all__ = [
    "forget_class",
    "ForgettingResult",
    "FedEraserConfig",
    "FedAFConfig",
    "OneShotClassUnlearningConfig",
    "reinitialize_clients",
]