"""Mechanisms for class-level forgetting in federated settings."""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch import nn

from ..data.datasets import FederatedDataset
from ..federated.client import Client, ClientConfig
from ..federated.fedavg import FederatedServer, ServerConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class ForgettingResult:
    original_state: dict
    forgotten_state: dict
    target_class: int


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
    rounds: int,
    method: str = "fine_tune",
    input_shape: Sequence[int] | None = None,
) -> ForgettingResult:
    """Perform class forgetting by removing data and retraining."""
    original_state = copy.deepcopy(server.state_dict())

    LOGGER.info("Removing class %d from all client datasets", target_class)
    dataset.remove_class(target_class)
    server.clients = reinitialize_clients(dataset, client_config)

    if method == "fine_tune":
        LOGGER.info("Fine-tuning the global model for %d rounds after class removal", rounds)
        server.train(rounds)
    elif method == "logit_suppression":
        if input_shape is None:
            raise ValueError("Input shape is required for logit suppression method")
        LOGGER.info("Applying logit suppression for class %d", target_class)
        suppress_class_logits(
            server.global_model,
            target_class,
            input_shape=input_shape,
            steps=200,
            device=client_config.device,
        )
    elif method == "classifier_reinit":
        LOGGER.info("Reinitialising classifier weights for class %d", target_class)
        reinitialize_classifier_head(server.global_model, target_class)
        if rounds > 0:
            LOGGER.info("Stabilising classifier via %d fine-tuning rounds", rounds)
            server.train(rounds)
    else:
        raise ValueError(f"Unknown forgetting method: {method}")

    forgotten_state = copy.deepcopy(server.state_dict())
    return ForgettingResult(original_state=original_state, forgotten_state=forgotten_state, target_class=target_class)


def suppress_class_logits(
    model: nn.Module,
    target_class: int,
    input_shape: Sequence[int],
    steps: int = 200,
    device: torch.device | None = None,
) -> None:
    """Perturb the model to suppress the logits of ``target_class``.

    This method complements fine-tuning by explicitly reducing the target class's
    logits. It runs on synthetic noise inputs to avoid reintroducing forgotten data.
    """
    device = device or torch.device("cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for step in range(steps):
        noise = torch.randn(32, *input_shape, device=device)
        optimizer.zero_grad()
        logits = model(noise)
        target_logit = logits[:, target_class]
        loss = target_logit.mean()
        loss.backward()
        optimizer.step()
        if (step + 1) % 50 == 0:
            LOGGER.debug("Suppression step %d/%d (loss=%.4f)", step + 1, steps, loss.item())
    model.cpu()


def reinitialize_classifier_head(model: nn.Module, target_class: int) -> None:
    """Randomly reinitialise the final layer weights for ``target_class``."""

    classifier: nn.Linear | None = None
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            classifier = module
            break

    if classifier is None:
        raise RuntimeError("Unable to locate a linear classifier head in the model")

    if target_class >= classifier.out_features:
        raise ValueError(
            f"Target class {target_class} exceeds classifier output dim {classifier.out_features}"
        )

    with torch.no_grad():
        nn.init.kaiming_uniform_(classifier.weight[target_class : target_class + 1], a=5 ** 0.5)
        if classifier.bias is not None:
            classifier.bias[target_class].zero_()



__all__ = [
    "forget_class",
    "ForgettingResult",
    "suppress_class_logits",
    "reinitialize_clients",
    "reinitialize_classifier_head",
]
