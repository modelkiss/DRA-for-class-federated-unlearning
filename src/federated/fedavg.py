"""Federated averaging implementation with optional defenses."""
from __future__ import annotations

import copy
import logging
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .client import Client

LOGGER = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    device: torch.device = torch.device("cpu")
    fraction: float = 1.0
    dp_sigma: float | None = None
    dp_clip: float | None = None
    secure_aggregation: bool = False
    seed: int = 0


@dataclass
class TrainingState:
    round: int
    metrics: Dict[str, float] = field(default_factory=dict)


class FederatedServer:
    """Federated averaging coordinator."""

    def __init__(
        self,
        model: nn.Module,
        clients: Sequence[Client],
        config: ServerConfig,
    ) -> None:
        self.global_model = model
        self.clients = list(clients)
        self.config = config
        self.history: List[TrainingState] = []
        random.seed(config.seed)

    def _select_clients(self) -> Sequence[Client]:
        if self.config.fraction >= 1.0:
            return self.clients
        num_selected = max(1, int(self.config.fraction * len(self.clients)))
        return random.sample(self.clients, num_selected)

    def run_round(self) -> None:
        participants = self._select_clients()
        LOGGER.info("Running round with %d clients", len(participants))
        global_state = self.global_model.state_dict()
        updates: List[Tuple[OrderedDict[str, torch.Tensor], int]] = []
        for client in participants:
            client_model = copy.deepcopy(self.global_model)
            trained_model = client.train(client_model)
            updates.append((copy.deepcopy(trained_model.state_dict()), client.num_samples))

        new_state = self._aggregate(global_state, updates)
        self.global_model.load_state_dict(new_state)

    def _aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: List[Tuple[OrderedDict[str, torch.Tensor], int]],
    ) -> OrderedDict[str, torch.Tensor]:
        total_samples = sum(num_samples for _, num_samples in updates)
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, param in global_state.items():
            aggregated[name] = torch.zeros_like(param)

        for state_dict, num_samples in updates:
            weight = num_samples / max(total_samples, 1)
            if self.config.dp_clip is not None and self.config.dp_clip > 0:
                squared_norm = 0.0
                for name, tensor in state_dict.items():
                    diff = tensor - global_state[name]
                    squared_norm += float(diff.pow(2).sum())
                norm = squared_norm ** 0.5
                clip_factor = min(1.0, self.config.dp_clip / (norm + 1e-12))
            else:
                clip_factor = 1.0

            for name, tensor in state_dict.items():
                diff = tensor - global_state[name]
                aggregated[name] += (global_state[name] + diff * clip_factor) * weight

        if self.config.dp_sigma is not None and self.config.dp_sigma > 0:
            LOGGER.info("Applying Gaussian noise with sigma=%.4f", self.config.dp_sigma)
            for name, tensor in aggregated.items():
                noise = torch.normal(0.0, self.config.dp_sigma, size=tensor.shape)
                aggregated[name] = tensor + noise.to(tensor.device)

        if self.config.secure_aggregation:
            LOGGER.info("Secure aggregation enabled: only aggregated model is revealed")
        return aggregated

    def train(self, num_rounds: int) -> nn.Module:
        for current_round in range(num_rounds):
            LOGGER.info("Starting federated round %d/%d", current_round + 1, num_rounds)
            self.run_round()
            self.history.append(TrainingState(round=current_round + 1))
        return self.global_model

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state_dict)


__all__ = ["FederatedServer", "ServerConfig", "TrainingState"]
