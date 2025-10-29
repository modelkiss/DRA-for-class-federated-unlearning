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
from .dp import DifferentialPrivacyConfig, DifferentialPrivacyController, create_dp_controller

LOGGER = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    device: torch.device = torch.device("cpu")
    fraction: float = 1.0
    dp_config: DifferentialPrivacyConfig | None = None
    secure_aggregation: str | None = None
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
        self.dp_controller: DifferentialPrivacyController | None = create_dp_controller(
            config.dp_config, seed=config.seed
        )

    def _select_clients(self) -> Sequence[Client]:
        if self.config.fraction == 1.0:
            return self.clients
        num_selected = max(1, int(self.config.fraction * len(self.clients)))
        return random.sample(self.clients, num_selected)

    def run_round(
            self,
            *,
            round_index: int | None = None,
            total_rounds: int | None = None,
    ) -> None:
        participants = self._select_clients()
        LOGGER.info("Running round with %d clients", len(participants))

        if round_index is None:
            round_index = len(self.history)
        if self.dp_controller is not None:
            self.dp_controller.prepare_round(round_index, total_rounds)

        global_state = self.global_model.state_dict()
        updates: List[Tuple[OrderedDict[str, torch.Tensor], int]] = []

        for client in participants:
            client_model = copy.deepcopy(self.global_model)
            trained_model = client.train(client_model)
            updates.append((copy.deepcopy(trained_model.state_dict()), client.num_samples))

        new_state = self._aggregate(
            global_state,
            updates,
            round_index=round_index,
            total_rounds=total_rounds,
        )
        self.global_model.load_state_dict(new_state)

    def _aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: List[Tuple[OrderedDict[str, torch.Tensor], int]],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        processed_updates = (
            self.dp_controller.preprocess_updates(global_state, updates)
            if self.dp_controller is not None
            else updates
        )
        total_samples = sum(num_samples for _, num_samples in processed_updates)
        aggregated: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, param in global_state.items():
            aggregated[name] = param.clone()

        for state_dict, num_samples in processed_updates:
            weight = num_samples / max(total_samples, 1)
            for name, tensor in state_dict.items():
                diff = tensor - global_state[name]
                aggregated[name] += (diff * weight).to(aggregated[name].dtype)

        if self.dp_controller is not None:
            aggregated = self.dp_controller.postprocess_aggregate(
                global_state,
                aggregated,
                round_index=round_index,
                total_rounds=total_rounds,
            )

        if self.config.secure_aggregation not in (None, "none"):
            LOGGER.info(
                "Secure aggregation enabled using '%s'; only aggregated model is revealed",
                self.config.secure_aggregation,
            )
        return aggregated

    def train(self, num_rounds: int) -> nn.Module:
        for current_round in range(num_rounds):
            LOGGER.info("Starting federated round %d/%d", current_round + 1, num_rounds)
            self.run_round(round_index=current_round, total_rounds=num_rounds)
            self.history.append(TrainingState(round=current_round + 1))
        return self.global_model

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return copy.deepcopy(self.global_model.state_dict())

    def load_state_dict(self, state_dict: OrderedDict[str, torch.Tensor]) -> None:
        self.global_model.load_state_dict(state_dict)


__all__ = ["FederatedServer", "ServerConfig", "TrainingState"]