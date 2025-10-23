"""Client side utilities for federated training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


@dataclass
class ClientConfig:
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    local_epochs: int = 1
    device: torch.device = torch.device("cpu")


class Client:
    """Federated client responsible for local updates."""

    def __init__(self, client_id: int, dataloader: DataLoader, config: ClientConfig) -> None:
        self.client_id = client_id
        self.dataloader = dataloader
        self.config = config

    def train(self, model: nn.Module) -> nn.Module:
        model = model.to(self.config.device)
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )
        for _ in range(self.config.local_epochs):
            for inputs, targets in self.dataloader:
                inputs, targets = inputs.to(self.config.device), targets.to(self.config.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
        return model.cpu()

    @property
    def num_samples(self) -> int:
        return len(self.dataloader.dataset)


__all__ = ["Client", "ClientConfig"]
