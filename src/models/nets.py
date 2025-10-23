"""Neural network architectures used in experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torchvision import models


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.block(x)


class CifarCNN(nn.Module):
    """Light-weight VGG style network for CIFAR datasets."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            BasicConvBlock(3, 64),
            BasicConvBlock(64, 128),
            BasicConvBlock(128, 256, pool=False),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        return self.classifier(x)


class LeNet(nn.Module):
    """LeNet style network for MNIST style datasets."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50 * 7 * 7, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.features(x)
        return self.classifier(x)


def resnet18_cifar(num_classes: int) -> nn.Module:
    """Return a ResNet18 tailored for CIFAR resolution."""
    model = models.resnet18(weights=None)
    # Adapt the first convolution for 32x32 inputs.
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


MODEL_REGISTRY = {
    "cifar_cnn": CifarCNN,
    "lenet": LeNet,
    "resnet18_cifar": resnet18_cifar,
}


def build_model(dataset: Literal["cifar10", "cifar100", "mnist", "fashionmnist"], num_classes: int) -> nn.Module:
    """Factory returning a default architecture for ``dataset``."""
    dataset = dataset.lower()
    if dataset in {"cifar10", "cifar100"}:
        return CifarCNN(num_classes)
    if dataset in {"mnist", "fashionmnist"}:
        return LeNet(num_classes)
    raise ValueError(f"Unsupported dataset: {dataset}")


__all__ = ["CifarCNN", "LeNet", "resnet18_cifar", "build_model", "MODEL_REGISTRY"]
