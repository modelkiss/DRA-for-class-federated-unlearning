"""Dataset helpers for the federated unlearning experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


DatasetFactory = Callable[[bool], Dataset]


@dataclass
class FederatedDataConfig:
    """Configuration used when instantiating federated datasets."""

    dataset: str
    num_clients: int
    batch_size: int
    iid: bool = True
    dirichlet_alpha: float = 0.5
    num_workers: int = 4
    root: str = "./data"
    augment: bool = True


def _build_transforms(dataset: str, train: bool) -> transforms.Compose:
    if dataset in {"cifar10", "cifar100"}:
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    # MNIST style datasets
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def get_datasets(name: str, root: str = "./data", download: bool = True) -> Tuple[Dataset, Dataset, int]:
    """Return train/test datasets and number of classes for ``name``."""
    name = name.lower()
    if name == "cifar10":
        train = datasets.CIFAR10(root=root, train=True, transform=_build_transforms(name, True), download=download)
        test = datasets.CIFAR10(root=root, train=False, transform=_build_transforms(name, False), download=download)
        num_classes = 10
    elif name == "cifar100":
        train = datasets.CIFAR100(root=root, train=True, transform=_build_transforms(name, True), download=download)
        test = datasets.CIFAR100(root=root, train=False, transform=_build_transforms(name, False), download=download)
        num_classes = 100
    elif name == "mnist":
        train = datasets.MNIST(root=root, train=True, transform=_build_transforms(name, True), download=download)
        test = datasets.MNIST(root=root, train=False, transform=_build_transforms(name, False), download=download)
        num_classes = 10
    elif name == "fashionmnist":
        train = datasets.FashionMNIST(root=root, train=True, transform=_build_transforms(name, True), download=download)
        test = datasets.FashionMNIST(root=root, train=False, transform=_build_transforms(name, False), download=download)
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {name}")
    return train, test, num_classes


def _iid_partition(num_samples: int, num_clients: int) -> List[List[int]]:
    indices = torch.randperm(num_samples).tolist()
    return [indices[i::num_clients] for i in range(num_clients)]


def _dirichlet_partition(labels: Sequence[int], num_clients: int, alpha: float) -> List[List[int]]:
    labels = torch.tensor(labels)
    num_classes = labels.max().item() + 1
    class_indices = [torch.where(labels == cls)[0] for cls in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls_indices in class_indices:
        if len(cls_indices) == 0:
            continue
        proportions = torch.distributions.Dirichlet(alpha * torch.ones(num_clients)).sample()
        proportions = (proportions / proportions.sum()) * len(cls_indices)
        proportions = proportions.floor().to(torch.int64)
        diff = len(cls_indices) - proportions.sum().item()
        for i in range(diff):
            proportions[i % num_clients] += 1
        start = 0
        permuted = cls_indices[torch.randperm(len(cls_indices))]
        for client_id, count in enumerate(proportions):
            if count == 0:
                continue
            end = start + count.item()
            client_indices[client_id].extend(permuted[start:end].tolist())
            start = end
    return client_indices


@dataclass
class FederatedDataset:
    """Utility class wrapping client data partitions."""

    train_loaders: Dict[int, DataLoader]
    test_loader: DataLoader
    num_classes: int
    class_names: Sequence[str] | None = None

    def remove_class(self, target_class: int) -> None:
        """Filter out all examples of ``target_class`` from each client's loader."""
        for client_id, loader in list(self.train_loaders.items()):
            dataset = loader.dataset
            if isinstance(dataset, Subset):
                parent_dataset = dataset.dataset
                indices = [idx for idx in dataset.indices if int(parent_dataset[idx][1]) != target_class]
                self.train_loaders[client_id] = DataLoader(
                    Subset(parent_dataset, indices),
                    batch_size=loader.batch_size,
                    shuffle=True,
                    num_workers=loader.num_workers,
                    pin_memory=True,
                )
            else:
                indices = [idx for idx in range(len(dataset)) if int(dataset[idx][1]) != target_class]
                self.train_loaders[client_id] = DataLoader(
                    Subset(dataset, indices),
                    batch_size=loader.batch_size,
                    shuffle=True,
                    num_workers=loader.num_workers,
                    pin_memory=True,
                )


def create_federated_dataloaders(config: FederatedDataConfig) -> FederatedDataset:
    """Instantiate federated train/test loaders according to ``config``."""
    train_dataset, test_dataset, num_classes = get_datasets(config.dataset, root=config.root)

    if config.iid:
        indices_per_client = _iid_partition(len(train_dataset), config.num_clients)
    else:
        labels = [int(train_dataset[i][1]) for i in range(len(train_dataset))]
        indices_per_client = _dirichlet_partition(labels, config.num_clients, config.dirichlet_alpha)

    train_loaders = {}
    for client_id, indices in enumerate(indices_per_client):
        subset = Subset(train_dataset, indices)
        loader = DataLoader(
            subset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        train_loaders[client_id] = loader

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    class_names = getattr(train_dataset, "classes", None)
    if class_names is not None:
        class_names = list(class_names)

    return FederatedDataset(
        train_loaders=train_loaders,
        test_loader=test_loader,
        num_classes=num_classes,
        class_names=class_names,
    )


__all__ = [
    "FederatedDataConfig",
    "FederatedDataset",
    "create_federated_dataloaders",
    "get_datasets",
]
