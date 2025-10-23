"""Abstractions describing secure aggregation in the simulator."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class SecureAggregation:
    """Metadata describing whether secure aggregation is enabled.

    The actual averaging logic is implemented in :mod:`src.federated.fedavg`.
    This wrapper simply records the decision so that experiment metadata can be
    annotated consistently.
    """

    enabled: bool = False

    def apply(self, tensors: Iterable[torch.Tensor]) -> None:  # pragma: no cover - metadata only
        """Placeholder hook for future extensions."""
        if not self.enabled:
            return
        # In a real implementation, this would mask each tensor before
        # transmission. We leave it as a stub for integration with cryptographic
        # protocols.


__all__ = ["SecureAggregation"]
