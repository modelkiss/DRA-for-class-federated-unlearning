"""Helpers for documenting differential privacy hyper-parameters."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DifferentialPrivacyConfig:
    """Configuration metadata for DP-enabled runs."""

    sigma: float | None = None
    clip: float | None = None
    target_epsilon: float | None = None
    target_delta: float | None = None

    def enabled(self) -> bool:
        return self.sigma is not None and self.sigma > 0


__all__ = ["DifferentialPrivacyConfig"]
