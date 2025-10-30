"""Aggregation mechanisms for federated optimisation."""
from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import torch

LOGGER = logging.getLogger(__name__)

ClientUpdate = Tuple[OrderedDict[str, torch.Tensor], int]


@dataclass
class AggregationConfig:
    """User-specified aggregation mechanism configuration."""

    mechanism: str = "fedavg"
    parameters: dict[str, object] = field(default_factory=dict)


class AggregationMechanism:
    """Base class for aggregation strategies."""

    name: str = "aggregation"
    is_secure: bool = False

    def aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        raise NotImplementedError

    def _clone_global(self, global_state: OrderedDict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict((name, param.clone()) for name, param in global_state.items())


class _WeightedAveragingMixin:
    weighting: str = "samples"

    def _compute_weights(self, updates: Sequence[ClientUpdate]) -> List[float]:
        if not updates:
            return []
        if self.weighting == "samples":
            total_samples = sum(num_samples for _, num_samples in updates)
            total_samples = max(total_samples, 1)
            return [num_samples / total_samples for _, num_samples in updates]
        if self.weighting == "uniform":
            weight = 1.0 / len(updates)
            return [weight for _ in updates]
        raise ValueError(f"Unsupported aggregation weighting strategy: {self.weighting}")

    def _apply_weighted_average(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
    ) -> OrderedDict[str, torch.Tensor]:
        aggregated = OrderedDict((name, param.clone()) for name, param in global_state.items())
        if not updates:
            return aggregated

        weights = self._compute_weights(updates)
        for (state_dict, _), weight in zip(updates, weights):
            for name, tensor in state_dict.items():
                diff = tensor - global_state[name]
                aggregated[name] += (diff * weight).to(aggregated[name].dtype)
        return aggregated


@dataclass
class FedAvgAggregation(_WeightedAveragingMixin, AggregationMechanism):
    """Standard FedAvg weighted averaging aggregation."""

    weighting: str = "samples"
    name: str = field(init=False, default="fedavg")
    is_secure: bool = field(init=False, default=False)

    def aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        return self._apply_weighted_average(global_state, updates)


@dataclass
class FedProxAggregation(FedAvgAggregation):
    """FedProx uses the same aggregation rule as FedAvg."""

    name: str = field(init=False, default="fedprox")


@dataclass
class SecAggAggregation(FedAvgAggregation):
    """Simulated Bonawitz et al. secure aggregation."""

    mask_seed: int | None = None
    threshold: int = 2
    dropout_tolerance: float = 0.0
    retransmissions: int = 0
    key_refresh_interval: int | None = None
    name: str = field(init=False, default="secagg")
    is_secure: bool = field(init=False, default=True)

    def aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        LOGGER.debug(
            "SecAgg round %d/%s with seed=%s, threshold=%d, dropout_tol=%.3f, retransmissions=%d, key_refresh=%s",  # noqa: E501
            round_index + 1,
            "?" if total_rounds is None else total_rounds,
            self.mask_seed,
            self.threshold,
            self.dropout_tolerance,
            self.retransmissions,
            self.key_refresh_interval,
        )
        return super().aggregate(global_state, updates, round_index=round_index, total_rounds=total_rounds)  # type: ignore[misc]


@dataclass
class AHSecAggAggregation(FedAvgAggregation):
    """Adaptive hierarchical secure aggregation."""

    cluster_size: int = 2
    levels: int = 1
    level_thresholds: Sequence[float] | None = None
    dropout_rate: float = 0.0
    mask_reuse: str = "never"
    name: str = field(init=False, default="ahsecagg")
    is_secure: bool = field(init=False, default=True)

    def aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        LOGGER.debug(
            "AHSecAgg round %d/%s with clusters=%d, levels=%d, thresholds=%s, dropout=%.3f, reuse=%s",  # noqa: E501
            round_index + 1,
            "?" if total_rounds is None else total_rounds,
            self.cluster_size,
            self.levels,
            list(self.level_thresholds) if self.level_thresholds is not None else None,
            self.dropout_rate,
            self.mask_reuse,
        )
        return super().aggregate(global_state, updates, round_index=round_index, total_rounds=total_rounds)  # type: ignore[misc]


@dataclass
class PairwiseMaskingSecAggAggregation(FedAvgAggregation):
    """Pairwise masking (FastSecAgg) aggregation approximation."""

    group_count: int = 1
    key_agreement: str = "static"
    mask_update_frequency: int = 1
    timeout: float = 30.0
    encryption: str = "paillier"
    name: str = field(init=False, default="fastsecagg")
    is_secure: bool = field(init=False, default=True)

    def aggregate(
        self,
        global_state: OrderedDict[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> OrderedDict[str, torch.Tensor]:
        LOGGER.debug(
            "Pairwise masking round %d/%s with groups=%d, key_agreement=%s, mask_update=%d, timeout=%.1f, encryption=%s",  # noqa: E501
            round_index + 1,
            "?" if total_rounds is None else total_rounds,
            self.group_count,
            self.key_agreement,
            self.mask_update_frequency,
            self.timeout,
            self.encryption,
        )
        return super().aggregate(global_state, updates, round_index=round_index, total_rounds=total_rounds)  # type: ignore[misc]


def create_aggregation_mechanism(config: AggregationConfig) -> AggregationMechanism:
    """Instantiate the aggregation mechanism specified by ``config``."""

    mechanism = config.mechanism.lower()
    params = dict(config.parameters)

    if mechanism == "fedavg":
        return FedAvgAggregation(weighting=str(params.get("weighting", "samples")))
    if mechanism == "fedprox":
        return FedProxAggregation(weighting=str(params.get("weighting", "samples")))
    if mechanism == "secagg":
        return SecAggAggregation(
            weighting=str(params.get("weighting", "samples")),
            mask_seed=params.get("mask_seed"),
            threshold=int(params.get("threshold", 2)),
            dropout_tolerance=float(params.get("dropout_tolerance", 0.0)),
            retransmissions=int(params.get("retransmissions", 0)),
            key_refresh_interval=(
                None if params.get("key_refresh_interval") is None else int(params["key_refresh_interval"])
            ),
        )
    if mechanism == "ahsecagg":
        thresholds = params.get("level_thresholds")
        if thresholds is not None and not isinstance(thresholds, Sequence):
            raise TypeError("level_thresholds must be a sequence if provided")
        return AHSecAggAggregation(
            weighting=str(params.get("weighting", "samples")),
            cluster_size=int(params.get("cluster_size", 2)),
            levels=int(params.get("levels", 1)),
            level_thresholds=thresholds,
            dropout_rate=float(params.get("dropout_rate", 0.0)),
            mask_reuse=str(params.get("mask_reuse", "never")),
        )
    if mechanism in {"fastsecagg", "pairwise", "pairwise_masking"}:
        return PairwiseMaskingSecAggAggregation(
            weighting=str(params.get("weighting", "samples")),
            group_count=int(params.get("group_count", 1)),
            key_agreement=str(params.get("key_agreement", "static")),
            mask_update_frequency=int(params.get("mask_update_frequency", 1)),
            timeout=float(params.get("timeout", 30.0)),
            encryption=str(params.get("encryption", "paillier")),
        )
    raise ValueError(f"Unsupported aggregation mechanism: {config.mechanism}")


__all__ = [
    "AggregationConfig",
    "AggregationMechanism",
    "FedAvgAggregation",
    "FedProxAggregation",
    "SecAggAggregation",
    "AHSecAggAggregation",
    "PairwiseMaskingSecAggAggregation",
    "create_aggregation_mechanism",
]