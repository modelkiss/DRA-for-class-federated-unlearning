"""Differential privacy controllers for federated optimisation."""
from __future__ import annotations

import ast
import logging
import math
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

LOGGER = logging.getLogger(__name__)

StateDict = OrderedDict[str, torch.Tensor]
ClientUpdate = Tuple[StateDict, int]


def _clone_state(state: StateDict) -> StateDict:
    return OrderedDict((name, tensor.clone()) for name, tensor in state.items())


def _clip_update(update: StateDict, reference: StateDict, clip_norm: float | None) -> StateDict:
    if clip_norm is None or clip_norm <= 0:
        return _clone_state(update)

    squared_norm = 0.0
    differences: Dict[str, torch.Tensor] = {}
    for name, tensor in update.items():
        diff = (tensor - reference[name]).to(torch.float32)
        differences[name] = diff
        squared_norm += float(diff.pow(2).sum().item())

    norm = math.sqrt(squared_norm)
    if norm <= clip_norm:
        return _clone_state(update)

    scale = clip_norm / (norm + 1e-12)
    clipped = OrderedDict()
    for name, tensor in update.items():
        delta = differences[name] * scale
        clipped_tensor = reference[name] + delta.to(reference[name].dtype)
        clipped[name] = clipped_tensor.to(tensor.dtype)
    return clipped


def _apply_gaussian_noise(
    state: StateDict,
    *,
    std: float,
    generator: torch.Generator,
) -> StateDict:
    if std <= 0:
        return state

    for name, tensor in state.items():
        noise = torch.normal(
            mean=0.0,
            std=std,
            size=tensor.shape,
            device=tensor.device,
            generator=generator,
        )
        state[name] = (tensor + noise.to(tensor.dtype)).type_as(tensor)
    return state


def _safe_eval(expression: str, *, round_index: int, total_rounds: int | None) -> float:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.Call,
        ast.Attribute,
        ast.Mod,
    )

    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported token '{ast.dump(node)}' in epsilon expression")
        if isinstance(node, ast.Call) and not isinstance(node.func, (ast.Name, ast.Attribute)):
            raise ValueError("Only direct function calls are allowed in epsilon expression")

    local_env = {
        "round": float(round_index),
        "total_rounds": float(total_rounds if total_rounds is not None else -1.0),
        "math": math,
    }
    return float(eval(compile(tree, filename="<epsilon>", mode="eval"), {"__builtins__": {}}, local_env))


@dataclass
class DifferentialPrivacyConfig:
    """High-level configuration for DP-enabled training."""

    method: str = "none"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def enabled(self) -> bool:
        return self.method.lower() not in {"", "none", "disabled"}


class DifferentialPrivacyController:
    """Abstract controller for applying differential privacy during aggregation."""

    def __init__(self, *, seed: int | None = None) -> None:
        self._base_seed = seed if seed is not None else random.randrange(0, 2**31 - 1)
        self._generator = torch.Generator()
        self._generator.manual_seed(self._base_seed)

    @property
    def generator(self) -> torch.Generator:
        return self._generator

    def prepare_round(self, round_index: int, total_rounds: int | None) -> None:
        """Hook executed before each aggregation round."""

    def preprocess_updates(
        self,
        global_state: StateDict,
        updates: List[ClientUpdate],
    ) -> List[ClientUpdate]:
        return updates

    def postprocess_aggregate(
        self,
        global_state: StateDict,
        aggregated_state: StateDict,
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> StateDict:
        return aggregated_state


class NoOpDifferentialPrivacyController(DifferentialPrivacyController):
    def preprocess_updates(self, global_state: StateDict, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        return updates


class DPSGDDifferentialPrivacyController(DifferentialPrivacyController):
    def __init__(
        self,
        *,
        epsilon: float,
        delta: float,
        noise_multiplier: float,
        clip: float,
        sampling_probability: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = max(0.0, noise_multiplier)
        self.clip = clip
        self.sampling_probability = max(0.0, min(1.0, sampling_probability))

    def preprocess_updates(self, global_state: StateDict, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        processed: List[ClientUpdate] = []
        for state_dict, samples in updates:
            clipped = _clip_update(state_dict, global_state, self.clip)
            effective_samples = max(1, int(samples * self.sampling_probability)) if self.sampling_probability > 0 else samples
            processed.append((clipped, effective_samples))
        return processed

    def postprocess_aggregate(
        self,
        global_state: StateDict,
        aggregated_state: StateDict,
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> StateDict:
        std = self.noise_multiplier * max(self.clip, 1e-12)
        LOGGER.debug(
            "DP-SGD round %d: ε=%.4f, δ=%.1e, σ=%.4f, clip=%.4f, q=%.4f",
            round_index,
            self.epsilon,
            self.delta,
            self.noise_multiplier,
            self.clip,
            self.sampling_probability,
        )
        return _apply_gaussian_noise(aggregated_state, std=std, generator=self.generator)


class LocalDPFLController(DifferentialPrivacyController):
    def __init__(
        self,
        *,
        noise_multiplier: float,
        clipping_norm: float,
        seed_consistency: bool,
        local_sampling_rate: float,
        epsilon: float,
        delta: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.noise_multiplier = max(0.0, noise_multiplier)
        self.clipping_norm = clipping_norm
        self.seed_consistency = seed_consistency
        self.local_sampling_rate = max(0.0, min(1.0, local_sampling_rate))
        self.epsilon = epsilon
        self.delta = delta

    def prepare_round(self, round_index: int, total_rounds: int | None) -> None:
        if not self.seed_consistency:
            self.generator.manual_seed(self._base_seed + round_index + 1)

    def preprocess_updates(self, global_state: StateDict, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        processed: List[ClientUpdate] = []
        std = self.noise_multiplier * max(self.clipping_norm, 1e-12)
        for state_dict, samples in updates:
            clipped = _clip_update(state_dict, global_state, self.clipping_norm)
            noisy = _clone_state(clipped)
            noisy = _apply_gaussian_noise(noisy, std=std, generator=self.generator)
            effective_samples = max(1, int(samples * self.local_sampling_rate)) if self.local_sampling_rate > 0 else samples
            processed.append((noisy, effective_samples))
        LOGGER.debug(
            "LDP-FL round processed with σ=%.4f, clip=%.4f, local q=%.4f, ε=%.4f, δ=%.1e",
            self.noise_multiplier,
            self.clipping_norm,
            self.local_sampling_rate,
            self.epsilon,
            self.delta,
        )
        return processed


class AdaptiveDPFLController(DifferentialPrivacyController):
    def __init__(
        self,
        *,
        epsilon_expression: str,
        adaptive_clipping: str,
        noise_growth_rate: float,
        max_rounds: int,
        budget_decay: float,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.epsilon_expression = epsilon_expression
        self.adaptive_clipping = adaptive_clipping.lower()
        self.noise_growth_rate = max(0.0, noise_growth_rate)
        self.max_rounds = max(1, max_rounds)
        self.budget_decay = max(0.0, budget_decay)
        self.base_clip = 1.0
        self.current_clip = self.base_clip

    def _compute_clip(self, round_index: int, total_rounds: int | None) -> float:
        progress = min(round_index / max(self.max_rounds, 1), 1.0)
        if self.adaptive_clipping == "linear":
            return max(self.base_clip * (1.0 - progress * self.budget_decay), 1e-6)
        if self.adaptive_clipping == "exponential":
            return self.base_clip * math.exp(-self.budget_decay * round_index)
        if self.adaptive_clipping == "cosine":
            return max(self.base_clip * 0.5 * (1.0 + math.cos(math.pi * progress)), 1e-6)
        return self.base_clip

    def prepare_round(self, round_index: int, total_rounds: int | None) -> None:
        if round_index >= self.max_rounds:
            LOGGER.warning(
                "Adaptive DP-FL maximum rounds (%d) exceeded; disabling additional noise",
                self.max_rounds,
            )
            self.current_clip = 0.0
            return

        if self.epsilon_expression:
            try:
                epsilon_value = _safe_eval(
                    self.epsilon_expression,
                    round_index=round_index,
                    total_rounds=total_rounds,
                )
                LOGGER.debug("Adaptive DP-FL epsilon(round=%d)=%.4f", round_index, epsilon_value)
            except Exception as exc:  # pragma: no cover - defensive programming
                LOGGER.warning("Failed to evaluate adaptive epsilon expression '%s': %s", self.epsilon_expression, exc)
        self.current_clip = max(self._compute_clip(round_index, total_rounds), 0.0)

    def preprocess_updates(self, global_state: StateDict, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        processed: List[ClientUpdate] = []
        for state_dict, samples in updates:
            clipped = _clip_update(state_dict, global_state, self.current_clip if self.current_clip > 0 else None)
            processed.append((clipped, samples))
        return processed

    def postprocess_aggregate(
        self,
        global_state: StateDict,
        aggregated_state: StateDict,
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> StateDict:
        if self.current_clip <= 0:
            return aggregated_state
        std = self.noise_growth_rate * (round_index + 1) * max(self.current_clip, 1e-12)
        decay = max(0.0, 1.0 - min(round_index, self.max_rounds) / self.max_rounds * self.budget_decay)
        std *= decay
        return _apply_gaussian_noise(aggregated_state, std=std, generator=self.generator)


class RDPFLController(DifferentialPrivacyController):
    def __init__(
        self,
        *,
        renyi_order: float,
        epsilon_accumulation: str,
        noise_multiplier: float,
        clip: float,
        accounting_window: int,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.renyi_order = max(1.01, renyi_order)
        self.epsilon_accumulation = epsilon_accumulation.lower()
        self.noise_multiplier = max(0.0, noise_multiplier)
        self.clip = clip
        self.accounting_window = max(1, accounting_window)
        self._epsilon_history: List[float] = []
        self.cumulative_epsilon = 0.0

    def preprocess_updates(self, global_state: StateDict, updates: List[ClientUpdate]) -> List[ClientUpdate]:
        processed: List[ClientUpdate] = []
        for state_dict, samples in updates:
            clipped = _clip_update(state_dict, global_state, self.clip)
            processed.append((clipped, samples))
        return processed

    def postprocess_aggregate(
        self,
        global_state: StateDict,
        aggregated_state: StateDict,
        *,
        round_index: int,
        total_rounds: int | None,
    ) -> StateDict:
        std = self.noise_multiplier * max(self.clip, 1e-12)
        _apply_gaussian_noise(aggregated_state, std=std, generator=self.generator)

        epsilon_increment = (self.noise_multiplier**2) * math.sqrt(max(round_index + 1, 1)) / max(self.renyi_order - 1.0, 1e-6)
        self._epsilon_history.append(epsilon_increment)
        if len(self._epsilon_history) > self.accounting_window:
            self._epsilon_history.pop(0)

        if self.epsilon_accumulation == "max":
            self.cumulative_epsilon = max(self._epsilon_history)
        elif self.epsilon_accumulation == "window":
            self.cumulative_epsilon = sum(self._epsilon_history)
        else:
            self.cumulative_epsilon += epsilon_increment

        LOGGER.debug(
            "RDP-FL round %d: α=%.2f, ε_increment=%.4f, cumulative_ε=%.4f",
            round_index,
            self.renyi_order,
            epsilon_increment,
            self.cumulative_epsilon,
        )
        return aggregated_state


def create_dp_controller(
    config: DifferentialPrivacyConfig | None,
    *,
    seed: int | None = None,
) -> DifferentialPrivacyController | None:
    if config is None or not config.enabled():
        return None

    method = config.method.lower()
    params = config.parameters

    if method == "dp-sgd":
        return DPSGDDifferentialPrivacyController(
            epsilon=float(params.get("epsilon", 1.0)),
            delta=float(params.get("delta", 1e-5)),
            noise_multiplier=float(params.get("noise_multiplier", params.get("sigma", 1.0))),
            clip=float(params.get("clip", params.get("threshold", 1.0))),
            sampling_probability=float(params.get("sampling_probability", params.get("q", 1.0))),
            seed=seed,
        )
    if method == "ldp-fl":
        return LocalDPFLController(
            noise_multiplier=float(params.get("per_client_noise_multiplier", params.get("noise_multiplier", 1.0))),
            clipping_norm=float(params.get("clipping_norm", params.get("clip", 1.0))),
            seed_consistency=bool(params.get("seed_consistency", True)),
            local_sampling_rate=float(params.get("local_sampling_rate", 1.0)),
            epsilon=float(params.get("epsilon", 1.0)),
            delta=float(params.get("delta", 1e-5)),
            seed=seed,
        )
    if method == "adaptive-dp-fl":
        return AdaptiveDPFLController(
            epsilon_expression=str(params.get("epsilon_expression", "1.0")),
            adaptive_clipping=str(params.get("adaptive_clipping", "linear")),
            noise_growth_rate=float(params.get("noise_growth_rate", 0.5)),
            max_rounds=int(params.get("max_rounds", 100)),
            budget_decay=float(params.get("budget_decay", 0.1)),
            seed=seed,
        )
    if method == "rdp-fl":
        return RDPFLController(
            renyi_order=float(params.get("renyi_order", params.get("alpha", 2.0))),
            epsilon_accumulation=str(params.get("epsilon_accumulation", "additive")),
            noise_multiplier=float(params.get("noise_multiplier", 1.0)),
            clip=float(params.get("clip", params.get("threshold", 1.0))),
            accounting_window=int(params.get("accounting_window", params.get("window", 10))),
            seed=seed,
        )

    raise ValueError(f"Unsupported differential privacy method: {config.method}")


__all__ = [
    "AdaptiveDPFLController",
    "ClientUpdate",
    "DPSGDDifferentialPrivacyController",
    "DifferentialPrivacyConfig",
    "DifferentialPrivacyController",
    "LocalDPFLController",
    "NoOpDifferentialPrivacyController",
    "RDPFLController",
    "StateDict",
    "create_dp_controller",
]
