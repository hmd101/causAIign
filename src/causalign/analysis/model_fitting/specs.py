"""Specification dataclasses for the refactored CBN fitting pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelSpec:
    link: str  # 'logistic' | 'noisy_or'
    params_tying: int  # 3,4,5
    param_bounds: Optional[Dict[str, Dict[str, float]]] = None  # future extension
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizerSpec:
    name: str  # 'lbfgs' | 'adam'
    lr: float
    epochs: int
    restarts: int
    weight_decay: Optional[float] = None
    tolerance_grad: Optional[float] = None
    tolerance_change: Optional[float] = None
    max_eval: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LossSpec:
    name: str  # 'mse' | 'huber' | 'mae'
    params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CVSpec:
    enabled: bool = False
    method: str = "kfold"  # or 'loocv'
    k: Optional[int] = None
    repeats: int = 1
    stratify_on: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RankingPolicy:
    primary_metric: str = "aic"
    fallbacks: List[str] = field(default_factory=lambda: ["rmse", "loss"])
    tie_breakers: List[str] = field(default_factory=lambda: ["params_tying", "loss", "seed_used"])
    # How to select representative restart parameters for metrics & persistence:
    # 'best_loss' (lowest loss_final), 'median_loss' (restart whose loss_final is median),
    # 'best_primary_metric' (restart minimizing primary metric if available in per-restart metrics; falls back to loss).
    selection_rule: str = "best_loss"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataFilterSpec:
    agents: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    prompt_categories: Optional[List[str]] = None
    reasoning_types: Optional[List[str]] = None
    tasks: Optional[List[str]] = None
    temperature: Optional[float] = None
    prompt_content_type: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunSpec:
    seed: int
    device: str = "auto"
    enable_loocv: bool = False
    schema_version: str = "1.2.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FitSpec:
    model: ModelSpec
    optimizer: OptimizerSpec
    loss: LossSpec
    ranking: RankingPolicy
    run: RunSpec
    cv: Optional[CVSpec] = None
    data_filters: Optional[DataFilterSpec] = None

    def to_minimal_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "loss": self.loss.to_dict(),
            "ranking": self.ranking.to_dict(),
            "run": self.run.to_dict(),
            "cv": self.cv.to_dict() if self.cv else None,
            "data_filters": self.data_filters.to_dict() if self.data_filters else None,
        }


__all__ = [
    "ModelSpec",
    "OptimizerSpec",
    "LossSpec",
    "CVSpec",
    "RankingPolicy",
    "DataFilterSpec",
    "RunSpec",
    "FitSpec",
]
