"""Result dataclasses for new fiting version (1.2)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

# Schema version bumped:
# 1.1.0: added ranking.selection_rule enforcement & per-restart metric propagation
# 1.2.0: removed unused per-restart CV placeholders (cv_* in RestartRecord) clarifying CV is group-level only
SCHEMA_VERSION = "1.2.0"


@dataclass
class RestartRecord:
    restart_index: int
    seed: int
    loss_final: float
    params: Dict[str, float]
    init_params: Dict[str, float]
    curve: Optional[List[float]] = None
    duration_sec: Optional[float] = None
    status: str = "success"  # success | failed | nan_loss | diverged
    # Per-restart training metrics (optional; populated when restart metrics collection enabled)
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    ece_10bin: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MetricsBlock:
    loss: float
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    ece_10bin: Optional[float] = None
    cv_rmse: Optional[float] = None
    cv_r2: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UncertaintyBlock:
    method: str
    se: Dict[str, float]
    warnings: Optional[List[str]] = None
    condition_number: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupFitResult:
    schema_version: str
    spec_hash: str
    short_spec_hash: str
    group_hash: str
    short_group_hash: str
    spec: Dict[str, Any]
    group_key: Dict[str, Any]
    data_spec: Dict[str, Any]
    restarts: List[Dict[str, Any]]
    best_restart_index: int
    best_params: Dict[str, float]
    metrics: Dict[str, Any]
    ranking: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    restart_summary: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "RestartRecord",
    "MetricsBlock",
    "UncertaintyBlock",
    "GroupFitResult",
]
