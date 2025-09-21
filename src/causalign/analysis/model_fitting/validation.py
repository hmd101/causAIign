from __future__ import annotations

"""Pydantic models for validating structured fit result JSON artifacts.

These models mirror (a subset of) the dataclass based schema in `result_types`.
We keep them separate to avoid adding a hard runtime dependency from the core
training path onto Pydantic; validation is applied at IO boundaries only.

Versioning strategy:
- The top-level object carries `schema_version` (string semver).
- This module knows the CURRENT_SCHEMA_VERSION. If a file with a *newer*
    major version is encountered, we raise. If an older minor/patch is loaded,
    we attempt a *best-effort* compatibility parse (fields may be absent) and
    surface deprecation warnings via return metadata (future TODO).

Lightweight helper APIs:
- validate_group_fit_result_dict(data) -> validated model (raises on error)
- ensure_schema_version_compatible(version) -> None (raises on incompatible)

The goal is to fail fast on corrupt / incompatible JSON while keeping
production code free of heavy validation logic.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, validator

CURRENT_SCHEMA_VERSION = "1.2.0"

# ---------------------------- Helper utilities ----------------------------

def _split_semver(v: str) -> List[int]:
    parts = v.split(".")
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except ValueError:
            out.append(0)
    while len(out) < 3:
        out.append(0)
    return out[:3]


def ensure_schema_version_compatible(version: str) -> None:
    cur = _split_semver(CURRENT_SCHEMA_VERSION)
    other = _split_semver(version)
    # Enforce same major version
    if other[0] != cur[0]:
        raise ValueError(
            f"Incompatible schema_version major: file={version} expected~={CURRENT_SCHEMA_VERSION}."
            " Please upgrade/downgrade tooling or migrate artifact."
        )
    # If file minor is greater than current minor, we cannot guarantee forward compatibility
    if other[1] > cur[1]:
        raise ValueError(
            f"Artifact schema minor version {version} is newer than supported {CURRENT_SCHEMA_VERSION}. Upgrade code."
        )

# ---------------------------- Pydantic models -----------------------------

class RestartRecordModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    restart_index: int
    seed: int
    loss_final: float
    params: Dict[str, float]
    init_params: Dict[str, float]
    curve: Optional[List[float]] = None
    duration_sec: Optional[float] = None
    status: str = Field(default="success")
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    ece_10bin: Optional[float] = None


class GroupFitResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    spec_hash: str
    short_spec_hash: str
    group_hash: str
    short_group_hash: str
    spec: Dict[str, Any]
    group_key: Dict[str, Any]
    data_spec: Dict[str, Any]
    restarts: List[RestartRecordModel]
    best_restart_index: int
    best_params: Dict[str, float]
    metrics: Dict[str, Any]
    ranking: Dict[str, Any]
    uncertainty: Optional[Dict[str, Any]] = None
    environment: Optional[Dict[str, Any]] = None
    provenance: Optional[Dict[str, Any]] = None
    restart_summary: Optional[Dict[str, Any]] = None

    @validator("schema_version")
    def _check_version(cls, v: str):  # noqa: D401
        ensure_schema_version_compatible(v)
        return v

    @validator("restarts")
    def _non_empty_restarts(cls, v: List[RestartRecordModel]):  # noqa: D401
        if not v:
            raise ValueError("restarts must contain at least one restart record")
        return v


# ---------------------------- Public helpers ------------------------------

def validate_group_fit_result_dict(data: Dict[str, Any]) -> GroupFitResultModel:
    if not isinstance(data, dict):
        raise TypeError("Expected dict for group fit result JSON")
    return GroupFitResultModel(**data)


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "ensure_schema_version_compatible",
    "validate_group_fit_result_dict",
    "GroupFitResultModel",
]
