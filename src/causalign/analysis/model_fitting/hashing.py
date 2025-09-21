"""Hashing utilities for reproducible specification and data fingerprints.

Centralizes all hashing so that changes to canonicalization rules are applied
consistently across spec, group, data, and figure hashes.

Design decisions:
* Use SHA256 → hex digest, truncate to 12 chars for filenames while retaining
  full hash in JSON metadata.
* Strict whitelist of fields included in a spec hash (exclude runtime metrics).
* Provide helper for stable JSON canonicalization (sorted keys, no whitespace).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


def _canonicalize(obj: Any) -> Any:
    """Produce a JSON-serializable object with deterministic ordering."""
    if isinstance(obj, Mapping):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v) for v in obj]
    return obj


def to_canonical_json(data: Any) -> str:
    """Return canonical JSON string with sorted keys & no extraneous whitespace."""
    canon = _canonicalize(data)
    return json.dumps(canon, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def sha256_digest(data: Any) -> str:
    """Full SHA256 hex digest of canonical JSON representation of data."""
    payload = to_canonical_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def short_hash(full_hash: str, length: int = 12) -> str:
    """Return stable shortened hash for filenames (default 12 chars)."""
    if length <= 0:
        raise ValueError("length must be positive")
    return full_hash[:length]


def compute_spec_hash(spec_dict: Mapping[str, Any]) -> tuple[str, str]:
    """Compute full & short hash for a model specification."""
    full = sha256_digest(spec_dict)
    return full, short_hash(full)


def compute_group_hash(agent: str, domains: Sequence[str] | None, prompt_category: str | None,
                       temperature: float | None, prompt_content_type: str | None) -> tuple[str, str]:
    """Group identity hash (agent + domains + prompt category + temperature + content type)."""
    doms = sorted([d for d in (domains or [])])
    key = {
        "agent": agent,
        "domains": doms,
        "prompt_category": prompt_category or None,
        "temperature": temperature if temperature is not None else None,
        "prompt_content_type": prompt_content_type or None,
    }
    full = sha256_digest(key)
    return full, short_hash(full)


def compute_data_hash(rows_signature: Sequence[tuple]) -> tuple[str, str]:
    """Hash a compact rows signature list of tuples (task, response, ...)."""
    full = sha256_digest(list(rows_signature))
    return full, short_hash(full)


def compute_figure_spec_hash(meta: Mapping[str, Any]) -> tuple[str, str]:
    """Hash figure metadata (exclude runtime where possible)."""
    full = sha256_digest(meta)
    return full, short_hash(full)


__all__ = [
    "to_canonical_json",
    "sha256_digest",
    "short_hash",
    "compute_spec_hash",
    "compute_group_hash",
    "compute_data_hash",
    "compute_figure_spec_hash",
]
