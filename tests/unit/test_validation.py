from __future__ import annotations

from pathlib import Path
import pytest

from causalign.analysis.model_fitting.validation import (
    validate_group_fit_result_dict,
    CURRENT_SCHEMA_VERSION,
)
from causalign.analysis.model_fitting.result_types import GroupFitResult


def test_validation_round_trip(tmp_path: Path):
    dummy = GroupFitResult(
        schema_version=CURRENT_SCHEMA_VERSION,
        spec_hash="a"*64,
        short_spec_hash="a"*12,
        group_hash="b"*64,
        short_group_hash="b"*12,
        spec={"model": {"link": "logistic"}},
        group_key={"agent": "agent1", "prompt_category": "pc", "domain": None},
        data_spec={"num_rows": 1, "data_hash": "c"*64},
        restarts=[{"restart_index":0, "seed":0, "loss_final":1.23, "params":{}, "init_params":{}}],
        best_restart_index=0,
        best_params={},
        metrics={"loss":1.23},
        ranking={"primary_metric":"aic", "fallbacks":[], "selection_rule":"best_loss"},
        uncertainty=None,
        environment=None,
        provenance=None,
        restart_summary=None,
    )
    data = dummy.to_dict()
    validated = validate_group_fit_result_dict(data)
    assert validated.best_restart_index == 0


def test_validation_rejects_empty_restarts():
    bad = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "spec_hash": "a"*64,
        "short_spec_hash": "a"*12,
        "group_hash": "b"*64,
        "short_group_hash": "b"*12,
        "spec": {"model": {"link": "logistic"}},
        "group_key": {"agent": "agent1"},
        "data_spec": {},
        "restarts": [],
        "best_restart_index": 0,
        "best_params": {},
        "metrics": {"loss": 1.0},
        "ranking": {"primary_metric": "aic", "fallbacks": [], "selection_rule": "best_loss"},
    }
    with pytest.raises(Exception):
        validate_group_fit_result_dict(bad)


def test_validation_rejects_newer_major_version():
    newer = {
        "schema_version": "2.0.0",  # major bump
        "spec_hash": "a"*64,
        "short_spec_hash": "a"*12,
        "group_hash": "b"*64,
        "short_group_hash": "b"*12,
        "spec": {"model": {"link": "logistic"}},
        "group_key": {"agent": "agent1"},
        "data_spec": {},
        "restarts": [{"restart_index":0, "seed":0, "loss_final":1.0, "params":{}, "init_params":{}}],
        "best_restart_index": 0,
        "best_params": {},
        "metrics": {"loss": 1.0},
        "ranking": {"primary_metric": "aic", "fallbacks": [], "selection_rule": "best_loss"},
    }
    with pytest.raises(Exception):
        validate_group_fit_result_dict(newer)
