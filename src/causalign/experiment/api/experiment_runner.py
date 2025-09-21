"""Backward compatibility shim.

Provides an import location `causalign.experiment.api.experiment_runner.ExperimentRunner`
expected by older code and tests, re-exporting the new implementation from
`api_runner`.
"""

from .api_runner import ExperimentRunner  # noqa: F401

__all__ = ["ExperimentRunner"]
