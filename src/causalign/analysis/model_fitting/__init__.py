"""Model fitting package for causal Bayes net (collider graph) using PyTorch.

Modules:
- data: loading processed data with the same interface as plotting, plus filters
- tasks: task probability evaluators for Logistic and Noisy-OR links
- models: parameterizations and tying for collider models
- losses: loss function registry
- trainer: training loop per agent/domain with device support (MPS/CUDA/CPU)
- io: utilities to persist results to JSON/CSV
- cli: command-line interface for running model fits

This package is designed to be easily extended to other graph structures later.
"""

from . import data, tasks, models, losses, trainer, io, cli  # noqa: F401

 