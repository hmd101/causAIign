from __future__ import annotations

"""
Loss registry for model fitting.

Targets are expected in [0,1]. Provide basic options (MSE, Huber).
"""

from typing import Callable, Dict

import torch
from torch import nn


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def huber_loss(pred: torch.Tensor, target: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """
    Huber loss (smooth L1 loss) - robust to outliers.
    
    For prediction errors |pred - target| <= delta, uses squared loss.
    For errors > delta, uses linear loss to reduce influence of outliers.
    
    Args:
        pred: Predicted values
        target: Target values  
        delta: Threshold for switching from quadratic to linear loss
    
    Returns:
        Huber loss value
    """
    diff = pred - target
    abs_diff = torch.abs(diff)
    
    # Quadratic for small errors, linear for large errors
    quadratic_part = 0.5 * diff ** 2
    linear_part = delta * (abs_diff - 0.5 * delta)
    
    # Use quadratic when |error| <= delta, linear otherwise
    loss = torch.where(abs_diff <= delta, quadratic_part, linear_part)
    
    return torch.mean(loss)


LOSS_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "mse": mse_loss,
    "huber": huber_loss,
}


