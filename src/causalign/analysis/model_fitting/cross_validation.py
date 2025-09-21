"""
Leave-One-Out Cross-Validation for Causal Bayes Net Models

This module implements task-level LOOCV for evaluating CBN generalization performance.
For each of the 11 tasks, we train on 10 tasks and predict the held-out task.
"""

import logging
import math
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from .losses import LOSS_REGISTRY
from .models import create_parameter_module, device_from_string
from .tasks import roman_task_to_probability
from .trainer import FitConfig, _run_adam, _run_lbfgs, _set_seeds

logger = logging.getLogger(__name__)


def perform_loocv_single_group(
    df_group: pd.DataFrame,
    config: FitConfig,
) -> Dict:
    """
    Perform leave-one-out cross-validation on tasks for a single group.
    
    For each of the 11 tasks:
    1. Train CBN on remaining 10 tasks
    2. Predict the held-out task
    3. Compare prediction to actual agent response
    
    Returns LOOCV metrics and fold-wise results.
    """
    logger.info(f"Starting LOOCV with {len(df_group)} observations")
    
    # Setup
    device = device_from_string(config.device)
    _set_seeds(config.seed)
    # TODO(seed-policy): Base seeding for LOOCV uses the global seed; per-fold restarts
    # derive seeds from (seed + r + hash(holdout_task) % 1000). Consider adding an
    # option to align LOOCV initialization with the winning full-data restart seed
    # (e.g., FitConfig flag) and persist fold-wise seeds in results for provenance.
    
    # Extract all tasks and responses
    tasks = list(df_group["task"].astype(str).values)
    # Responses are accessed via test/train splits below; no need to pre-extract here.
    
    unique_tasks = sorted(list(set(tasks)))
    n_tasks = len(unique_tasks)
    
    if n_tasks < 2:
        raise ValueError(f"Need at least 2 unique tasks for LOOCV, got {n_tasks}")
    
    logger.info(f"Performing LOOCV on {n_tasks} unique tasks: {unique_tasks}")
    
    # Storage for fold results
    fold_results = []
    holdout_predictions = []
    holdout_actuals = []
    
    loss_fn = LOSS_REGISTRY[config.loss_name]
    
    # Leave-one-out loop
    for holdout_task in unique_tasks:
        logger.debug(f"LOOCV fold: holding out task {holdout_task}")
        
        # Split data
        train_mask = df_group["task"] != holdout_task
        test_mask = df_group["task"] == holdout_task
        
        train_df = df_group[train_mask].copy()
        test_df = df_group[test_mask].copy()
        
        if len(train_df) == 0 or len(test_df) == 0:
            logger.warning(f"Skipping task {holdout_task}: insufficient data")
            continue
        
        # Train on fold
        try:
            fold_result = _fit_single_fold(
                train_df, holdout_task, config, device, loss_fn
            )
            
            # Predict held-out task
            holdout_pred = _predict_holdout_task(
                holdout_task, fold_result["best_params"], config.link
            )
            
            # Get actual response for held-out task (average if multiple)
            holdout_actual = test_df["response"].mean()
            
            # Store results
            fold_info = {
                "holdout_task": holdout_task,
                "train_tasks": sorted(train_df["task"].unique()),
                "train_loss": fold_result["train_loss"],
                "holdout_pred": float(holdout_pred),
                "holdout_actual": float(holdout_actual),
                "holdout_error": float(abs(holdout_pred - holdout_actual)),
                "holdout_squared_error": float((holdout_pred - holdout_actual) ** 2),
                "fitted_params": fold_result["best_params"],
                "n_train_obs": len(train_df),
                "n_test_obs": len(test_df),
            }
            
            fold_results.append(fold_info)
            holdout_predictions.append(float(holdout_pred))
            holdout_actuals.append(float(holdout_actual))
            
        except Exception as e:
            logger.warning(f"Failed to fit fold for task {holdout_task}: {e}")
            continue
    
    if not fold_results:
        raise ValueError("No successful LOOCV folds completed")
    
    # Compute LOOCV metrics
    loocv_metrics = _compute_loocv_metrics(
        holdout_predictions, holdout_actuals, fold_results
    )
    
    result = {
        "loocv_metrics": loocv_metrics,
        "fold_results": fold_results,
        "n_successful_folds": len(fold_results),
        "n_total_folds": n_tasks,
    }
    
    logger.info(f"LOOCV completed: {len(fold_results)}/{n_tasks} successful folds")
    logger.info(f"LOOCV-RMSE: {loocv_metrics['loocv_rmse']:.4f}")
    
    return result


def _fit_single_fold(
    train_df: pd.DataFrame,
    holdout_task: str,
    config: FitConfig,
    device: torch.device,
    loss_fn,
) -> Dict:
    """Fit CBN parameters on training fold"""
    
    # Extract training data
    train_tasks = list(train_df["task"].astype(str).values)
    train_targets = torch.tensor(
        train_df["response"].astype(float).values, 
        dtype=torch.float32, 
        device=device
    )
    
    best_loss = math.inf
    best_params = None
    
    # Multi-restart optimization (fewer restarts for efficiency)
    n_restarts = max(1, config.restarts // 2)  # Use fewer restarts for LOOCV
    
    for r in range(n_restarts):
        # TODO(seed-policy): Optionally reuse the winner's best restart seed (or a fixed
        # schedule) for LOOCV when a config flag is enabled. Also record the exact
        # per-fold seed in the returned fold result for traceability. See
        # scripts/README_export_cbn_best_fits.md (TODO section).
        _set_seeds(config.seed + r + hash(holdout_task) % 1000)
        
        # Create and initialize model
        module = create_parameter_module(config.link, config.num_params)
        module.to(device)
        
        for p in module.parameters():
            torch.nn.init.normal_(p, mean=0.0, std=0.1)
        
        # Train
        if config.optimizer == "lbfgs":
            loss_val, _ = _run_lbfgs(
                module, config.link, train_tasks, train_targets, loss_fn, 
                max_iter=config.epochs // 2  # Fewer iterations for efficiency
            )
        else:
            loss_val, _ = _run_adam(
                module, config.link, train_tasks, train_targets, loss_fn,
                lr=config.lr, epochs=config.epochs // 2
            )
        
        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {k: v.detach().cpu().item() for k, v in module.get_params().items()}
    
    return {
        "train_loss": best_loss,
        "best_params": best_params,
    }


def _predict_holdout_task(
    holdout_task: str,
    fitted_params: Dict[str, float],
    link: str,
) -> float:
    """Generate prediction for held-out task using fitted parameters"""
    
    # Convert to tensors
    param_tensors = {}
    for key, value in fitted_params.items():
        param_tensors[key] = torch.tensor(float(value), dtype=torch.float32)
    
    # Generate prediction
    pred_tensor = roman_task_to_probability(holdout_task, link, param_tensors)
    return float(pred_tensor.item())


def _compute_loocv_metrics(
    predictions: List[float],
    actuals: List[float],
    fold_results: List[Dict],
) -> Dict[str, float]:
    """Compute comprehensive LOOCV metrics"""
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Basic error metrics
    errors = predictions - actuals
    squared_errors = errors ** 2
    absolute_errors = np.abs(errors)
    
    loocv_rmse = float(np.sqrt(np.mean(squared_errors)))
    loocv_mae = float(np.mean(absolute_errors))
    
    # R-squared for held-out predictions
    ss_res = np.sum(squared_errors)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    loocv_r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else float("nan")
    
    # Consistency across folds (lower = more consistent)
    fold_errors = [fold["holdout_error"] for fold in fold_results]
    loocv_consistency = float(np.std(fold_errors))
    
    # Calibration (simplified)
    loocv_calibration = _compute_calibration_error(predictions, actuals)
    
    # Bias (systematic over/under-prediction)
    loocv_bias = float(np.mean(errors))
    
    return {
        "loocv_rmse": loocv_rmse,
        "loocv_mae": loocv_mae,
        "loocv_r2": loocv_r2,
        "loocv_consistency": loocv_consistency,
        "loocv_calibration": loocv_calibration,
        "loocv_bias": loocv_bias,
        "loocv_max_error": float(np.max(absolute_errors)),
        "loocv_min_error": float(np.min(absolute_errors)),
    }


def _compute_calibration_error(predictions: np.ndarray, actuals: np.ndarray) -> float:
    """Compute calibration error for LOOCV predictions"""
    # Simple binned calibration error
    n_bins = min(5, len(predictions))  # Fewer bins for small sample
    
    if n_bins < 2:
        return float("nan")
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    calibration_error = 0.0
    
    for i in range(n_bins):
        bin_mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper bound in last bin
            bin_mask = (predictions >= bin_edges[i]) & (predictions <= bin_edges[i + 1])
        
        if np.any(bin_mask):
            bin_pred_mean = np.mean(predictions[bin_mask])
            bin_actual_mean = np.mean(actuals[bin_mask])
            bin_weight = np.sum(bin_mask) / len(predictions)
            calibration_error += bin_weight * abs(bin_pred_mean - bin_actual_mean)
    
    return float(calibration_error)
