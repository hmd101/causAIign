from __future__ import annotations

"""Trainer for collider models with multi-restart optimization.

Implements per-agent (and optional domain) parameter fitting using PyTorch with
support for Adam or LBFGS, capturing all restart outcomes for richer analysis.
"""

from typing import Dict, List, Sequence, Tuple

import logging
import math
import random
import copy
import time

import torch
from torch import nn

from .models import create_parameter_module, device_from_string
from .tasks import roman_task_to_probability
from .losses import LOSS_REGISTRY


logger = logging.getLogger(__name__)


class FitConfig:
    """Configuration for fitting collider models to behavioral data.
    
    Specifies model architecture (link function, parameter tying), optimization
    settings (optimizer, learning rate, epochs, restarts), and computational
    resources (device, random seed).
   
    """
    
    def __init__(
            self,
            link: str = "logistic",  # or "noisy_or"
            num_params: int = 3,  # 3,4,5
            loss_name: str = "mse",
            optimizer: str = "lbfgs",  # or "adam"
            lr: float = 0.1,
            epochs: int = 300,
            restarts: int = 5,
            seed: int = 0,
            device: str = "auto",  # "auto", "cpu", "cuda", "mps"
            enable_loocv: bool = False,  # Enable leave-one-out cross-validation
            compute_uncertainty: bool = False,  # Approximate parameter uncertainty via Hessian
    ):
        self.link = link
        self.num_params = num_params
        self.loss_name = loss_name
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.restarts = restarts
        self.seed = seed
        self.device = device
        self.enable_loocv = enable_loocv
        self.compute_uncertainty = compute_uncertainty


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _objective_for_batch(
    module: nn.Module,
    link: str,
    tasks: Sequence[str],
    targets: torch.Tensor,
    loss_fn,
) -> torch.Tensor:
    params = module.get_params()
    preds: List[torch.Tensor] = []
    for roman in tasks:
        preds.append(roman_task_to_probability(roman, link, params))
    pred_vec = torch.stack(preds, dim=0)
    return loss_fn(pred_vec, targets)


def _run_lbfgs(
    module: nn.Module,
    link: str,
    tasks: Sequence[str],
    targets: torch.Tensor,
    loss_fn,
    max_iter: int,

) -> Tuple[float, List[float]]:
    optimizer = torch.optim.LBFGS(module.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    curve: List[float] = []

    def closure():
        """LBFGS closure function - required by PyTorch's LBFGS optimizer.
        
        The closure is a function that:
        1. Clears gradients from previous iterations
        2. Computes the forward pass (loss calculation)
        3. Computes gradients via backpropagation
        4. Returns the loss value
        
        LBFGS is a quasi-Newton optimization method that may need to evaluate
        the function and gradients multiple times per optimization step to
        perform line searches and build its approximation of the Hessian matrix.
        The closure allows LBFGS to re-evaluate the loss and gradients as needed
        during its internal optimization procedures.
        
        This is different from first-order optimizers like Adam/SGD that only
        need one forward/backward pass per step.
        """
        optimizer.zero_grad(set_to_none=True)
        loss = _objective_for_batch(module, link, tasks, targets, loss_fn)
        loss.backward()
        # Record each closure evaluation
        curve.append(float(loss.detach().cpu().item()))
        return loss

    final_loss = optimizer.step(closure)
    return float(final_loss.detach().cpu().item()), curve


def _run_adam(
    module: nn.Module,
    link: str,
    tasks: Sequence[str],
    targets: torch.Tensor,
    loss_fn,
    lr: float,
    epochs: int,
) -> Tuple[float, List[float]]:
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    best = math.inf
    curve: List[float] = []
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = _objective_for_batch(module, link, tasks, targets, loss_fn)
        loss.backward()
        optimizer.step()
        current = float(loss.detach().cpu().item())
        curve.append(current)
        best = min(best, current)
    return best, curve


def fit_single_group(
    df_group,
    config: FitConfig,
    collect_restart_metrics: bool = False,
) -> Dict:
    """Fit parameters for one subset (e.g., one agent or one (agent, domain)).

    This is the main model fitting function that implements a complete training loop
    with multiple random restarts to find optimal parameters for a causal Bayes net model.
    
    The method follows this learning process:
    1. Setup: Initialize device, seeds, and extract training data
    2. Multi-restart optimization loop: Try multiple random initializations
    3. For each restart: Initialize parameters → Run optimizer → Track best result
    4. Evaluation: Compute metrics on best fitted model
    5. Return: Package results with parameters, metrics, and training curves

    Expects df_group to contain columns: task (Roman numerals), response (in [0,1]).
    Returns a result dict with parameters, loss, meta.
    """
    # Setup: Configure device and reproducibility
    device = device_from_string(config.device)
    _set_seeds(config.seed)

    # Extract training data from dataframe
    tasks = list(df_group["task"].astype(str).values)
    targets = torch.tensor(df_group["response"].astype(float).values, dtype=torch.float32, device=device)

    loss_fn = LOSS_REGISTRY[config.loss_name]

    # Track best + collect all restart records
    best_loss = math.inf
    best_state = None
    best_init_state = None
    best_seed_used = None
    best_restart_index = None
    best_curve: List[float] = []
    all_restarts: List[Dict] = []

    # Multi-restart optimization loop to avoid local minima
    # Each restart uses a different random initialization
    best_module = None  # type: ignore
    for r in range(config.restarts):
        # Set unique seed for this restart to get different random initialization
        _set_seeds(config.seed + r)
        
        # Create fresh parameter module for this restart
        module = create_parameter_module(config.link, config.num_params)
        module.to(device)

        # Random parameter initialization 
        for p in module.parameters():
            nn.init.normal_(p, mean=0.0, std=0.1)

        # Capture initial parameter state before training
        init_params = {k: v.detach().cpu().item() for k, v in module.get_params().items()}  # type: ignore[call-arg]

        # Run optimization algorithm (learning loop)
        # This is where the actual parameter fitting happens
        t0 = time.perf_counter()
        if config.optimizer == "lbfgs":
            loss_val, curve = _run_lbfgs(module, config.link, tasks, targets, loss_fn, max_iter=config.epochs)
        else:
            loss_val, curve = _run_adam(module, config.link, tasks, targets, loss_fn, lr=config.lr, epochs=config.epochs)
        duration = time.perf_counter() - t0

        final_params = {k: v.detach().cpu().item() for k, v in module.get_params().items()}  # type: ignore[call-arg]

        # Optionally compute quick per-restart metrics (no CV) using current final params
        restart_metrics = {}
        if collect_restart_metrics:
            # Build tensors for metric calc (duplicate logic kept light)
            param_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in final_params.items()}
            preds_restart: List[torch.Tensor] = []
            for roman in tasks:
                preds_restart.append(roman_task_to_probability(roman, config.link, param_tensors))
            pred_vec_r = torch.stack(preds_restart, dim=0).cpu()
            targ_vec_r = targets.detach().cpu()
            diff_r = pred_vec_r - targ_vec_r
            sse_r = float(torch.sum(diff_r ** 2).item())
            n_r = pred_vec_r.shape[0]
            mse_r = sse_r / n_r if n_r else float("nan")
            mae_r = float(torch.mean(torch.abs(diff_r)).item()) if n_r else float("nan")
            rmse_r = float(torch.sqrt(torch.tensor(mse_r)).item()) if n_r else float("nan")
            y_mean_r = torch.mean(targ_vec_r) if n_r else torch.tensor(float("nan"))
            sst_r = float(torch.sum((targ_vec_r - y_mean_r) ** 2).item()) if n_r else float("nan")
            r2_r = float(1.0 - (sse_r / sst_r)) if sst_r and sst_r > 0 else float("nan")
            k_r = int(config.num_params)
            sse_safe_r = sse_r if sse_r > 1e-12 else 1e-12
            aic_r = float(n_r * math.log(sse_safe_r / n_r) + 2 * k_r) if n_r else float("nan")
            bic_r = float(n_r * math.log(sse_safe_r / n_r) + k_r * math.log(n_r)) if n_r else float("nan")
            # Simple ECE10 reuse
            def _ece10(pred: torch.Tensor, true: torch.Tensor) -> float:
                edges = torch.linspace(0.0, 1.0, steps=11)
                ece_val = 0.0
                total = pred.shape[0]
                for b in range(10):
                    lo = edges[b].item()
                    hi = edges[b + 1].item()
                    mask = (pred >= lo) & (pred < hi) if b < 9 else (pred >= lo) & (pred <= hi)
                    if torch.any(mask):
                        p_mean = float(torch.mean(pred[mask]).item())
                        y_mean_b = float(torch.mean(true[mask]).item())
                        weight = float(torch.count_nonzero(mask).item()) / total
                        ece_val += weight * abs(p_mean - y_mean_b)
                return float(ece_val)
            ece_r = _ece10(pred_vec_r, targ_vec_r) if n_r else float("nan")
            restart_metrics = {
                "rmse": rmse_r,
                "mae": mae_r,
                "r2": r2_r,
                "aic": aic_r,
                "bic": bic_r,
                "ece_10bin": ece_r,
            }

        # Record this restart
        all_restarts.append({
            "restart_index": r,
            "seed": config.seed + r,
            "loss_final": float(loss_val),
            "params": final_params,
            "init_params": init_params,
            "curve": curve,
            "status": "success",
            "duration_sec": float(duration),
            **restart_metrics,
        })

        # Keep track of best result across all restarts based on final loss
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = final_params
            best_init_state = init_params
            best_seed_used = config.seed + r
            best_restart_index = r
            best_curve = curve
            # Keep a deep copy of the module at best state (with unconstrained params)
            best_module = copy.deepcopy(module)

    # Post-training evaluation: Compute comprehensive metrics on best fitted model
    n = int(len(tasks))
    if n == 0:
        raise ValueError("No rows to fit after filtering.")

    # Reconstruct parameter tensors from best result for evaluation
    if best_state is None:
        raise RuntimeError("No successful restart produced a best_state (unexpected).")

    if config.link == "logistic":
        param_tensors = {
            "pC1": torch.tensor(best_state["pC1"], dtype=torch.float32),
            "pC2": torch.tensor(best_state["pC2"], dtype=torch.float32),
            "w0": torch.tensor(best_state["w0"], dtype=torch.float32),
            "w1": torch.tensor(best_state["w1"], dtype=torch.float32),
            "w2": torch.tensor(best_state["w2"], dtype=torch.float32),
        }
    else:
        param_tensors = {
            "pC1": torch.tensor(best_state["pC1"], dtype=torch.float32),
            "pC2": torch.tensor(best_state["pC2"], dtype=torch.float32),
            "b": torch.tensor(best_state["b"], dtype=torch.float32),
            "m1": torch.tensor(best_state["m1"], dtype=torch.float32),
            "m2": torch.tensor(best_state["m2"], dtype=torch.float32),
        }

    # Generate predictions using fitted parameters for metric calculation
    preds_eval: List[torch.Tensor] = []
    for roman in tasks:
        preds_eval.append(roman_task_to_probability(roman, config.link, param_tensors))
    pred_vec = torch.stack(preds_eval, dim=0).cpu()
    targ_vec = targets.detach().cpu()

    # Compute  metrics (loss, error measures, goodness of fit)
    diff = pred_vec - targ_vec
    sse = float(torch.sum(diff ** 2).item())
    mse = sse / n
    mae = float(torch.mean(torch.abs(diff)).item())
    rmse = float(torch.sqrt(torch.tensor(mse)).item())
    y_mean = torch.mean(targ_vec)
    sst_val = float(torch.sum((targ_vec - y_mean) ** 2).item())
    r2 = float(1.0 - (sse / sst_val)) if sst_val > 0 else float("nan")

    # Also compute task-aggregated metrics to match LOOCV granularity
    # Aggregate targets by unique task (average over any duplicate rows per task)
    try:
        # Build per-task means for targets and predictions (predictions are constant within a task)
        from collections import defaultdict
        idx_by_task: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(tasks):
            idx_by_task[str(t)].append(i)
        task_means_y: List[float] = []
        task_means_pred: List[float] = []
        for t, idxs in idx_by_task.items():
            tv = targ_vec[idxs]
            pv = pred_vec[idxs]
            task_means_y.append(float(torch.mean(tv).item()))
            # Predictions are identical within a task; still average for safety
            task_means_pred.append(float(torch.mean(pv).item()))
        if len(task_means_y) >= 2:
            import numpy as _np
            y_task = _np.array(task_means_y, dtype=float)
            p_task = _np.array(task_means_pred, dtype=float)
            sse_task = float(_np.sum((p_task - y_task) ** 2))
            sst_task = float(_np.sum((y_task - _np.mean(y_task)) ** 2))
            r2_task = float(1.0 - (sse_task / sst_task)) if sst_task > 0 else float("nan")
            rmse_task = float(_np.sqrt(_np.mean((p_task - y_task) ** 2)))
        else:
            r2_task = float("nan")
            rmse_task = float("nan")
    except Exception:
        # Be robust to any unexpected shape issues
        r2_task = float("nan")
        rmse_task = float("nan")

    def _ece_10(pred: torch.Tensor, true: torch.Tensor) -> float:
        """Compute Expected Calibration Error (ECE) using 10 equally-sized bins.
        
        ECE measures the difference between predicted probabilities and actual outcomes
        across probability bins. For each bin, it computes the weighted absolute difference
        between the mean predicted probability and the mean actual outcome.
        
        Args:
            pred: Predicted probabilities in [0,1]
            true: True binary outcomes in [0,1]
            
        Returns:
            ECE value as a float, where 0 indicates perfect calibration
        """
        edges = torch.linspace(0.0, 1.0, steps=11)
        ece = 0.0
        total = pred.shape[0]
        for b in range(10):
            lo = edges[b].item()
            hi = edges[b + 1].item()
            mask = (pred >= lo) & (pred < hi) if b < 9 else (pred >= lo) & (pred <= hi)
            if torch.any(mask):
                p_mean = float(torch.mean(pred[mask]).item())
                y_mean_b = float(torch.mean(true[mask]).item())
                weight = float(torch.count_nonzero(mask).item()) / total
                ece += weight * abs(p_mean - y_mean_b)
        return float(ece)

    # Calculate calibration metric
    ece10 = _ece_10(pred_vec, targ_vec)

    # Compute information criteria for model selection
    k = int(config.num_params)
    sse_safe = sse if sse > 1e-12 else 1e-12
    aic = float(n * math.log(sse_safe / n) + 2 * k)
    bic = float(n * math.log(sse_safe / n) + k * math.log(n))

    # Perform LOOCV if enabled
    loocv_results = None
    if config.enable_loocv:
        try:
            from .cross_validation import perform_loocv_single_group
            logger.info("Performing leave-one-out cross-validation...")
            loocv_results = perform_loocv_single_group(df_group, config)
            logger.info(f"LOOCV completed: RMSE={loocv_results['loocv_metrics']['loocv_rmse']:.4f}")
        except Exception as e:
            logger.warning(f"LOOCV failed: {e}")
            loocv_results = None

    # Package complete results from training and evaluation
    uncertainty_block = None

    if getattr(config, "compute_uncertainty", False):
    # --- Parameter Uncertainty Estimation ---------------------------------------------
    # We approximate standard errors (SEs) for the *constrained* model parameters
    # (those returned in best_state, e.g., pC1, w0, m1) by:
    #   1. Computing the Jacobian J of the predicted probabilities w.r.t. the
    #      unconstrained (raw) parameter vector theta at the fitted optimum.
    #   2. Using a Gauss-Newton / Fisher-style approximation to the Hessian:
    #         H ≈ (J^T J)
    #      (for an MSE loss the factor 2/n scales similarly across params and is
    #       absorbed into the residual variance estimate s^2 = SSE / (n - p)).
    #   3. Cov(theta) ≈ s^2 * (J^T J)^{-1} with mild Tikhonov regularization
    #      if the matrix is ill-conditioned.
    #   4. We then transform SEs from the unconstrained scale to the constrained
    #      parameter scale via the chain rule derivative of each transform:
    #         logistic:   p = sigmoid(theta)          => dp/dtheta = p (1 - p)
    #         tanh-bound: w = B * tanh(theta)         => dw/dtheta = B * (1 - tanh(theta)^2)
    #   5. Reported SEs are therefore approximate and assume local quadratic
    #      curvature, independent observations, and well-specified model.
    # Interpretation: A 95% Wald interval can be formed as param ± 1.96 * SE.
    # Caveats: Does not account for model misspecification or parameter tying
    # (ties reuse the same raw parameter so SEs for tied params mirror each other).
    # -----------------------------------------------------------------------------------
        warnings_list: List[str] = []
        try:
            # Ensure best_module exists; otherwise reconstruct (approximate) by inverting transforms
            if best_module is None:
                warnings_list.append("Best module not captured; uncertainty skipped.")
            else:
                best_module.to(device)
                best_module.train(False)

                # Collect unconstrained parameter tensors (those that require grad)
                unconstrained_params: List[torch.Tensor] = []
                param_names: List[str] = []
                for name, p in best_module.named_parameters():
                    if p.requires_grad:
                        unconstrained_params.append(p)
                        param_names.append(name)

                p_dim = len(unconstrained_params)
                if p_dim == 0:
                    warnings_list.append("No trainable parameters found for uncertainty computation.")
                else:
                    # Build Jacobian J (n x p) of predictions w.r.t. unconstrained params
                    J_rows: List[torch.Tensor] = []
                    # Recompute predictions with grad
                    preds_for_jac: List[torch.Tensor] = []
                    for roman in tasks:
                        preds_for_jac.append(roman_task_to_probability(roman, config.link, best_module.get_params()))  # type: ignore[arg-type]
                    preds_all = torch.stack(preds_for_jac, dim=0)  # (n,)
                    for i in range(preds_all.shape[0]):
                        grads = torch.autograd.grad(preds_all[i], unconstrained_params, retain_graph=True, allow_unused=True)
                        g_vec = []
                        for g in grads:
                            if g is None:
                                g_vec.append(torch.zeros((), device=preds_all.device))
                            else:
                                g_vec.append(g.view(-1))
                        J_rows.append(torch.stack(g_vec))  # shape (p,)
                    J = torch.stack(J_rows, dim=0)  # (n, p)

                    # Gauss-Newton approximation: H ≈ 2/n * J^T J for MSE-like losses; covariance ≈ s2 * (J^T J)^{-1}
                    JTJ = J.T @ J  # (p, p)
                    cond = None
                    try:
                        cond = float(torch.linalg.cond(JTJ).item())
                    except Exception:
                        warnings_list.append("Condition number computation failed.")

                    # Residual variance estimate
                    p_eff = p_dim
                    dof = max(n - p_eff, 1)
                    s2_hat = sse / dof
                    # Regularize if near-singular
                    reg_eps = 1e-6
                    try:
                        JTJ_inv = torch.linalg.inv(JTJ + reg_eps * torch.eye(p_dim, device=JTJ.device))
                    except Exception:
                        warnings_list.append("JTJ inversion failed; adding stronger regularization.")
                        JTJ_inv = torch.linalg.pinv(JTJ + 1e-4 * torch.eye(p_dim, device=JTJ.device))

                    var_theta = s2_hat * JTJ_inv.diag()  # (p,)
                    se_theta = torch.sqrt(torch.clamp(var_theta, min=0.0)).cpu().numpy().tolist()

                    # Map unconstrained param names to constrained exposed parameter names via transforms
                    # Build dictionary of constrained params (already computed in best_state)
                    se_constrained: Dict[str, float] = {}

                    # Helper to fetch raw theta for derivative evaluation
                    name_to_theta = {name: p.detach() for name, p in best_module.named_parameters()}  # type: ignore[name-defined]

                    # Determine model type by key in best_state
                    is_logistic = config.link == "logistic"

                    # Build mapping from exposed constrained param to (unconstrained name(s), transform)
                    # Logistic: pC1 (theta_pC), pC2 (theta_pC2 or share), w0(theta_w0), w1(theta_w1), w2(theta_w2 or share)
                    # Noisy-OR: pC1(theta_pC), pC2(theta_pC2 or share), b(theta_b), m1(theta_m1), m2(theta_m2 or share)
                    bound = 3.0
                    theta_name_index = {n: i for i, n in enumerate(param_names)}

                    def sigmoid(x: torch.Tensor) -> torch.Tensor:
                        return torch.sigmoid(x)

                    for exposed_name in best_state.keys():  # type: ignore[union-attr]
                        if is_logistic:
                            if exposed_name in ("pC1", "pC2"):
                                base_name = "theta_pC" if (exposed_name == "pC1" or best_module.tying.tie_priors) else "theta_pC2"  # type: ignore[attr-defined]
                                if base_name in theta_name_index:
                                    idx = theta_name_index[base_name]
                                    theta_val = name_to_theta[base_name]
                                    p_val = sigmoid(theta_val)
                                    deriv = p_val * (1 - p_val)
                                    se_constrained[exposed_name] = float(deriv.item() * se_theta[idx])
                            elif exposed_name in ("w0", "w1", "w2"):
                                base_name = f"theta_{exposed_name}"
                                if exposed_name == "w2" and best_module.tying.tie_strengths:  # type: ignore[attr-defined]
                                    base_name = "theta_w1"
                                if base_name in theta_name_index:
                                    idx = theta_name_index[base_name]
                                    theta_val = name_to_theta[base_name]
                                    deriv = bound * (1 - torch.tanh(theta_val) ** 2)
                                    se_constrained[exposed_name] = float(deriv.item() * se_theta[idx])
                        else:
                            # noisy_or
                            if exposed_name in ("pC1", "pC2"):
                                base_name = "theta_pC" if (exposed_name == "pC1" or best_module.tying.tie_priors) else "theta_pC2"  # type: ignore[attr-defined]
                            elif exposed_name == "b":
                                base_name = "theta_b"
                            elif exposed_name in ("m1", "m2"):
                                base_name = "theta_m1" if (exposed_name == "m2" and best_module.tying.tie_strengths) else f"theta_{exposed_name}"  # type: ignore[attr-defined]
                            else:
                                base_name = None
                            if base_name and base_name in theta_name_index:
                                idx = theta_name_index[base_name]
                                theta_val = name_to_theta[base_name]
                                p_val = sigmoid(theta_val)
                                deriv = p_val * (1 - p_val)
                                se_constrained[exposed_name] = float(deriv.item() * se_theta[idx])

                    if not se_constrained:
                        warnings_list.append("Failed to map any constrained parameters for SE computation.")

                    uncertainty_block = {
                        "method": "gauss_newton",
                        "se": se_constrained,
                        "condition_number": cond,
                        "warnings": warnings_list or None,
                    }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Uncertainty computation failed: {e}")
            uncertainty_block = {
                "method": "gauss_newton",
                "se": {},
                "warnings": [f"failed: {e}"],
            }

    result = {
        "loss": best_loss,
        "params": best_state,
        "init_params": best_init_state,
        "num_rows": int(df_group.shape[0]),
        "tasks": tasks,
        "seed_used": best_seed_used,
        "restart_index": best_restart_index,
        "all_restarts": all_restarts,
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            # Task-aggregated metrics aligned with LOOCV computation
            "r2_task": r2_task,
            "rmse_task": rmse_task,
            "ece_10bin": ece10,
            "sse": sse,
            "sst": sst_val,
            "aic": aic,
            "bic": bic,
        },
        "loss_curve": best_curve,
    "loocv": loocv_results,
    "uncertainty": uncertainty_block,
    }
    return result


