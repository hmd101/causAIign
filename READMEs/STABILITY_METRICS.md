## Optimization Stability Metrics

This document describes how optimization stability is quantified for model fits in CausalAlign.

### 1. Data Sources

Per-restart metrics are written (when enabled) to `restart_metrics.parquet` inside each experiment directory (and learning-rate subdirectories `lr*/`). The evaluation script (`scripts/evaluate_model_fits.py`) aggregates these to create per‐specification summary statistics. The aggregated index (`fit_index.parquet`) currently stores: `restart_<metric>_mean`, `restart_<metric>_median`, `restart_<metric>_var`, plus `restart_count`. Missing higher-order summaries (std, min, max, iqr, range) are derived on demand at evaluation time.

### 2. Aggregation Procedure

For a chosen stability metric (default: `loss`, mapping to `loss_final` in per-restart rows), group by `spec_hash`:

Stats computed: mean, std, min, max, interquartile range (iqr = Q3−Q1), variance, median, and range (= max−min). Column naming scheme:

```
restart_<metric>_{mean,std,min,max,iqr,var,median,range}
```

Population vs sample: current derivation uses the pandas default (sample std with ddof=1), so single-restart specs yield `std = NaN`. Planned enhancement: switch to population (`ddof=0`) to make single-restart std = 0.

### 3. Derived Stability Measures

Let m = restart_<metric>_mean, s = restart_<metric>_std, a = restart_<metric>_min, b = restart_<metric>_max. With denominator D = max(|m|, ε) (ε default 1e-8):

```
stability_cv        = s / D
stability_rel_range = (b - a) / D
```

Interpretation:
* `stability_cv` (coefficient of variation) captures typical relative dispersion across restarts (robust to a single outlier).
* `stability_rel_range` captures the worst-case spread (sensitive to any extreme restart).

### 4. Stability Flag Logic

```
stable_opt = (stability_cv <= cv_threshold) OR (stability_rel_range <= range_threshold)
```

unless `--stability-require-all` is set, in which case both conditions must hold. Default thresholds:

* `cv_threshold = 0.02`
* `range_threshold = 0.10`

Rationale: using both an average-dispersion and an extrema-based criterion reduces false confidence (CV alone) and overreaction to one noisy restart (range alone).

### 5. Edge Cases & Missing Data

| Situation | Effect on metrics |
|-----------|-------------------|
| Single restart | `std = NaN`, `range = 0`, `stability_cv = NA`, `stability_rel_range = 0` (treated stable via range) |
| Identical restart values | Both metrics 0 → stable |
| Mean near 0 | Denominator clamped by ε → avoids blow-up; very small mean makes criteria stricter |
| Missing min/max (no long-form rows) | Both stability metrics NA; `stable_opt` NA |
| Outlier restart | Range flags instability even if CV modest |

### 6. Threshold Tuning Guidance

Start with defaults. If many desirable specs fail only the CV test with CV in [0.02, 0.05] but have very small rel_range, consider relaxing `cv_threshold` to 0.03–0.05. Keep `range_threshold` conservative (≤ 0.15) to still catch genuine multi-modal convergence. For noisier objectives, a pairing like (CV ≤ 0.05, rel_range ≤ 0.15) is reasonable.

### 7. Recommended Review Workflow

1. Run evaluation with a set of  thresholds.
2. Filter to `stable_opt == True` for primary comparisons.
3. Inspect rows with `stable_opt == False` and compare `stability_cv` vs `stability_rel_range` to diagnose cause (outlier vs broad dispersion).
4. If a row barely fails (e.g. rel_range ≈ threshold), consider revisiting threshold or inspecting per-restart distribution visually.
5. Log any manual overrides for reproducibility.

### 8. Future Enhancements

* Persist std/min/max/iqr/range during fitting to eliminate runtime derivation.
* Switch to population std (ddof=0) and explicitly treat single-restart specs as trivially stable.
* Add `stability_iqr = iqr / |mean|` for outlier robustness.
* Provide a CLI option to force both criteria (`--stability-require-all`) or a stricter composite label.

### 9. Quick Formulas Cheat Sheet

```
CV  = std / |mean|
Range% = (max - min) / |mean|
Stable (default) if CV ≤ 0.02 OR Range% ≤ 0.10
```

---
For implementation details see `scripts/evaluate_model_fits.py` (derivation block) and the stability flag computation in `_compute_stability_flags`.
