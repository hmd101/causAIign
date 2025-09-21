# Stage 04 — CBN fitting and evaluation

This stage fits causal Bayes net (CBN) models to agent outputs and exports best-fit summaries for downstream analysis.

Primary tools:
- `fit_models.py` -- fit a causal Bayes net to an agent's likelihood judgments 
- `export_cbn_best_fits.py` — for evaluation: read per-run fit results, select best configuration per experiment×prompt-category×agent, and write:
	- `results/cross_cogn_strategies/winners.csv`
	- a manifest with selected model/params

Example:
- python scripts/04_cbn_fit_and_eval/export_cbn_best_fits.py --input-root results/modelfits --out-root results/cross_cogn_strategies

Outputs feed into Stage 05 summaries/plots like `summarize_fit_metric_ranges.py` and `plot_fit_metric_distributions.py`.



### Causal Bayes Net Model Fitting Public API 

You can fit simple causal Bayes net models to both human and LLM responses programmatically without invoking the CLI using the public helpers in `causalign.analysis.model_fitting.api`.

Minimal example:

```python
from pathlib import Path
import pandas as pd
from causalign.analysis.model_fitting.trainer import FitConfig
from causalign.analysis.model_fitting.api import fit_dataset, load_index, query_top_by_primary_metric

# DataFrame must contain at least: subject, task, response (optionally prompt_category, domain)
df = pd.DataFrame({
    "subject": ["agentA"] * 5,
    "task": ["I","II","III","IV","V"],
    "response": [0.1,0.2,0.9,0.4,0.7],
})

cfg = FitConfig(link="logistic", num_params=3, loss_name="mse", optimizer="lbfgs", epochs=50, restarts=5)
out_dir = Path("results/model_fits_demo")

results = fit_dataset(
    df,
    cfg,
    output_dir=out_dir,
    per_restart_metrics=True,
    primary_metric="aic",
)

index_df = load_index(out_dir)
print(index_df[["agent","loss","aic","restart_loss_median"]])

# Retrieve best row by primary metric (global or per group)
best = query_top_by_primary_metric(out_dir, "aic")
print("Best spec/group:", best.iloc[0].short_spec_hash, best.iloc[0].short_group_hash)
```

#### Ranking Helpers

For convenience, ranking utilities operate on the Parquet index:

```python
from causalign.analysis.model_fitting.api import rank_index, top_by_metric

ranked = rank_index(index_df, "aic")  # adds aic_rank (1 = best)
top_per_agent = top_by_metric(index_df, "aic", group_cols=["agent"])
```

Metrics treated as higher-is-better currently: `r2`. All others (loss, aic, bic, rmse, mae) are minimized.

#### Optimization Stability Metrics

Per-restart optimization stability is summarized with two scale-normalized dispersion measures computed from aggregated restart statistics (default stability metric: loss):

* `stability_cv = std / |mean|`
* `stability_rel_range = (max - min) / |mean|`

The flag `stable_opt` is True if either measure is below its threshold (defaults: CV ≤ 0.02 or relative range ≤ 0.10) unless `--stability-require-all` is passed (requires both). See `STABILITY_METRICS.md` for full definitions, edge cases (single restart, outliers), and tuning guidance.

#### File Artifacts

Each fit writes:
* `fit_<shortSpec>_<shortGroup>.json` – structured schema (schema_version, spec_hash, group_hash, restarts, metrics, ranking)
* `fit_index.parquet` – deduplicated index with restart aggregate statistics (loss, rmse, aic mean/median/variance)

#### Selection Rules

Restart selection strategy is controlled via `selection_rule` in `RankingPolicy` (CLI: `--selection-rule`):
* `best_loss` (default) – choose restart with minimal training loss
* `median_loss` – pick restart at median loss (robust to outliers)
* `best_primary_metric` – choose restart optimizing the primary metric if per-restart metric captured

#### Cross-Validation Scope

Current schema (v1.2.0) defines cross-validation metrics at the group level only (no per-restart CV). Aggregated CV metrics appear in the `metrics` block (`cv_rmse`, `cv_r2` if LOOCV enabled).

#### Reproducibility

stable hashes (`spec_hash`, `group_hash`, `data_hash`) allow detecting identical specification/data combinations. The short 12-char prefixes are used in filenames; full SHA256 stored in JSON.
