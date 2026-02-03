# Cognitive Strategies Workflow

This document explains the analysis workflow around the two scripts:

- `scripts/cognitive_stragies.py` — compute per-agent raw/model indices and classify normative reasoners inside each parameter-analysis run (per experiment/tag).
- `scripts/aggregate_cognitive_strategies.py` — crawl all classified runs, aggregate results across experiments/tags, summarize, plot, and export CSV/LaTeX tables.

The workflow is designed so you can vary thresholds (EA, MV, R²) and have the outputs reflect that choice, while keeping runs reproducible and easy to compare.

## What the scripts do

### 1) `cognitive_stragies.py`

Inputs (per experiment + tag under `results/parameter_analysis/<experiment>/<tag>/`):
- `winners_with_params.csv` (required): best-fit CBN parameters per agent × prompt category.
- `winners.csv` (optional): includes meta like `loocv_r2`.
- Processed responses loaded via the pipeline (see `causalign.analysis.model_fitting.data`).

Outputs (written to `.../<experiment>/<tag>/cogn_analysis/`):
- `strategy_metrics.csv`: per agent × prompt_category raw means → EA_raw and MV_raw, plus model predictions → EA_model and MV_model, and fitted parameters (b, m1, m2, pC1, pC2, m_bar, m_gap). Values are on [0, 1] scale.
- `classified_strategy_metrics.csv`: if `--classify-reasoning` is provided, the above with boolean flags and thresholds reflected. Saved into a thresholds-encoded subfolder:
  - `cogn_analysis/ea_diff_<EA>_mv_diff_<MV>_loocv_r2_<R2>/classified_strategy_metrics.csv`
  - The CSV includes `ea_diff_threshold`, `mv_diff_threshold`, and `loocv_r2_threshold` columns for traceability.
- Optional violin plots (parameters for normative and non-normative subsets) if `--plot` is used; saved next to the classified CSV.
- Optional robustness table across prompt categories (`robustness_<pcnum>_to_<pccot>.csv`) if both categories present.

Key definitions:
- EA_raw = VIII − VI (equivalently c − a) based on per-task mean responses.
- MV_raw = IV − V (equivalently d − e) based on per-task mean responses.
- The raw responses are treated as probabilities in [0, 1]. If upstream data are in 0–100, make sure the pipeline normalizes them to [0,1].

Prompt category normalization:
- Numeric labels: `{"pcnum", "numeric", "num", "single_numeric", "single_numeric_response"}`
- CoT labels: `{"pccot", "cot", "chain_of_thought", "chain-of-thought", "cot_stepwise"}`
- The script tries reasonable synonyms to avoid missing data due to label differences.

Threshold-based classification (row-wise):
- `meets_ea`: EA_raw > `--ea-diff`
- `meets_mv`: |MV_raw| < `--mv-diff`
- `meets_loocv_r2`: loocv_r2 > `--loocv-r2`
- `normative_reasoner`: all three must be true.

CLI (most relevant flags):
- `--experiment`, `--version`, `--tag` (required trio)
- `--classify-reasoning` (writes thresholded copy)
- `--ea-diff`, `--mv-diff`, `--loocv-r2` (thresholds)
- `--plot` (produces two violin PDFs next to the classified CSV)
- `--pcnum-label`, `--pccot-label` (for robustness across prompt categories)

Example (optional commands):

```bash
# Optional example only
python scripts/cognitive_stragies.py \
  --experiment rw17_overloaded_e \
  --version 2 \
  --tag v1_noisy_or_pccot_p3-4_lr0.1_noh \
  --classify-reasoning --plot \
  --ea-diff 0.3 --mv-diff 0.05 --loocv-r2 0.89
```

This will produce:
- `.../cogn_analysis/strategy_metrics.csv`
- `.../cogn_analysis/ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89/classified_strategy_metrics.csv`
- `.../cogn_analysis/ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89/violin_normative_params.pdf`
- `.../cogn_analysis/ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89/violin_non_normative_params.pdf`


### 2) `aggregate_cognitive_strategies.py`

Inputs:
- Crawls all `classified_strategy_metrics.csv` files under `results/parameter_analysis/**/cogn_analysis/**/` and infers:
  - Experiment name and tag from the path
  - Thresholds (`ea_diff`, `mv_diff`, `loocv_r2`) from the thresholds-encoded parent folder

Outputs (by default under `results/cross_cogn_strategies/`):
- `masters_classified_strategy_metrics.csv`: concatenated dataset across experiments/tags with parsed thresholds and normalized prompt categories.
- `normative_share_by_exp_pc.csv` + `.tex`: per experiment × prompt-category, absolute counts and shares of normative reasoners, including threshold columns.
- `normative_by_agent_matrix.csv` + `.tex`: wide matrix marking normative reasoner status (True/False/blank) per agent across conditions, with per-agent aggregates.
- Violin plots (if `--plot`) for parameters split by normative vs non-normative, grouped by experiment with paired Numeric/CoT colors and legend.

Example (optional commands):

```bash
# Optional example only
python scripts/aggregate_cognitive_strategies.py \
  --root results/parameter_analysis \
  --out results/cross_cogn_strategies \
  --plot
```


## How to vary thresholds and keep results comparable

The current pattern already supports multi-threshold runs:

- `cognitive_stragies.py` writes one classified copy per threshold triple inside a named subfolder like `ea_diff_0.3_mv_diff_0.05_loocv_r2_0.89/`.
- `aggregate_cognitive_strategies.py` parses these values from the folder names and propagates them into the output CSVs.

Recommended enhancements for robust tracking:

1) Save a metadata sidecar next to every thresholded output
- Write a small `metadata.json` file alongside `classified_strategy_metrics.csv` with:
  - `experiment`, `tag`, threshold values, script name/version, timestamp, git commit (if available)
  - counts (n agents, n normative, n per-prompt-category), and any filters applied

Example structure:
```json
{
  "script": "cognitive_stragies.py",
  "experiment": "rw17_overloaded_e",
  "tag": "v1_noisy_or_pccot_p3-4_lr0.1_noh",
  "thresholds": {"ea_diff": 0.3, "mv_diff": 0.05, "loocv_r2": 0.89},
  "timestamp": "2025-09-06T10:22:33Z",
  "git_commit": "abc1234",
  "counts": {"agents": 42, "normative": 18, "pcnum": 21, "pccot": 21}
}
```

2) Maintain a run index CSV
- Append a row per run to `results/cross_cogn_strategies/run_index.csv` capturing the same fields as above. This gives a single table to join against plots/tables later.

3) Include thresholds in aggregated outputs
- This is already done for the summary table; keep it consistent everywhere (agent matrix, LaTeX headers, plots titles or captions).

4) Optional: config-first runs
- Allow an optional `--config thresholds.yaml` to define named threshold sets. The script can expand these into multiple runs, writing one subfolder per set automatically.

Example `thresholds.yaml`:
```yaml
sets:
  strict: { ea_diff: 0.35, mv_diff: 0.04, loocv_r2: 0.92 }
  default: { ea_diff: 0.30, mv_diff: 0.05, loocv_r2: 0.89 }
  lenient: { ea_diff: 0.25, mv_diff: 0.07, loocv_r2: 0.80 }
```


## Downstream visualization ideas

- Violin/box plots for parameter distributions (b, m1, m2, pC1, pC2) split by normative status and prompt category, grouped by experiment. Already supported; consider adding counts/thresholds in titles and color legends.
- Heatmaps of normative share:
  - Experiment × Prompt Category
  - As a function of thresholds (facet by threshold set or use small multiples)
- Threshold sweep curves:
  - Plot normative share vs. each threshold while holding the others fixed; per experiment or aggregated.
- Agent matrix heatmaps:
  - Convert `normative_by_agent_matrix.csv` to a heatmap where True/False/NA use distinct colors; one panel per experiment.
- Stability overlays:
  - Scatter EA_raw vs. MV_raw with threshold bands; color by normative.
- Interactive dashboards:
  - Use Altair or Plotly to filter by experiment/tag/threshold set and view dynamic summaries.


## Suggested implementation tweaks (low effort)

These are small, additive changes to the scripts to improve traceability without breaking current behavior:

- In `cognitive_stragies.py`, after writing `classified_strategy_metrics.csv`, also write a `metadata.json` in the same folder with thresholds, experiment, tag, counts, and timestamp.
- In `aggregate_cognitive_strategies.py`, when generating outputs for a specific threshold triple, also write a `metadata.json` summarizing the files produced and the inferred thresholds. Optionally, append an entry to a global `run_index.csv`.
- Add a `--config thresholds.yaml` option (optional) to drive batch runs across named threshold sets.
- Add optional plot labels/titles that include thresholds and total counts (n normative / n total) for clarity.


## Troubleshooting

- Missing EA_raw/MV_raw for CoT: the script now tries multiple prompt-category synonyms; still, double-check the prompt labels in your processed dataset if values remain blank.
- Out-of-range values (>1): ensure upstream responses are normalized to [0,1]. If your pipeline emits percentages, divide by 100 before aggregation.
- Empty aggregation outputs: confirm `classified_strategy_metrics.csv` exist under the thresholds subfolders and that the folder names encode thresholds as `ea_diff_..._mv_diff_..._loocv_r2_...`.


## Summary

- Use `cognitive_stragies.py` to compute and classify per-run metrics with explicit thresholds. Each threshold set gets its own subfolder.
- Use `aggregate_cognitive_strategies.py` to build cross-run summaries, shares, matrices, and plots; thresholds flow through to the outputs.
- Persist thresholds and run metadata via `metadata.json` sidecars and a global `run_index.csv`. This makes threshold sweeps and downstream visualizations simple and reproducible.
