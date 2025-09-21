# Human Fitting Modes

This repository supports three complementary ways to fit CBNs to human data. These modes control which rows are used and how artifacts are named so that results never overwrite each other. Note: the RW17 data has each individual human only covered for a small subset of tasks making individual causal Bayes nets for RW17 data mute.

## Modes at a glance

- Aggregated ("humans")
  - Uses pre-aggregated human responses (subject == "humans").
  - Good for quick, legacy-style fits.
- Pooled ("humans-pooled")
  - Pools all individual human rows into a single synthetic agent.
  - Good for comparing a typical human profile against CBNs using all individual responses.
- Individual ("human-<id>")
  - Fits each individual separately (one model per participant).
  - Good for variability analyses and parameter distributions.

Human-like labels include: humans, humans-pooled, human-<id>. Non-human agents are always unaffected by human-mode filters.

## Where this is controlled

- Grid runner: `scripts/grid_fit_models.py`
  - Flag: `--humans-mode {auto, aggregated, pooled, individual}`
  - Special tokens for `--agents`:
    - `humans` → aggregated by default; with `--humans-mode pooled` becomes `humans-pooled`.
    - `humans-pooled` → pooled synthetic agent explicitly.
    - `all-humans` → expands to every individual `human-<id>`.
- Export tables: `scripts/export_cbn_best_fits.py`
  - Flag: `--humans-mode {all, aggregated, pooled, individual}`
  - Produces per-mode outputs (see Tag suffixes below) and writes humans_mode into `manifest.json`.
- Plots/analysis
  - Parameters heatmap: `scripts/visualize_cbn_parameters.py` → `--humans-mode` filters agents and prefers tags with matching suffix.
  - Agent vs CBN plots (manifest-driven): `scripts/plot_agent_vs_cbn_predictions.py` → `--humans-mode` filters manifest rows and writes into a mode subfolder when needed.

## Output isolation (no overwrites)

- Model fits (grid runner): results are written under
  - `results/model_fitting/<experiment>/<lr_subdir>/humans_mode/<aggregated|pooled|individual>/`
  - Individual fits also live under `variants/<agent>/` (e.g., `variants/human-42/`).
- Exported winners/LaTeX (exporter): tag gets a suffix:
  - `hm-agg`, `hm-pooled`, or `hm-indiv` appended to the tag.
  - `manifest.json` records `{"humans_mode": "aggregated|pooled|individual"}`.
- Plots may add a subfolder named after the humans-mode if the tag itself has no `hm-*` suffix.

## Data loading semantics

- Aggregated vs non-aggregated files
  - Aggregated: versioned Roman file (e.g., `2_v_collider_cleaned_data_roman.csv`) or legacy `humans_avg_equal_sample_size_cogsci.csv`.
  - Non-aggregated: versioned non-Roman file (e.g., `2_v_collider_cleaned_data.csv`).
  - Friendly fallback: when non-aggregated is selected but the file is missing, the loader will try `2_v_collider_cleaned_data_indiv_humans.csv` if present.
- Prompt category mapping for human-like rows
  - If you request `--prompt-categories numeric` but the data only has `single_numeric_response`, the grid runner automatically remaps to `single_numeric_response` for human-like agents.
- Temperature filter
  - Human-like rows are permitted to have NaN temperatures when a temperature filter is applied; they are not dropped for missing temperature.
- Human ID column aliases
  - Auto-detected for splitting/pooled modes: `human_subj_id` (preferred), `humans_subj_id`, `participant_id`, `worker_id`, `subject_id`, `human_id`.
  - Naming normalization: individual agents use `human-<id>` (singular), not `humans-<id>`.

## Domain semantics

- `--domains all` means pooled across all available domains (single fit) unless `--by-domain` is set.
- `--by-domain` fits each domain separately; if no domain data exist for a combination, it falls back to a pooled fit for that case.

## Quick examples

- Pooled humans fit (no overwrite with other modes):

```bash
python scripts/grid_fit_models.py \
  --experiment rw17_indep_causes --version 2 \
  --models noisy_or --params 3 4 \
  --optimizer lbfgs --loss huber --restarts 10 \
  --enable-loocv --humans-mode pooled \
  --prompt-categories numeric --agents humans
```

- Individual humans (all participants):

```bash
python scripts/grid_fit_models.py \
  --experiment rw17_indep_causes --version 2 \
  --models noisy_or --params 3 4 \
  --optimizer lbfgs --loss huber --restarts 10 \
  --enable-loocv --humans-mode individual \
  --prompt-categories numeric --agents all-humans
```

- Aggregated humans (legacy-style):

```bash
python scripts/grid_fit_models.py \
  --experiment rw17_indep_causes --version 2 \
  --models noisy_or --params 3 4 \
  --optimizer lbfgs --loss huber --restarts 10 \
  --enable-loocv --humans-mode aggregated \
  --prompt-categories numeric --agents humans
```

- Export per-mode winners and LaTeX tables (tag gets `hm-*` suffix):

```bash
python scripts/export_cbn_best_fits.py \
  --experiments rw17_indep_causes \
  --models noisy_or --versions 2 \
  --humans-mode pooled --export-params
```

- Visualize parameters (heatmaps) for pooled humans only:

```bash
python scripts/visualize_cbn_parameters.py \
  --experiment rw17_indep_causes \
  --tag-glob "v2_*noisy_or*" \
  --humans-mode pooled
```

- Plot agent vs CBN (manifest-driven) for pooled humans tag:

```bash
python scripts/plot_agent_vs_cbn_predictions.py \
  --experiment rw17_indep_causes --version 2 \
  --tag v2_noisy_or_pcnum_p3-4_lr0.1_hm-pooled \
  --agents humans-pooled --no-show
```

## Troubleshooting

- “No groups formed after filtering”: if using `--prompt-categories numeric` for human-like agents, the runner will remap to `single_numeric_response` when that’s the only available column; ensure your processed data include the expected prompt category.
- “Data file not found … cleaned_data.csv”: for non-aggregated runs, the loader now tries `*_cleaned_data_indiv_humans.csv` automatically if the standard file is missing.
- Counting individuals for `all-humans`: use `--inspect-filters` to print discovered agents and confirm the number of `human-<id>` variants.
