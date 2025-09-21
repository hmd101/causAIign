# CBN best-fits exporter: model selection policy

This document explains how `scripts/export_cbn_best_fits.py` selects a single “winner” model per group and what filters impact the candidate set.

## Scope and filtering (which rows are considered)
For each requested experiment, the script loads the model fit index and applies these optional filters before selection:
- Version(s): defaults to the latest version in the index (numeric max if possible), or use `--versions`.
- Model link(s): restrict via `--models logistic noisy_or` (otherwise includes all present).
- Parameter tying counts: `--params 3 4` keeps only those tying configurations (3p/4p).
- Learning-rate subfolders: `--lr <tokens>` narrows to runs in matching LR subdirectories (e.g., `lr0p1`, `base`).
- Prompt categories: `--prompt-categories ...` (case-insensitive, with legacy aliases handled).
- Humans mode: `--humans-mode aggregated|pooled|individual` keeps only those human rows, while always keeping non-human agents.
- Domain scope: `--include-domains all` keeps pooled rows only (domain is NaN in the index); or pass explicit domain names to include.
- Exclude humans: `--exclude-humans` drops aggregated “humans”.

If any filter empties the scope, the experiment is skipped.

## Grouping keys (what constitutes a “winner”)
- By default, winners are chosen per `(agent, domain)` pair within each `(experiment, version)` scope.
- If `--no-collapse-prompt` is used and `prompt_category` exists, winners are chosen per `(prompt_category, agent, domain)` instead.

Pooled domain rows have `domain = NaN` in the index; in outputs these are labeled by `--pooled-domain-label` (default: `all`).

## Winner selection (primary metric, fallbacks, tie-breakers)
Within each group, the policy is:
1) Primary objective: maximize `loocv_r2` (drop NaNs if any finite exist).
2) Fallback chain if all `loocv_r2` are missing/NaN:
   - maximize `cv_r2`
   - maximize `r2`
   - minimize `bic`
   - minimize `aic`
3) Tie-breakers (applied in order when the leading metric is within a tiny epsilon or equivalent by stable sort):
   - lower `loocv_rmse`
   - lower `bic`
   - lower `aic`
   - lower `loss`
   - fewer `params_tying` (prefer simpler models)
   - smaller `short_spec_hash` (stable final tie-break)

The single row at the top after this ordering is the group’s winner.

## Presentation ordering and outputs
- After winners are selected, they are globally sorted by `loocv_r2` (descending; NaNs last) for the final table output.
- Outputs per experiment:
  - CSV: `results/parameter_analysis/<experiment>/<tag>/winners.csv` (winners + key metrics and provenance)
  - Optional CSV (default on): `winners_with_params.csv` with canonicalized parameter values (see below)
  - LaTeX: `publication/thesis/tuebingen_thesis_msc/tables/<experiment>/cbn_best_fits_<tag>.tex`
  - Manifest: JSON describing the filter settings used

The `<tag>` encodes filters (versions, models, prompt categories, tying counts, LR filters, human mode, etc.) for reproducibility.

## Columns shown vs. selection metrics
- Selection always uses the policy above (centered on `loocv_r2` with robust fallbacks) regardless of which columns are displayed.
- Flags like `--metrics-in-sample`, `--metrics-out-of-sample`, and `--exclude-metrics` only affect which metric columns appear in the LaTeX table; they do not change winner selection.

## Learning rate detection (display only)
- A learning-rate column is added only if winners use different LRs. The value is inferred by preference from:
  1) any existing LR-like column (`learning_rate`, `lr`, `opt_lr`), else
  2) the fit JSON’s optimizer settings, else
  3) the `lr_subdir` token (e.g., `lr0p1` -> `0.1`).

This does not influence selection—only display.

## Parameter export details (winners_with_params.csv)
- For each winner, the script opens the corresponding fit JSON, picks the best restart (lowest final loss), and extracts its parameter values.
- Parameters are canonicalized to CBN keys and tying is respected/expanded:
  - Canonical keys: `b`, `m1`, `m2`, `pC1`, `pC2`
  - Common aliases are mapped (e.g., `bias` -> `b`, `m` -> `m1`/`m2`, `pc` -> `pC1`/`pC2`).
  - If tying implies equality (e.g., 3p/4p), missing twins are duplicated from the provided value.
- Optionally, readout weights (`w0`, `w1`, `w2`) are included when `--include-readout-weights` is set.

## Notes and guardrails
- If the index lacks a `link` column, the script may backfill it by peeking into the fit JSONs so `--models` filtering still works.
- Missing metrics for display are warned about and omitted from the table; winner selection proceeds with the available metrics.
- When `--include-domains all` is used, `domain` is presented as the configured pooled label (`all` by default) in the outputs.

## TODO: Seed reuse policy for LOOCV
- Current behavior: LOOCV folds use deterministic, fold-specific seeds derived from the base seed and the held-out task. Full-data (in-sample) runs use restart seeds `seed + r`. LOOCV does not reuse the exact restart seed from the winning full-data fit.
- Potential enhancement: Add an option to reuse the winning full-data restart seed (or a fixed restart seed schedule) for LOOCV refits of the winning spec to align initialization exactly.
- Rationale: Could reduce between-spec variance due to differing initialization schedules and simplify reproducibility narratives when reporting the winner’s LOOCV metrics.
- Action items:
  1) Add a config flag (e.g., `--loocv-reuse-winner-seed`) that, when enabled, pins fold restart seeds to the winner’s best restart seed (or a fixed sequence).
  2) Persist fold-wise seeds in the LOOCV results block for provenance.
  3) Document the policy in Methods and this README once implemented.
