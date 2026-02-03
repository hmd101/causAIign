# Pipeline overview (A→Z)

This package supports a full workflow from prompt generation for LLMs, comparing responses with humans,  to causal Bayes net fitting to an agent's set of responses and downstream analysis.
Below is a structure and where to find the tools

## Folder layout (proposed)
- `01_prompts/` — prompt creation
  - `generate_prompts.py` (wrapper for `scripts/generate_experiment_prompts.py`)
   - `build_standard_overlays.py` (produce canned RW17 overlay YAMLs)
   - `generate_rw17_overlays.py` (systematic spec → overlays YAML)
   - `generate_filler_blocks.py` and `generate_overlay_fillers.py` (token-matched neutral fillers)
   - `generate_random_names.py` (utility for symbolic variable names)
- `02_llm_and_processing/` — run LLMs and process outputs
   - `run_llm_prompts.py` (delegates to `run_experiment.py` or `scripts/run_data_pipeline.py`, one at a time)
- `03_analysis_raw/` — analyses on processed (pre-CBN) data
  - correlations, domain differences, coverage
- `04_cbn_fit_and_eval/` — fit CBNs and export best fits
- `05_downstream_and_viz/` — cross-experiment summaries and plots

During the current refactoring , all  existing scripts remain at their original locations for backward compatibility.
Wrappers in the folders above help new users navigate the workflow.

## Typical A→Z workflow
1. Generate prompts
   - `python scripts/01_prompts/generate_prompts.py --experiment <exp> --version <v>`
2. Run LLMs on prompts (Step 1)
   - `python scripts/02_llm_and_processing/run_llm_prompts.py --delegate run_experiment -- --version <v> --experiment <exp> --model <provider>`
3. Process raw outputs into tidy datasets (Step 2, separate call)
   - `python scripts/02_llm_and_processing/run_llm_prompts.py --delegate pipeline -- --experiment <exp> --version <v>`
4. Analyze raw data
   - domain differences, human–LLM correlations, etc. (see `03_analysis_raw/` and READMEs)
5. Fit CBNs and export winners
   - use `scripts/export_cbn_best_fits.py` and related tools (see `04_cbn_fit_and_eval/`)
6. Downstream summaries and visualizations
   - e.g., metric distributions, cross-experiment tables, EA/MV/LOOCV R² overlays (see `05_downstream_and_viz/` — `plot_ea_mv_levels.py`)

