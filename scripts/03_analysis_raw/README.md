# Stage 03 — Analyses on processed (pre-CBN) data

Canonical scripts in this stage:
- summarize_ea_mv_levels.py — Percent of agents meeting EA/MV thresholds and human baseline; CSV + LaTeX
	- python scripts/03_analysis_raw/summarize_ea_mv_levels.py --metric ea
- human_llm_alignment_correlation.py — Spearman alignment vs humans with bootstrapped CIs; plots + LaTeX
- prompt_category_differences.py — MWU (always), KW omnibus, BH–FDR corrections, rank-biserial effect sizes
- aggregate_cognitive_strategies.py — Strategy metrics aggregation and plots
- compute_token_stats.py — Token length stats for datasets
- create_llm_coverage_csv.py — LLM output coverage manifest
- validate_task_probs.py — Sanity-check task probabilities used in analyses
- analyze_parameter_patterns.py — Focused analyses of fitted parameter patterns

See scripts/README_PIPELINE.md for the A→Z overview and stage boundaries.
# Raw data analysis (step 3)

This folder groups scripts that operate on raw, processed datasets (pre-CBN):
- Correlations (human–LLM alignment)
- Domain/prompt differences
- Token statistics, prompt coverage, etc.

Use alongside the pipeline wrapper in `../02_llm_and_processing/`.
