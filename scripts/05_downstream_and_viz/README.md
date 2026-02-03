# Stage 05 — Downstream summaries and visualizations

This stage contains (publication-ready) plots and cross-experiment summaries that consume outputs from earlier stages.

Included tools (examples use default paths):
- plot_ea_mv_levels.py — Per-agent EA/MV/LOOCV R²/Normativity overlays by experiment
	- python scripts/05_downstream_and_viz/plot_ea_mv_levels.py --metric ea --show_human_baseline --show
- summarize_fit_metric_ranges.py — Summaries from winners.csv (best fits)
	- python scripts/05_downstream_and_viz/summarize_fit_metric_ranges.py --winners results/cross_cogn_strategies/winners.csv
- plot_fit_metric_distributions.py — Distribution plots of fit metrics across experiments/categories
	- python scripts/05_downstream_and_viz/plot_fit_metric_distributions.py --metric loocv_r2 --show

Outputs are written under results/plots/ and results/tables/.
# Downstream analyses and visualizations (step 5)

Scripts aggregating across experiments/tags and visualizing:
- Fit metric distributions
- Cross-experiment tables
- Publication figures

