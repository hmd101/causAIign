# EA and MV analysis: summaries and plots

This short guide explains how to generate the EA/MV overlay plots (with optional CI whiskers) and how to read the companion summary CSVs and LaTeX snippets.

## What are EA and MV?
- EA (Explaining-Away): positive when evidence E reduces belief in C1 once C2 is known. We plot EA levels per agent and prompt type.
- MV (Markov Violation): magnitude of deviation from the causal-Markov constraint; we plot signed levels, but thresholds use the absolute value.

Default thresholds used in this repo:
- EA passes if EA > 0.3
- MV passes if |MV| ≤ 0.05

Human baseline: Pulled from RW17, numeric, pooled domain when available.

## Create plots
The script overlays numeric and CoT prompts per agent. Optional whiskers show bootstrap CIs (across domains when pooled data are present); the point marks the mean.

- Script: `scripts/plot_ea_mv_levels.py`
- Key flags:
  - `--metric {ea,mv}`
  - `--input-csv <path>` (e.g., results/cross_cogn_strategies/masters_classified_strategy_metrics.csv)
  - `--show-ci` to draw CI whiskers
  - `--bootstrap <B>` number of bootstrap resamples (default 1000)
  - `--ci <level>` percentile CI (default 95)
  - `--seed <int>` RNG seed
  - `--show-human-baseline` to draw the RW17 numeric baseline
  - `--thresholds` to draw default threshold band/line (EA>0.3 or |MV|≤0.05)
  - `--experiments` to select specific experiments
  - `--tag` to filter by tag

Outputs are saved under `results/plots/ea_mv_levels/<metric>/<tag-or-all>/` with both PDF and PNG.

Notes:
- CoT is plotted as dots-only; numeric includes a connecting line and optional whiskers.
- For pooled domain rows (domain==all), whiskers are estimated using the per-domain rows for that agent/category when available; otherwise whiskers are omitted.

## Generate quantitative summaries
Use the summarizer to compute “X% of LLMs pass threshold or exceed human baseline,” at multiple aggregation levels.

- Script: `scripts/summarize_ea_mv_levels.py`
- What it produces:
  - Per experiment + prompt category CSV
  - Per experiment collapsed across prompts (ANY/BOTH) CSV
  - Overall across experiments (ANY/ALL) CSV
  - Optional LaTeX snippets with a compact textual summary
- Key flags:
  - `--metric {ea,mv}` and matching thresholds (`--th_ea`, `--th_mv`)
  - `--input-csv <path>`
  - `--out-prefix <path>` prefix for CSV outputs
  - `--bootstrap <B>`, `--ci <level>`, `--seed <int>` for median/CI columns
  - `--no-human` to skip human comparisons or `--human <value>` to set a fixed baseline

### How to read the CSVs
Each CSV contains counts, percentages, medians, and bootstrap CI columns. Naming conventions:
- n_*: number of agents in the group
- k_*: number of agents satisfying the condition
- pct_*: 100*k/n (percentage of agents satisfying the condition)
- median_metric_*: median of the metric across agents within the group
- median_metric_*_lo / _hi: bootstrap percentile CI bounds for that median

Conditions/groups:
- threshold: EA > th_ea or |MV| ≤ th_mv
- human: metric vs RW17 human baseline (EA > human_EA; |MV| ≤ human_|MV|)
- ANY/BOTH: within an experiment, whether an agent passes in at least one prompt (ANY) or both prompts (BOTH)
- OVERALL ANY/ALL: across experiments, whether an agent passes in at least one experiment (ANY) or in all experiments (ALL)

Special handling for MV:
- Percentages vs threshold/human use |MV|.
- Median and CI columns in the CSVs also use |MV|, so they are comparable to threshold/human conditions.

## Typical runs
- MV with threshold 0.05 and human baseline:
  - Summaries: run the summarizer with `--metric mv --th_mv 0.05 --bootstrap 1000 --ci 95`.
  - Plots: run plot script with `--metric mv --thresholds --show-human-baseline --show-ci`.
- EA with threshold 0.3 and human baseline:
  - Summaries: `--metric ea --th_ea 0.3 --bootstrap 1000 --ci 95`.
  - Plots: `--metric ea --thresholds --show-human-baseline --show-ci`.

Troubleshooting:
- If you see warnings about downcasting in pandas, they’re benign for CSV writing.
- Ensure your input CSV has the expected columns: agent, experiment, prompt_category, domain, EA_raw, MV_raw, tag.
- The human baseline is inferred from RW17 numeric; you can override via `--human`.
