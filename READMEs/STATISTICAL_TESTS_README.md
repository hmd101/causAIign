# Statistical tests overview

This note summarizes the hypothesis tests implemented in:
- `scripts/domain_differences.py`
- `scripts/prompt_category_differences.py`

All tests operate on raw likelihood judgments in [0, 100]. Humans are excluded by default unless a flag includes them. GPT‑5 variants are disambiguated via an `agent_variant` label with `-v_<verbosity>-r_<reasoning_effort>` when available.

## Common elements
- Nonparametric tests on untransformed likelihoods (robust to non-normality and ties).
- Effect sizes: rank-biserial correlation from the Mann–Whitney U statistic.
- Distances: 1‑Wasserstein (Earth Mover’s) reported for two-sample contrasts; optional permutation p‑values by shuffling labels within task strata to mitigate task‑mix confounds.
- Multiple testing: Benjamini–Hochberg FDR (BH‑FDR) applied within coherent families; many outputs also include a “global” BH across all rows in that result for convenience.
- Minimum sample size per group (`min_n`) filters ensure adequate data for each test.

## Domain differences (`scripts/domain_differences.py`)
Analyses distribution shifts across RW17 “domains.” Two complementary views:

1) Within‑agent across domains
- Omnibus: Kruskal–Wallis across domains for each non‑human `agent_variant` (only domains with ≥ `min_n`).
- Post‑hoc: pairwise Mann–Whitney U between domains for that agent.
- Corrections:
  - BH‑FDR across agents for omnibus p‑values.
  - BH‑FDR within each agent for post‑hoc pairwise p‑values (and optionally for Wasserstein permutation p‑values).
- Extras: optional 1‑Wasserstein distances (with bootstrap CIs and permutation tests), ECDF overlays per agent by domain.

2) Within‑domain across agents
- Omnibus: Kruskal–Wallis across `agent_variant` within the domain (groups with ≥ `min_n`).
- Post‑hoc: pairwise Mann–Whitney U between agents within the domain.
- Corrections:
  - BH‑FDR across domains for omnibus p‑values.
  - BH‑FDR within each domain for pairwise p‑values (and optional BH on Wasserstein permutation p‑values).
- Extras: optional 1‑Wasserstein metrics, ECDF overlays per domain by agent.

Spike‑composition (optional)
- Three‑bin spike summaries: counts/proportions in [0,5], (5,95), [95,100].
- Pairwise chi‑square tests on 2×3 contingency tables.
- BH‑FDR within families: within‑agent across domains, and within‑domain across agents.
- Note: a task‑stratified permutation alternative is documented to avoid false positives due to differing task mixes.

Outputs (per experiment and prompt filter)
- KW omnibus CSVs and pairwise MWU CSVs for both views.
- Optional spike‑composition summaries and chi‑square results.
- ECDF overlay plots.

## Prompt‑category differences (`scripts/prompt_category_differences.py`)
Answers two questions using original processed data under `data/processed/llm_with_humans/rw17/<experiment>`.

A) Within each experiment, does prompt category (numeric vs CoT) change behavior per agent?
- Test: Mann–Whitney U (two‑sided) comparing numeric vs CoT for each `agent_variant` present in the experiment (both groups require ≥ `min_n`).
- Effect size: rank‑biserial; Distance: 1‑Wasserstein (optional permutation p‑value within task strata).
- Corrections:
  - BH‑FDR within the experiment across agents.
  - Global BH‑FDR across all (experiment, agent) tests.
- Output per experiment: `mwu_numeric_vs_cot_by_agent_v1.csv`; summary table across experiments.

B) Across experiments, does an agent’s behavior differ (optionally per prompt)?
- Omnibus: Kruskal–Wallis across experiments for each (agent_variant, prompt_category), requiring ≥ `min_n` per experiment; BH‑FDR within each prompt and global BH.
- Post‑hoc (optional): If KW is significant at BH‑FDR within‑prompt < α (configurable), run pairwise MWU between experiments for that (agent, prompt). Report U, p, rank‑biserial, 1‑Wasserstein, and optional permutation p (task‑stratified). Apply BH‑FDR within‑family (per agent+prompt across pairs) and global BH across all pairs.
- Outputs: `across_experiments_kw_by_agent_prompt_v1.csv` and `across_experiments_pairwise_by_agent_prompt_v1.csv`.

## Interpreting outputs
- `p_value`: raw test p‑value; `p_fdr_*`: BH‑FDR adjusted p in the specified family (within‑agent/domain/experiment/prompt) or global.
- `effect_rb`: rank‑biserial in [−1,1]; magnitude and sign reflect stochastic dominance directions.
- `W` (Wasserstein): magnitude‑only distributional distance (in likelihood units). `ws_perm_p` is a permutation p‑value that conditions on task strata to reduce confounding from differing task mixes.
- For BH‑FDR thresholds, α=0.05 is used by default. Report both within‑family and global results as appropriate.

## Practical notes
- Humans: excluded by default; can be included via flags.
- Prompts: normalized labels; defaults focus on {numeric, cot}, with `numeric-conf` optionally included.
- Agent variants: GPT‑5 variants are split by verbosity/reasoning effort to avoid conflating settings.
- Minimum n: adjust `min_n` to reflect reliability; very small samples may switch MWU to exact computation.

## Typical usage
- Domain differences: compare distributions across domains for a model and across models within a domain; optionally analyze spikes and generate ECDF plots.
- Prompt differences: assess numeric vs CoT within each experiment per agent; test whether an agent changes across experiments (per prompt), with post‑hoc pairs when warranted.

For detailed CLI options and generated paths, see the docstrings and argparse help in each script.
