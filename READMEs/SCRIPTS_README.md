# CBN Results Scripts

Note: We now provide structured wrappers under `scripts/01_*/...` to guide an A→Z
workflow. See `scripts/README_PIPELINE.md`. Original scripts remain for backward
compatibility and are referenced here as-is.

This repo provides two primary CLI tools for exporting CBN best fits and visualizing fitted parameters.

- export_cbn_best_fits.py — Export best-fitting CBN specs per (agent, domain), write LaTeX tables and winners artifacts.
- visualize_cbn_parameters.py — Visualize the winners’ fitted parameters as heatmaps and write summaries.

Both scripts live under `scripts/` and share a common directory layout for inputs/outputs:

- Winners artifacts: `results/parameter_analysis/<experiment>/<tag>/`
  - `winners.csv`
  - `winners_with_params.csv`
  - `manifest.json` (provenance: filters, versions, models, domains, etc.)
- LaTeX tables: `publication/thesis/tuebingen_thesis_msc/tables/<experiment>/`

## 1) export_cbn_best_fits.py

Selects the best CBN spec per (agent, domain) using a robust winner policy and writes:
- `winners.csv` with metrics and metadata
- `winners_with_params.csv` with canonical CBN parameters (b, m1, m2, pC1, pC2); ties expanded
- `manifest.json` with provenance
- LaTeX table (`cbn_best_fits_<tag>.tex`)

Key arguments:
- `--experiment` / `--experiments` — pick one or more experiments (auto-discovered if omitted)
- `--versions` — version(s) to include; default: latest per experiment
- `--models` — one or both of `logistic`, `noisy_or`
- `--params` — parameter tying counts to include (e.g., `3 4`)
- `--include-domains` — domain scope; pass names (A B C) or `all` for pooled-only
- `--lr` — learning-rate filter (e.g., `0.1`, `lr0p1`, `base`, `all`)
- `--exclude-humans` — drop `agent == humans`
- `--export-params` / `--no-export-params` — toggle `winners_with_params.csv` (on by default)
- `--include-readout-weights` — add `w0,w1,w2` when present
- `--fail-missing-canonical` — error if canonical five cannot be constructed from fit JSON
- `--collapse-prompt` / `--no-collapse-prompt` — collapse across prompt_category or not
- `--tables-dir` — base path for LaTeX outputs
- `--pooled-domain-label` — label used for pooled rows in outputs (default `all`)

Examples:
- Pooled-only, noisy_or v1, lr 0.1, exclude humans:
  ```bash
  python scripts/export_cbn_best_fits.py \
    --experiment random_abstract \
    --versions 1 \
    --models noisy_or \
    --include-domains all \
    --lr 0.1 \
    --params 3 4 \
    --exclude-humans
  ```
- Two domains, logistic v2, include readout weights, fail on missing canonical:
  ```bash
  python scripts/export_cbn_best_fits.py \
    --experiment rw17_indep_causes \
    --versions 2 \
    --models logistic \
    --include-domains domain_A domain_B \
    --params 3 4 \
    --include-readout-weights \
    --fail-missing-canonical
  ```

Outputs are printed at the end of the run.

## 2) visualize_cbn_parameters.py

Reads winners artifacts and visualizes the fitted parameters as heatmaps; also writes summary CSVs.

Key arguments:
- `--experiment` / `--experiments` — pick experiment(s)
- `--tag` — exact tag to visualize (recommended; ensures 1:1 with the table)
- `--tag-glob` — glob to match multiple tags (warns, averages across tags)
- `--agents` — only include listed agents (use `all` for all)
- `--exclude-agents` — exclude agents
- `--domains` — only include listed domains (use `all` for pooled rows)
- `--pooled-only` — keep pooled rows only
- `--out-root` — output directory (default `results/parameter_analysis`)
- `--no-plots` — write CSVs only

Examples:
- Single tag, pooled-only heatmap and summaries:
  ```bash
  python scripts/visualize_cbn_parameters.py \
    --experiment random_abstract \
    --tag v1_noisy_or_pcnum_p3-4_lr0.1_noh \
    --domains all
  ```
- Multiple tags (will warn and average):
  ```bash
  python scripts/visualize_cbn_parameters.py \
    --experiment rw17_indep_causes \
    --tag-glob "v2*noisy_or*" \
    --no-plots
  ```

Outputs:
- Heatmaps: `heatmap_means_<tag>.png/pdf` when a single tag; otherwise `heatmap_means_multi.*`
- CSVs: `winner_parameters_long.csv`, and `by_*.csv` summaries

### About `--tag` vs `--tag-glob` (and why `*` and order matter)

- Prefer `--tag` when you want to visualize exactly one run. This guarantees a 1:1 match with a single tag directory and avoids averaging.
- Use `--tag-glob` to select one or more tag directories by pattern. The value is a glob (not regex) matched against the tag string (i.e., the folder name under `results/parameter_analysis/<experiment>/`).
  - `*` means “any sequence of characters (including empty)”.
  - Order matters: the pattern must follow the order in the actual tag name. Tags are constructed like:
    - `v<versions>_<models?>_pc<prompt-abbrev>_p<param-counts>_lr<lr tokens>_<noh?>`
    - Example tag: `v1_noisy_or_pcnum_p3-4_lr0.1_noh`
  - Therefore, a pattern like `"*noisy_or*lr0.1*"` matches the example, but `"*lr0.1*noisy_or*"` will not (because lr appears after `noisy_or` in the name).
  - Always quote the glob in your shell (e.g., zsh) to prevent the shell from expanding it before the script sees it.
    - TODO: The (glob) tags are very error-prone and need a better data-discovery strategy

Practical examples:

- Match exactly one tag (recommended for publication figures):
  ```bash
  python scripts/visualize_cbn_parameters.py \
    --experiment random_abstract \
    --tag v1_noisy_or_pcnum_p3-4_lr0.1_noh \
    --domains all
  ```

- Match all v1 noisy_or runs at lr 0.1, any param counts in numeric prompts:
  ```bash
  python scripts/visualize_cbn_parameters.py \
    --experiment random_abstract \
    --tag-glob "v1*noisy_or*pcnum*lr0.1*" \
    --domains all
  ```

- Match both models at v2, then average across those tags (script will warn):
  ```bash
  python scripts/visualize_cbn_parameters.py \
    --experiment rw17_indep_causes \
    --tag-glob "v2*pcnum*p3-4*noh" \
    --no-plots
  ```

Tips to get exactly the subfolder configuration you want:
- Inspect the tag names under `results/parameter_analysis/<experiment>/` and compose a glob that matches only those you intend to include.
- Keep token order consistent with the tag structure (e.g., don’t put `lr` before `noisy_or` in the glob if the tag has `noisy_or` before `lr`).
- Quote the glob to keep zsh from expanding it against your current directory.

## Notes
- export_cbn_best_fits writes canonical parameter columns (b, m1, m2, pC1, pC2) by expanding ties and aliasing names across link functions.
- Prefer running visualize_cbn_parameters with an exact `--tag` to avoid cross-tag averaging.
