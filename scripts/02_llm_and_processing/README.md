# Stage 02 — LLM runs and their response processing

This stage covers two distinct steps that must be run separately:
1) Run LLMs on prompts to produce raw outputs under `data/output_llm/`.
2) Process those outputs (and optionally merge with human data) into tidy datasets under `data/processed/`.

Key entry point:
- `run_llm_prompts.py` — a thin wrapper that delegates to one script at a time.
  - It never chains both steps in one call; invoke it twice (once per step).

Examples and Typical usages (from repo root):
- Run an experiment against a model/provider and write raw outputs:
  - python scripts/02_llm_and_processing/run_llm_prompts.py --delegate run_experiment -- --experiment <exp> --version <v> --model <provider>
- Later, process accumulated outputs into `data/processed/` (merging, normalization, metrics):
  - python scripts/02_llm_and_processing/run_llm_prompts.py --delegate pipeline -- --experiment <exp> --version <v>

Notes:
- The `--` separates wrapper flags from the delegated script’s own arguments.
- The processing step scans `data/output_llm/<experiment>/` for the specified version.
- See `READMEs/README_Workflow.md` for configuration, data locations, and troubleshooting.
