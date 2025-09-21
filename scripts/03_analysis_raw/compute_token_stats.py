#!/usr/bin/env python3
"""
Compute average token lengths for prompts across input CSVs.

- Scans experiment folders under data/input_llm/rw17/* by default.
- For each CSV, reads the 'prompt' column and computes average token count
  using src.causalign.prompts.core.token_utils.average_token_count.
- Prints a compact report and writes an optional CSV summary.

Usage:
  python scripts/03_analysis_raw/compute_token_stats.py \
    [--root data/input_llm/rw17] [--pattern ""] [--prompt-col prompt] [--out results/token_stats.csv]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Optional

import pandas as pd

# Ensure project root on sys.path so src imports resolve when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.causalign.prompts.core.token_utils import average_token_count
except Exception as e:  # pragma: no cover - environment-specific
    raise RuntimeError(
        "Could not import average_token_count from src.causalign.prompts.core.token_utils.\n"
        "Ensure you're running from the project root and that src/ is available."
    ) from e


def get_csv_files(folder: Path, pattern: str = "") -> list[Path]:
    """Return list of CSV files in folder matching substring pattern in filename."""
    if not folder.exists():
        return []
    return [p for p in folder.glob("*.csv") if pattern in p.name]


def compute_avg_token_length(csv_path: Path, prompt_col: str = "prompt") -> float:
    df = pd.read_csv(csv_path)
    if prompt_col not in df.columns:
        raise ValueError(f"Column '{prompt_col}' not found in {csv_path}")
    prompts = df[prompt_col].dropna().astype(str).tolist()
    return float(average_token_count(prompts))


essential_experiments = {
    "random_abstract": "data/input_llm/rw17/random_abstract",
    "abstract_overloaded_lorem_de": "data/input_llm/rw17/abstract_overloaded_lorem_de",
    "rw17_indep_causes": "data/input_llm/rw17/rw17_indep_causes",
    "rw17_overloaded_lorem_de": "data/input_llm/rw17/rw17_overloaded_lorem_de",
    "rw17_overloaded_de": "data/input_llm/rw17/rw17_overloaded_de",
    "rw17_overloaded_d": "data/input_llm/rw17/rw17_overloaded_d",
    "rw17_overloaded_e": "data/input_llm/rw17/rw17_overloaded_e",
}


def compute_folder_token_stats(folder: Path, pattern: str = "", prompt_col: str = "prompt") -> Dict[str, float | str]:
    csv_files = get_csv_files(folder, pattern)
    stats: Dict[str, float | str] = {}
    for csv_path in csv_files:
        try:
            avg_len = compute_avg_token_length(csv_path, prompt_col)
            stats[str(csv_path)] = avg_len
        except Exception as e:
            stats[str(csv_path)] = f"Error: {e}"
    return stats


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Compute average token lengths for prompt CSVs.")
    p.add_argument("--root", type=str, default="data/input_llm/rw17", help="Root folder that holds experiment subfolders")
    p.add_argument("--pattern", type=str, default="", help="Substring filter for filenames")
    p.add_argument("--prompt-col", type=str, default="prompt", help="Name of the prompt text column")
    p.add_argument("--out", type=str, default="", help="Optional path to write a CSV summary")
    args = p.parse_args(argv)

    root = Path(args.root)
    experiments: dict[str, str] = {}
    if root.exists():
        # Auto-discover immediate subfolders under root
        for p in sorted(root.iterdir()):
            if p.is_dir():
                experiments[p.name] = str(p)
    else:
        # Fall back to curated mapping
        experiments = essential_experiments

    all_rows: list[dict] = []
    for name, folder in experiments.items():
        folder_path = Path(folder)
        stats = compute_folder_token_stats(folder_path, pattern=args.pattern, prompt_col=args.prompt_col)
        print(f"\nExperiment: {name}")
        for csv_path, avg_len in stats.items():
            print(f"{csv_path}: avg token length = {avg_len}")
            all_rows.append({
                "experiment": name,
                "csv": csv_path,
                "avg_tokens": avg_len,
            })

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False)
        print(f"Wrote summary: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
