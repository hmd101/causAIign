#!/usr/bin/env python3
"""
Create LLM coverage CSV per experiment by scanning data/output_llm/<experiment>/<agent>/*.csv.

Outputs one CSV per experiment at data/output_llm/llm_coverage_exp_<experiment>.csv

Notes
-----
- Supports GPT-5 variants with verbosity/reasoning-effort suffixes in filenames.
- For non-GPT models, extracts temperature from filename when present.
- Detects basic API error markers in file contents and flags the row.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re


def project_root() -> Path:
    # scripts/03_analysis_raw/<file>.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


CSV_HEADER = [
    "Experiment",
    "prompt_category",
    "version",
    "LLM-agent",
    "temperature",
    "verbosity",
    "reasoning_effort",
    "API-error",
]


def parse_filename(filename: str, agent: str) -> dict:
    # For gpt-5 family
    if agent.startswith("gpt-5"):
        # Example: 2_v_CoT_gpt-5-mini_0.0_vrbsty_low_effort_low_temp.csv
        pattern = r"(?P<version>\d+)_v_(?P<prompt_category>[^_]+)_gpt-5[^_]*_0\.0_vrbsty_(?P<verbosity>[^_]+)_effort_(?P<reasoning_effort>[^_]+)_temp\.csv"
        m = re.match(pattern, filename)
        if m:
            d = m.groupdict()
            d["temperature"] = ""  # ignore temp for gpt-5
            return d
    # For non-gpt-5 models
    # Match both 1_v_CoT_... and 1_CoT_... with experiment name before agent
    pattern = r"(?P<version>\d+)(?:_v)?_(?P<prompt_category>[^_]+)_[^_]+_(?P<agent>.+?)_(?P<temperature>[\d.]+)_temp\.csv"
    m = re.match(pattern, filename)
    if m:
        d = m.groupdict()
        d["verbosity"] = ""
        d["reasoning_effort"] = ""
        return d
    return {}


def file_has_api_error(filepath: Path) -> bool:
    error_patterns = [
        r"Error: Claude API error: Error code: 400",
        r"Error: OpenAI API error: Error code:",
    ]
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                for pat in error_patterns:
                    if re.search(pat, line):
                        return True
    except Exception:
        pass
    return False


def collect_metadata(experiment_name: str, version_filter: str | None = None) -> list[list[str]]:
    out_dir = project_root() / "data" / "output_llm"
    exp_path = out_dir / experiment_name
    if not exp_path.is_dir():
        print(f"Experiment folder not found: {exp_path}")
        return []
    rows: list[list[str]] = []
    for agent in sorted(os.listdir(exp_path)):
        agent_path = exp_path / agent
        if not agent_path.is_dir():
            continue
        for fname in sorted(os.listdir(agent_path)):
            if not fname.endswith(".csv"):
                continue
            if version_filter:
                # Accept files starting with '1_' or '1_v_' for version 1
                if not (fname.startswith(f"{version_filter}_") or fname.startswith(f"{version_filter}_v_")):
                    continue
            meta = parse_filename(fname, agent)
            if not meta:
                continue
            file_path = agent_path / fname
            api_error = "TRUE" if file_has_api_error(file_path) else ""
            allowed = {"CoT", "numeric", "numeric-conf", "single_numeric_response"}
            # Extract prompt_category robustly from filename; prefer first token after version
            m = re.match(r"\d+(?:_v)?_([^_]+)", fname)
            raw_category = m.group(1) if m else meta.get("prompt_category", "")
            prompt_category = raw_category if raw_category in allowed else ""
            rows.append([
                experiment_name,
                prompt_category,
                meta.get("version", ""),
                agent,
                meta.get("temperature", ""),
                meta.get("verbosity", ""),
                meta.get("reasoning_effort", ""),
                api_error,
            ])
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Create LLM coverage CSV for experiment(s)")
    p.add_argument("--experiment", nargs="+", required=True, help="Experiment name(s) in data/output_llm directory")
    p.add_argument("--version", type=str, help="Only process CSV files starting with this version number")
    args = p.parse_args(argv)

    out_root = project_root() / "data" / "output_llm"
    for experiment_name in args.experiment:
        exp_path = out_root / experiment_name
        if not exp_path.is_dir():
            print(f"Error: Experiment folder not found: {exp_path}")
            continue
        rows = collect_metadata(experiment_name, version_filter=args.version)
        if not rows:
            print(f"Warning: No metadata found for experiment: {experiment_name}")
            continue
        out_csv = out_root / f"llm_coverage_exp_{experiment_name}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)
        print(f"CSV created: {out_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
