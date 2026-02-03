#!/usr/bin/env python3
"""
Fit collider CBN models to processed data.

Usage examples:
    python scripts/fit_models.py --version 8 --experiment pilot_study --agents humans,gpt-4o
    python scripts/fit_models.py --version 8 --experiment pilot_study --model noisy_or --params 4
    python scripts/fit_models.py --input-file path/to/data.csv --tasks VI,VII,VIII --optimizer adam --epochs 500
"""

import sys
from pathlib import Path

# Ensure src on sys.path so we can import the top-level package 'causalign'
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from causalign.analysis.model_fitting.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

 