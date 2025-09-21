#!/usr/bin/env python3
"""
Build six standard RW17 overlay YAMLs:
1) Append explanations only (e)
2) Append detailed + explanations (de)
3) Append detailed only (d)
4) Control explanations only (filler e)
5) Control detailed + explanations (filler de)
6) Control detailed only (filler d)

Files are written to src/causalign/prompts/custom_domains/ with descriptive names.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List

import yaml


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve()
    for p in here.parents:
        candidate = p / "src"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            break


# Ensure src on path for type imports if needed (no direct imports used below)
_ensure_src_on_path()

BASES: List[str] = ["economy", "sociology", "weather"]
OUT_DIR = Path("src/causalign/prompts/custom_domains")

# Naming helpers
APPEND_FILES = {
    "e": "rw17_overlays_append_e.yaml",
    "de": "rw17_overlays_append_de.yaml",
    "d": "rw17_overlays_append_d.yaml",
}
CONTROL_FILES = {
    "e": "rw17_overlays_control_e.yaml",
    "de": "rw17_overlays_control_de.yaml",
    "d": "rw17_overlays_control_d.yaml",
}

# Short codes for names
DC = {"economy": "econ", "sociology": "soc", "weather": "weath"}


def default_src_for(base: str) -> str:
    # Deterministic other domain choice: first different in fixed order
    order = ["economy", "sociology", "weather"]
    for d in order:
        if d != base:
            return d
    return base


def new_name_for(base: str, fields: str, mode: str, control: bool, src: str | None = None) -> str:
    # Use a single kind for append (ovl) and replace (ovlRep). Control adds a =ctl suffix.
    kind = "ovl" if mode == "append" else "ovlRep"
    parts = []
    if "d" in fields:
        parts.append("d")
    if "e" in fields:
        parts.append("e")
    suffix = "".join(parts)
    name = f"{DC[base]}_{kind}_{suffix}"
    # Include source domain for non-control; use ctl token for control
    if control:
        name = f"{name}=ctl"
    elif src:
        name = f"{name}={DC[src]}"
    return name


def build_set(
    fields: str,
    mode: str = "append",
    control: bool = False,
    filler_words_e: int = 30,
    filler_words_d: int = 20,
) -> List[Dict[str, Any]]:
    overlays: List[Dict[str, Any]] = []
    for base in BASES:
        src = default_src_for(base)
        ov: Dict[str, Any] = {"domain": base}
        # Variables block for X/Y/Z
        vars_block: Dict[str, Dict[str, Any]] = {}
        for v in ("X", "Y", "Z"):
            entry: Dict[str, Any] = {}
            if control:
                # Neutral filler to match token budget
                if "d" in fields:
                    entry["detailed_filler_words"] = int(filler_words_d)
                if "e" in fields:
                    entry["explanations_filler_words"] = int(filler_words_e)
            else:
                # Append from another domain
                if "d" in fields:
                    entry["detailed_append_from"] = [src]
                if "e" in fields:
                    entry["explanations_append_from"] = [src]
            if entry:
                vars_block[v] = entry
        if vars_block:
            ov["variables"] = vars_block
        ov["new_name"] = new_name_for(base, fields, mode, control, src=src)
        overlays.append(ov)
    return overlays


def write_yaml(filename: str, overlays: List[Dict[str, Any]]):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / filename
    with path.open("w") as f:
        yaml.safe_dump(overlays, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {len(overlays)} overlays -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Build standard RW17 overlay YAMLs")
    parser.add_argument("--with_control", action="store_true", help="Include control overlays inside each append YAML as well")
    parser.add_argument("--control-e-words", type=int, default=30, dest="control_e_words", help="Control filler words for explanations (e)")
    parser.add_argument("--control-d-words", type=int, default=20, dest="control_d_words", help="Control filler words for detailed (d)")
    args = parser.parse_args()

    # Build append sets
    append_e = build_set("e", mode="append", control=False)
    append_de = build_set("de", mode="append", control=False)
    append_d = build_set("d", mode="append", control=False)

    # Optionally build control overlays to include in append files
    if args.with_control:
        ctl_e = build_set("e", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words)
        ctl_de = build_set("de", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words)
        ctl_d = build_set("d", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words)
        append_e = append_e + ctl_e
        append_de = append_de + ctl_de
        append_d = append_d + ctl_d

    # Write append YAMLs (with or without control overlays inline)
    write_yaml(APPEND_FILES["e"], append_e)
    write_yaml(APPEND_FILES["de"], append_de)
    write_yaml(APPEND_FILES["d"], append_d)

    # Always also write standalone control variants (filler-only)
    write_yaml(CONTROL_FILES["e"], build_set("e", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words))
    write_yaml(CONTROL_FILES["de"], build_set("de", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words))
    write_yaml(CONTROL_FILES["d"], build_set("d", mode="append", control=True, filler_words_e=args.control_e_words, filler_words_d=args.control_d_words))


if __name__ == "__main__":
    main()
    main()
