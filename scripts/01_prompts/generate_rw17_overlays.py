#!/usr/bin/env python3
"""
Generate systematic RW17 overlays with a compact, decipherable naming scheme.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

import yaml

BASE_DOMAINS = ["economy", "sociology", "weather"]

CONFIG = {
    "economy": {
        "intro_from": [["sociology"], ["weather"]],
        "vars": {
            "X": {"detailed_from": [["sociology"]], "explanations_from": [["sociology"], ["weather"]]},
            "Y": {"detailed_from": [["weather"]], "explanations_from": [["economy"]]},
            "Z": {"detailed_from": [["sociology"]], "explanations_from": [["sociology"]]},
        },
    },
    "sociology": {
        "intro_from": [["economy"], ["weather"]],
        "vars": {
            "X": {"detailed_from": [["economy"]], "explanations_from": [["economy"], ["weather"]]},
            "Y": {"detailed_from": [["weather"]], "explanations_from": [["sociology"]]},
            "Z": {"detailed_from": [["economy"]], "explanations_from": [["weather"]]},
        },
    },
    "weather": {
        "intro_from": [["economy"], ["sociology"]],
        "vars": {
            "X": {"detailed_from": [["economy"]], "explanations_from": [["sociology"]]},
            "Y": {"detailed_from": [["sociology"]], "explanations_from": [["economy"]]},
            "Z": {"detailed_from": [["economy"]], "explanations_from": [["economy"], ["sociology"]]},
        },
    },
}

VAR_EXPL_CODES = {"X": "xe", "Y": "ye", "Z": "ze"}
VAR_DET_CODES = {"X": "xd", "Y": "yd", "Z": "zd"}
DOMAIN_CODES = {"economy": "econ", "sociology": "soc", "weather": "weath"}


def _src_token(sources: list[str]) -> str:
    codes = [DOMAIN_CODES.get(s, s[:3]) for s in sources]
    return "+".join(codes)


def build_name(base: str, spec: dict) -> str:
    tokens = []
    for var in ("X", "Y", "Z"):
        varspec = spec.get(var, {}) or {}
        det_sources = varspec.get("detailed") or []
        if det_sources:
            tokens.append(f"{VAR_DET_CODES[var]}={_src_token(det_sources)}")
        expl_sources = varspec.get("explanations") or []
        if expl_sources:
            tokens.append(f"{VAR_EXPL_CODES[var]}={_src_token(expl_sources)}")
    suffix = ("_" + "_".join(tokens)) if tokens else ""
    return f"{DOMAIN_CODES.get(base, base)}_overloaded{suffix}"


def build_overlay(base: str, spec: dict) -> dict:
    ov: Dict[str, Any] = {"domain": base}
    ov_vars = {}

    if spec.get("intro"):
        ov["introduction_append_from"] = spec["intro"]

    for var in ("X", "Y", "Z"):
        v = spec.get(var, {})
        if not v:
            continue
        ov_vars[var] = {}
        if v.get("detailed"):
            ov_vars[var]["detailed_append_from"] = v["detailed"]
        if v.get("explanations"):
            ov_vars[var]["explanations_append_from"] = v["explanations"]

    if ov_vars:
        ov["variables"] = ov_vars

    ov["new_name"] = build_name(base, spec)
    return ov


def enumerate_specs(base: str, cfg: dict) -> list[dict]:
    groups_intro = cfg.get("intro_from", [])
    intro_choices = [g for g in groups_intro] or [[]]

    var_specs = {}
    for var in ("X", "Y", "Z"):
        vcfg = cfg.get("vars", {}).get(var, {})
        var_specs[var] = {
            "detailed": [g for g in vcfg.get("detailed_from", [])] or [[]],
            "explanations": [g for g in vcfg.get("explanations_from", [])] or [[]],
        }

    overlays = []
    for intro_sel in intro_choices:
        for xd_sel in var_specs["X"]["detailed"]:
            for xe_sel in var_specs["X"]["explanations"]:
                for yd_sel in var_specs["Y"]["detailed"]:
                    for ye_sel in var_specs["Y"]["explanations"]:
                        for zd_sel in var_specs["Z"]["detailed"]:
                            for ze_sel in var_specs["Z"]["explanations"]:
                                spec = {
                                    "intro": intro_sel,
                                    "X": {"detailed": xd_sel, "explanations": xe_sel},
                                    "Y": {"detailed": yd_sel, "explanations": ye_sel},
                                    "Z": {"detailed": zd_sel, "explanations": ze_sel},
                                }
                                overlays.append(build_overlay(base, spec))
    return overlays


def main(out_path: Path) -> None:
    all_overlays = []
    for base in BASE_DOMAINS:
        cfg = CONFIG.get(base, {})
        if not cfg:
            continue
        all_overlays.extend(enumerate_specs(base, cfg))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(all_overlays, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {len(all_overlays)} overlays -> {out_path}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("src/causalign/prompts/custom_domains/rw17_overlays_generated.yaml")
    main(out)
    main(out)
