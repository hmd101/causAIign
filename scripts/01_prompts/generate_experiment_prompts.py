#!/usr/bin/env python3
"""
Experiment Prompt Generator
=================================

Purpose
-------
Generate LLM prompts for a chosen experiment using the project's unified prompt
generation stack. This replaces ad-hoc notebook runs and provides a consistent
CLI across experiment types (human-data-matched RW17 vs. abstract generators).

This is the new canonical location. A compatibility shim remains at
`scripts/generate_experiment_prompts.py`.

What this script calls/uses
---------------------------
- Experiment config API: src/causalign/experiment/config/experiment_config.py
- PromptFactory: src/causalign/prompts/generators/prompt_factory.py
- Abstract generator: src/causalign/prompts/generators/abstract_generator.py
- PathManager: src/causalign/config/paths.py

Outputs
-------
CSV files per (prompt_style × graph_type) under the selected output directory.

Next steps
----------
1) Review generated CSVs
2) Run LLM experiments with run_experiment.py
3) Process results via scripts/run_data_pipeline.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve().parents[2]

# Ensure project root on sys.path regardless of this file's depth
PROJECT_ROOT = _find_repo_root(Path(__file__).parent)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """Entry point for prompt generation.

    Flow
    ----
    1) Parse CLI and optionally list/describe experiments or styles.
    2) Resolve the experiment via get_experiment_config.
    3) Choose generator: RW17 via PromptFactory, or Abstract via AbstractGenerator.
    4) Generate per (prompt_style × graph_type) and save CSVs.
    """
    # Local imports after sys.path modification
    from src.causalign.experiment.config.experiment_config import (
        get_experiment_config,
        list_available_experiments,
        print_experiment_summary,
    )
    from src.causalign.prompts.generators.prompt_factory import PromptFactory

    parser = argparse.ArgumentParser(
        description="Generate prompts for LLM experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate prompts for pilot study (version 5)
  python scripts/01_prompts/generate_experiment_prompts.py --experiment pilot_study --version 5

  # Generate prompts for graph comparison study
  python scripts/01_prompts/generate_experiment_prompts.py --experiment graph_comparison --version 2

  # List all available experiments
  python scripts/01_prompts/generate_experiment_prompts.py --list-experiments

  # Show detailed experiment descriptions
  python scripts/01_prompts/generate_experiment_prompts.py --show-experiments
        """,
    )

    parser.add_argument("--experiment", "-e", help="Experiment name to generate prompts for")
    parser.add_argument("--version", "-v", default="10", help="Version identifier for file naming (default: 7)")
    parser.add_argument("--output-dir", help="Custom output directory (default: data/input_llm/rw17/{experiment_name}/)")
    parser.add_argument("--list-experiments", action="store_true", help="List all available experiments")
    parser.add_argument("--show-experiments", action="store_true", help="Show detailed experiment descriptions")
    parser.add_argument("--list-styles", action="store_true", help="List all available prompt styles")
    parser.add_argument("--indep-causes-collider", action="store_true", help="Treat causes in collider graphs as independent variables")
    parser.add_argument(
        "--content-category",
        choices=["abstract", "realistic", "overloaded", "unspecified"],
        help="Annotate prompts with a content_category tag (e.g., abstract, realistic, overloaded)",
    )
    # Optional: custom abstract domains via YAML
    parser.add_argument("--use-abstract", action="store_true", help="Force use of the abstract generator (ignores human_data_match)")
    parser.add_argument("--custom-domains-dir", help="Directory containing custom domain YAML files (for abstract generator)")
    parser.add_argument("--custom-domain-files", nargs="+", help="Specific YAML files to load as custom domains (for abstract generator)")
    parser.add_argument("--domains", nargs="+", help="Restrict to these domain names (abstract generator only)")

    args = parser.parse_args()

    # Listing/describe modes
    if args.list_experiments:
        experiments = list_available_experiments()
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp}")
        return

    if args.show_experiments:
        print_experiment_summary()
        return

    if args.list_styles:
        styles = PromptFactory.list_available_styles()
        print("Available prompt styles:")
        for style in styles:
            description = PromptFactory.get_style_description(style)
            print(f"  - {style}: {description}")
        return

    if not args.experiment:
        parser.error("--experiment is required (or use --list-experiments)")

    try:
        # Resolve experiment configuration
        config = get_experiment_config(args.experiment)

        # Human-readable summary
        print(f" Generating prompts for experiment: {config.name}")
        print(f"Description: {config.description}")
        print(f" Prompt styles: {config.prompt_styles}")
        print(f" Graph types: {config.graph_types}")
        print(f" Version: {args.version}")
        if args.indep_causes_collider:
            print(" Independent causes in collider graphs: Enabled")
        if args.content_category:
            print(f"content_category: {args.content_category} (will be added as a new column to all generated prompts)")
        if args.use_abstract or not getattr(config, "human_data_match", False):
            print(" Generator: abstract (custom domains/YAML supported)")
        else:
            print(" Generator: rw17 (human-data-matched)")
        print("-" * 60)

        # Output directory resolution
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            from src.causalign.config.paths import PathManager
            paths = PathManager()
            output_dir = paths.rw17_input_llm_dir / config.name

        # Generator selection: abstract vs RW17
        if args.use_abstract or not getattr(config, "human_data_match", False):
            from src.causalign.prompts.generators.abstract_generator import (
                AbstractGenerator,
                create_custom_abstract_generator,
            )

            overlays_yaml = None
            cs = getattr(config, "custom_settings", None) or {}
            if cs and cs.get("rw17_overlays_file"):
                overlays_yaml = cs["rw17_overlays_file"]

            if args.custom_domain_files or args.custom_domains_dir:
                generator = create_custom_abstract_generator(
                    version=str(args.version),
                    custom_domain_files=args.custom_domain_files,
                    custom_domains_dir=args.custom_domains_dir,
                    output_dir=str(output_dir),
                    overlays_yaml=overlays_yaml,
                )
            else:
                generator = AbstractGenerator(
                    version=str(args.version),
                    output_dir=output_dir,
                    overlays_yaml=overlays_yaml,
                )

            if args.domains:
                generator = generator.with_domains(args.domains)
            generator = generator.with_graph_types(config.graph_types)

            # If overlays YAML present, restrict to clone domains (new_name)
            overlays_domains: set[str] = set()
            if overlays_yaml:
                from pathlib import Path as _P

                import yaml
                overlays_path = _P(overlays_yaml).expanduser()
                if overlays_path.exists():
                    with open(overlays_path, "r") as f:
                        overlays_data = yaml.safe_load(f) or []
                        for ov in overlays_data:
                            nn = ov.get("new_name")
                            if nn:
                                overlays_domains.add(nn)
                else:
                    print(f"⚠️ Overlays file not found: {overlays_yaml}")
            if overlays_domains:
                generator = generator.with_domains(sorted(overlays_domains))

            results: dict[str, list[tuple[pd.DataFrame, Path]]] = {}
            for style_name in config.prompt_styles:
                style = PromptFactory.create_style(style_name)
                style_results = []
                for graph_type in config.graph_types:
                    single_graph_generator = generator.with_graph_types([graph_type])
                    prompts_df, saved_path = single_graph_generator.generate_and_save(
                        style,
                        args.indep_causes_collider,
                        content_category=args.content_category,
                    )
                    style_results.append((prompts_df, saved_path))
                    print(f"Generated {len(prompts_df)} prompts for {style_name} + {graph_type}")
                results[style_name] = style_results
        else:
            # RW17 flow handled by PromptFactory
            results = PromptFactory.generate_experiment_prompts(
                experiment_config=config,
                version=args.version,
                output_dir=output_dir,
                indep_causes_collider=args.indep_causes_collider,
                content_category=args.content_category,
            )

        print("\n✅ Prompt generation complete!")
        print("Generated files:")
        total_prompts = 0
        last_saved_path: Path | None = None
        for style_name, style_results in results.items():
            print(f"\n{style_name}:")
            for prompts_df, saved_path in style_results:
                print(f"   {saved_path} ({len(prompts_df)} prompts)")
                total_prompts += len(prompts_df)
                last_saved_path = saved_path
        print(f"\n Total prompts generated: {total_prompts}")
        out_dir_disp = last_saved_path.parent if last_saved_path is not None else "N/A"
        print(f"💾Files saved to: {out_dir_disp}")

        print("\n Next Steps:")
        print("1. Review generated prompt files")
        print("2. Run LLM experiments using run_experiment.py")
        print("3. Process results using scripts/run_data_pipeline.py")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
