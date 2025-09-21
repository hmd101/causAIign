#!/usr/bin/env python3
"""
Custom Domain Creator

User-friendly script for creating and managing custom abstract domains.
Allows users to define their own variables and create custom experiments.

Usage:
    python scripts/create_custom_domain.py --create-template my_domain
    python scripts/create_custom_domain.py --generate my_domain.yaml --version 9
    python scripts/create_custom_domain.py --list-domains
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.causalign.prompts.custom_domains.domain_loader import CustomDomainLoader
from src.causalign.prompts.generators.abstract_generator import create_custom_abstract_generator
from src.causalign.prompts.styles import NumericOnlyStyle, ConfidenceStyle, ChainOfThoughtStyle


def main():
    parser = argparse.ArgumentParser(
        description="Create and manage custom abstract domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a template for a new domain
  python scripts/create_custom_domain.py --create-template robotics
  
  # Generate prompts using a custom domain
  python scripts/create_custom_domain.py --generate my_domain.yaml --version 9
  
  # Generate prompts using multiple custom domains
  python scripts/create_custom_domain.py --generate robotics.yaml biology.yaml --version 9
  
  # List all available custom domains
  python scripts/create_custom_domain.py --list-domains
  
  # Validate a custom domain file
  python scripts/create_custom_domain.py --validate my_domain.yaml
        """,
    )

    parser.add_argument(
        "--create-template",
        help="Create a template YAML file for a new domain"
    )
    
    parser.add_argument(
        "--generate",
        nargs="+",
        help="Generate prompts using custom domain YAML file(s)"
    )
    
    parser.add_argument(
        "--version",
        default="9",
        help="Version identifier for generated prompts (default: 9)"
    )
    
    parser.add_argument(
        "--style",
        choices=["numeric", "confidence", "cot"],
        default="numeric",
        help="Prompt style (default: numeric)"
    )
    
    parser.add_argument(
        "--graph-type",
        choices=["collider", "fork", "chain"],
        default="collider", 
        help="Graph type (default: collider)"
    )
    
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="List all available custom domains"
    )
    
    parser.add_argument(
        "--validate",
        help="Validate a custom domain YAML file"
    )
    
    parser.add_argument(
        "--custom-domains-dir",
        help="Custom directory containing domain YAML files"
    )
    
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated prompts"
    )

    args = parser.parse_args()

    try:
        # Initialize domain loader
        custom_dir = Path(args.custom_domains_dir) if args.custom_domains_dir else None
        loader = CustomDomainLoader(custom_dir)

        if args.create_template:
            # Create domain template
            domain_name = args.create_template
            output_file = f"{domain_name}.yaml"
            
            template_path = loader.create_domain_template(domain_name, output_file)
            print(f"✅ Created domain template: {template_path}")
            print(f"📝 Edit {template_path} to define your custom domain")
            print(f"🚀 Then run: python scripts/create_custom_domain.py --generate {output_file} --version {args.version}")

        elif args.list_domains:
            # List available domains
            predefined_domains = ["abstract_math", "fantasy", "symbolic"]
            custom_domains = loader.list_available_domains()
            
            print("📊 Available Domains:")
            print("\n🔧 Predefined Abstract Domains:")
            for domain in predefined_domains:
                print(f"  - {domain}")
                
            if custom_domains:
                print(f"\n🎨 Custom Domains:")
                for domain in custom_domains:
                    print(f"  - {domain}")
            else:
                print(f"\n🎨 Custom Domains: None found")
                print(f"�� Create custom domains with: --create-template my_domain")

        elif args.validate:
            # Validate domain file
            domain_file = args.validate
            try:
                domain_dict = loader.load_domain_from_yaml(domain_file)
                domain_name = domain_dict["domain_name"]
                variables = list(domain_dict["variables"].keys())
                print(f"✅ Domain file '{domain_file}' is valid")
                print(f"   Domain: {domain_name}")
                print(f"   Variables: {variables}")
            except Exception as e:
                print(f"❌ Domain file '{domain_file}' is invalid: {e}")
                sys.exit(1)

        elif args.generate:
            # Generate prompts using custom domains
            domain_files = args.generate
            
            print(f"🔬 Generating prompts from custom domains: {domain_files}")
            print(f"🔢 Version: {args.version}")
            print(f"🎨 Style: {args.style}")
            print(f"📊 Graph: {args.graph_type}")
            print("-" * 60)
            
            # Create style object
            style_map = {
                "numeric": NumericOnlyStyle(),
                "confidence": ConfidenceStyle(),
                "cot": ChainOfThoughtStyle()
            }
            style = style_map[args.style]
            
            # Create custom generator
            generator = create_custom_abstract_generator(
                version=args.version,
                custom_domain_files=domain_files,
                custom_domains_dir=args.custom_domains_dir,
                output_dir=args.output_dir
            )
            
            # Configure graph types
            generator = generator.with_graph_types([args.graph_type])
            
            # Generate and save prompts
            prompts_df, saved_path = generator.generate_and_save(style)
            
            print(f"\n✅ Prompt generation complete!")
            print(f"📁 Generated file: {saved_path}")
            print(f"📊 Total prompts: {len(prompts_df)}")
            
            # Show domains used
            domains_used = prompts_df['domain'].unique()
            print(f"🎨 Domains used: {list(domains_used)}")
            
            print(f"\n🚀 Next Steps:")
            print(f"1. Review generated prompts: {saved_path}")
            print(f"2. Run LLM experiments: python run_experiment.py --dataset {saved_path} --model gpt-4o")

        else:
            parser.print_help()

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
