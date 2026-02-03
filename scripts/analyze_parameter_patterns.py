"""Deprecation shim: use scripts/03_analysis_raw/analyze_parameter_patterns.py"""
from __future__ import annotations

from pathlib import Path
import runpy
import sys

HERE = Path(__file__).resolve()
TARGET = HERE.parent / "03_analysis_raw" / "analyze_parameter_patterns.py"
print("[DEPRECATION] Please call scripts/03_analysis_raw/analyze_parameter_patterns.py; this wrapper will be removed.")
if not TARGET.exists():
    print(f"[ERROR] Canonical script missing: {TARGET}")
    sys.exit(1)
runpy.run_path(str(TARGET), run_name="__main__")
#!/usr/bin/env python3
"""
Focused analysis of parameter patterns across CBN models.

This script provides targeted analyses for specific research questions:
1. How do parameter values differ between logistic and noisy-or models?
2. What's the effect of parameter tying (3p vs 4p vs 5p)?
3. Which agents show the most/least variability in fitted parameters?
4. Are there systematic differences across domains?

Usage:
    python scripts/analyze_parameter_patterns.py [options]
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_and_prep_data(results_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for analysis."""
    df = pd.read_csv(results_path)
    
    # Create model_label for easier grouping
    df['model_label'] = df['model'] + '_' + df['params_tying'].astype(str) + 'p'
    
    # Handle domain NaN -> 'all_domains'
    df['domain_clean'] = df['domain'].fillna('all_domains').astype(str)
    df.loc[df['domain_clean'].str.lower().isin(['nan', 'none', '']), 'domain_clean'] = 'all_domains'
    
    # Extract parameter data
    param_rows = []
    for _, row in df.iterrows():
        model_type = row['model']
        params = ['pC1', 'pC2', 'w0', 'w1', 'w2'] if model_type == 'logistic' else ['pC1', 'pC2', 'b', 'm1', 'm2']
        
        for param in params:
            fit_col = f'fit_{param}'
            if fit_col in df.columns and pd.notna(row[fit_col]):
                param_rows.append({
                    'agent': row['agent'],
                    'domain': row['domain_clean'],
                    'prompt_category': row['prompt_category'],
                    'model': row['model'],
                    'params_tying': row['params_tying'],
                    'model_label': row['model_label'],
                    'parameter': param,
                    'value': row[fit_col],
                    'loss': row['loss'],
                    'aic': row['aic'],
                    'r2': row['r2']
                })
    
    param_df = pd.DataFrame(param_rows)
    return df, param_df

def analyze_model_differences(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze differences between logistic and noisy-or models."""
    
    # Focus on common parameters
    common_params = ['pC1', 'pC2']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison: Logistic vs Noisy-OR', fontsize=16)
    
    for i, param in enumerate(common_params):
        for j, scheme in enumerate([3, 4, 5]):
            ax = axes[i, j]
            
            data = param_df[(param_df['parameter'] == param) & 
                           (param_df['params_tying'] == scheme)]
            
            if data.empty:
                continue
            
            # Box plot comparing models
            sns.boxplot(data=data, x='model', y='value', ax=ax)
            ax.set_title(f'{param} - {scheme}p Model')
            ax.set_ylabel(f'{param} Value')
            
            # Add statistical test result
            logistic_vals = data[data['model'] == 'logistic']['value']
            noisy_or_vals = data[data['model'] == 'noisy_or']['value']
            
            if len(logistic_vals) > 0 and len(noisy_or_vals) > 0:
                from scipy import stats
                stat, p_val = stats.mannwhitneyu(logistic_vals, noisy_or_vals, alternative='two-sided')
                ax.text(0.5, 0.95, f'p = {p_val:.3f}', transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'model_comparison_analysis.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved model comparison: {plot_path}")

def analyze_parameter_tying_effects(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze effects of parameter tying schemes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Parameter Tying Effects', fontsize=16)
    
    # 1. Parameter variability by tying scheme
    ax1 = axes[0, 0]
    variability = param_df.groupby(['model', 'params_tying', 'parameter'])['value'].std().reset_index()
    sns.barplot(data=variability, x='params_tying', y='value', hue='model', ax=ax1)
    ax1.set_title('Parameter Variability by Tying Scheme')
    ax1.set_ylabel('Standard Deviation')
    
    # 2. Model performance by tying scheme
    ax2 = axes[0, 1]
    perf_data = param_df.groupby(['model', 'params_tying']).agg({
        'aic': 'mean', 'r2': 'mean'
    }).reset_index()
    
    ax2_twin = ax2.twinx()
    sns.barplot(data=perf_data, x='params_tying', y='aic', hue='model', ax=ax2, alpha=0.7)
    sns.lineplot(data=perf_data, x='params_tying', y='r2', hue='model', ax=ax2_twin, marker='o')
    ax2.set_title('Model Performance by Tying')
    ax2.set_ylabel('AIC (lower is better)')
    ax2_twin.set_ylabel('R² (higher is better)')
    
    # 3. Causal strength parameters (w1, w2 for logistic; m1, m2 for noisy-or)
    ax3 = axes[1, 0]
    strength_data = param_df[param_df['parameter'].isin(['w1', 'w2', 'm1', 'm2'])]
    if not strength_data.empty:
        sns.violinplot(data=strength_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax3)
        ax3.set_title('Causal Strength Parameters')
        ax3.set_ylabel('Parameter Value')
    
    # 4. Prior parameters (pC1, pC2)
    ax4 = axes[1, 1]
    prior_data = param_df[param_df['parameter'].isin(['pC1', 'pC2'])]
    if not prior_data.empty:
        sns.violinplot(data=prior_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax4)
        ax4.set_title('Prior Parameters')
        ax4.set_ylabel('Parameter Value')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'parameter_tying_analysis.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parameter tying analysis: {plot_path}")

def analyze_agent_differences(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze differences across agents."""
    
    # Calculate parameter variability per agent
    agent_variability = param_df.groupby(['agent', 'parameter']).agg({
        'value': ['mean', 'std', 'count']
    }).round(3)
    agent_variability.columns = ['mean', 'std', 'count']
    agent_variability = agent_variability.reset_index()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Agent-Specific Parameter Patterns', fontsize=16)
    
    # 1. Parameter means by agent
    ax1 = axes[0, 0]
    pivot_means = agent_variability.pivot(index='agent', columns='parameter', values='mean')
    sns.heatmap(pivot_means, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax1)
    ax1.set_title('Mean Parameter Values by Agent')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # 2. Parameter variability by agent
    ax2 = axes[0, 1]
    pivot_stds = agent_variability.pivot(index='agent', columns='parameter', values='std')
    sns.heatmap(pivot_stds, annot=True, fmt='.2f', cmap='Reds', ax=ax2)
    ax2.set_title('Parameter Variability by Agent')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 3. Agent ranking by R²
    ax3 = axes[1, 0]
    agent_perf = param_df.groupby('agent')['r2'].mean().sort_values(ascending=False)
    agent_perf.plot(kind='bar', ax=ax3)
    ax3.set_title('Agent Performance (R²)')
    ax3.set_ylabel('Mean R²')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Human vs LLM comparison
    ax4 = axes[1, 1]
    # Classify agents
    human_agents = ['humans']
    param_df['agent_type'] = param_df['agent'].apply(
        lambda x: 'Human' if x in human_agents else 'LLM'
    )
    
    if 'Human' in param_df['agent_type'].values:
        sns.boxplot(data=param_df, x='agent_type', y='value', hue='parameter', ax=ax4)
        ax4.set_title('Human vs LLM Parameter Values')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'agent_analysis.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed agent statistics
    stats_path = output_dir / 'agent_parameter_statistics.csv'
    agent_variability.to_csv(stats_path, index=False)
    
    print(f"Saved agent analysis: {plot_path}")
    print(f"Saved agent statistics: {stats_path}")

def analyze_domain_effects(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze domain-specific effects."""
    
    # Only analyze if we have multiple domains
    domains = param_df['domain'].unique()
    if len(domains) < 2:
        print("Skipping domain analysis - insufficient domain diversity")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Domain-Specific Parameter Patterns', fontsize=16)
    
    # 1. Parameter means by domain
    ax1 = axes[0, 0]
    domain_means = param_df.groupby(['domain', 'parameter'])['value'].mean().unstack()
    sns.heatmap(domain_means, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax1)
    ax1.set_title('Mean Parameter Values by Domain')
    
    # 2. Domain performance comparison
    ax2 = axes[0, 1]
    domain_perf = param_df.groupby('domain').agg({
        'r2': 'mean',
        'aic': 'mean',
        'loss': 'mean'
    })
    domain_perf['r2'].plot(kind='bar', ax=ax2)
    ax2.set_title('Domain Performance (R²)')
    ax2.set_ylabel('Mean R²')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # 3. Parameter variability by domain
    ax3 = axes[1, 0]
    domain_vars = param_df.groupby(['domain', 'parameter'])['value'].std().unstack()
    sns.heatmap(domain_vars, annot=True, fmt='.2f', cmap='Reds', ax=ax3)
    ax3.set_title('Parameter Variability by Domain')
    
    # 4. Pooled vs individual domain comparison
    ax4 = axes[1, 1]
    pooled_data = param_df[param_df['domain'] == 'all_domains']
    individual_data = param_df[param_df['domain'] != 'all_domains']
    
    if not pooled_data.empty and not individual_data.empty:
        combined = pd.concat([
            pooled_data.assign(fit_type='Pooled'),
            individual_data.assign(fit_type='Individual')
        ])
        sns.boxplot(data=combined, x='fit_type', y='r2', ax=ax4)
        ax4.set_title('Pooled vs Individual Domain Fits')
        ax4.set_ylabel('R²')
    
    plt.tight_layout()
    
    plot_path = output_dir / 'domain_analysis.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved domain analysis: {plot_path}")

def create_executive_summary(df: pd.DataFrame, param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create an executive summary of key findings."""
    
    summary = []
    summary.append("# CBN Parameter Analysis Summary\n")
    
    # Basic statistics
    summary.append(f"## Dataset Overview")
    summary.append(f"- Total model fits: {len(df)}")
    summary.append(f"- Models: {', '.join(df['model'].unique())}")
    summary.append(f"- Parameter schemes: {', '.join(map(str, sorted(df['params_tying'].unique())))}")
    summary.append(f"- Agents: {len(df['agent'].unique())} ({', '.join(sorted(df['agent'].unique())[:5])}{'...' if len(df['agent'].unique()) > 5 else ''})")
    summary.append(f"- Domains: {len(df['domain_clean'].unique())}")
    summary.append("")
    
    # Model performance comparison
    summary.append("## Model Performance")
    perf = df.groupby('model').agg({'r2': 'mean', 'aic': 'mean', 'loss': 'mean'}).round(3)
    for model in perf.index:
        summary.append(f"- {model.capitalize()}: R²={perf.loc[model, 'r2']:.3f}, AIC={perf.loc[model, 'aic']:.1f}")
    summary.append("")
    
    # Parameter tying effects
    summary.append("## Parameter Tying Effects")
    tying_perf = df.groupby('params_tying').agg({'r2': 'mean', 'aic': 'mean'}).round(3)
    for scheme in sorted(tying_perf.index):
        summary.append(f"- {scheme}p model: R²={tying_perf.loc[scheme, 'r2']:.3f}, AIC={tying_perf.loc[scheme, 'aic']:.1f}")
    summary.append("")
    
    # Best performing configurations
    summary.append("## Best Performing Configurations")
    best_r2 = df.loc[df['r2'].idxmax()]
    best_aic = df.loc[df['aic'].idxmin()]
    summary.append(f"- Highest R²: {best_r2['agent']} with {best_r2['model']}_{best_r2['params_tying']}p (R²={best_r2['r2']:.3f})")
    summary.append(f"- Lowest AIC: {best_aic['agent']} with {best_aic['model']}_{best_aic['params_tying']}p (AIC={best_aic['aic']:.1f})")
    summary.append("")
    
    # Parameter insights
    summary.append("## Key Parameter Insights")
    
    # Prior parameters
    prior_stats = param_df[param_df['parameter'].isin(['pC1', 'pC2'])].groupby(['model', 'parameter'])['value'].agg(['mean', 'std']).round(3)
    summary.append("### Prior Parameters (pC1, pC2)")
    for (model, param), stats in prior_stats.iterrows():
        summary.append(f"- {model} {param}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
    summary.append("")
    
    # Causal strength parameters
    strength_params = {'logistic': ['w1', 'w2'], 'noisy_or': ['m1', 'm2']}
    summary.append("### Causal Strength Parameters")
    for model, params in strength_params.items():
        model_data = param_df[(param_df['model'] == model) & (param_df['parameter'].isin(params))]
        if not model_data.empty:
            stats = model_data.groupby('parameter')['value'].agg(['mean', 'std']).round(3)
            for param, stat in stats.iterrows():
                summary.append(f"- {model} {param}: μ={stat['mean']:.3f}, σ={stat['std']:.3f}")
    summary.append("")
    
    # Save summary
    summary_path = output_dir / 'executive_summary.md'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary))
    
    print(f"Saved executive summary: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-path', type=Path, 
                       default=Path('results/modelfits/combined.csv'),
                       help='Path to combined results CSV')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/parameter_analysis'),
                       help='Output directory for analysis results')
    parser.add_argument('--analyses', nargs='*',
                       choices=['models', 'tying', 'agents', 'domains', 'summary', 'all'],
                       default=['all'],
                       help='Types of analyses to run')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {args.results_path}")
    
    # Load and process data
    df, param_df = load_and_prep_data(args.results_path)
    
    print(f"Loaded {len(df)} model fits, extracted {len(param_df)} parameter values")
    
    # Run analyses based on selection
    analyses = args.analyses
    if 'all' in analyses:
        analyses = ['models', 'tying', 'agents', 'domains', 'summary']
    
    if 'models' in analyses:
        print("\nAnalyzing model differences...")
        analyze_model_differences(param_df, args.output_dir)
    
    if 'tying' in analyses:
        print("\nAnalyzing parameter tying effects...")
        analyze_parameter_tying_effects(param_df, args.output_dir)
    
    if 'agents' in analyses:
        print("\nAnalyzing agent differences...")
        analyze_agent_differences(param_df, args.output_dir)
    
    if 'domains' in analyses:
        print("\nAnalyzing domain effects...")
        analyze_domain_effects(param_df, args.output_dir)
    
    if 'summary' in analyses:
        print("\nCreating executive summary...")
        create_executive_summary(df, param_df, args.output_dir)
    
    print(f"\nAll analyses saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
if __name__ == '__main__':
    main()
