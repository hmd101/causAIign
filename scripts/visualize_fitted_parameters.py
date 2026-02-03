#!/usr/bin/env python3
"""
Visualize fitted parameter values across CBN models, agents, and domains.

This script creates comprehensive visualizations of the fitted parameter values
from causal Bayes net model fitting results, allowing comparison across:
- Model types (logistic vs noisy-or)
- Parameter tying schemes (3p, 4p, 5p)
- Agents (humans, LLMs)
- Domains and prompt categories

Usage:
    python scripts/visualize_fitted_parameters.py [options]
"""

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# CColors
# CAblue: RGB (10, 80, 110)
CAblue = (10/255, 80/255, 110/255)       # (0.039, 0.314, 0.431)

# CAlightblue: RGB (58, 160, 171)
CAlightblue = (58/255, 160/255, 171/255) # (0.227, 0.627, 0.671)


def load_combined_results(results_path: Path) -> pd.DataFrame:
    """Load and preprocess the combined model fitting results."""
    df = pd.read_csv(results_path)
    
    # Create model_label for easier grouping
    df['model_label'] = df['model'] + '_' + df['params_tying'].astype(str) + 'p'
    
    # Handle domain NaN -> 'all_domains'
    df['domain_clean'] = df['domain'].fillna('all_domains').astype(str)
    df.loc[df['domain_clean'].str.lower().isin(['nan', 'none', '']), 'domain_clean'] = 'all_domains'
    
    return df

def extract_parameter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and reshape parameter data for visualization."""
    # Define parameter columns for each model type
    logistic_params = ['pC1', 'pC2', 'w0', 'w1', 'w2']
    noisy_or_params = ['pC1', 'pC2', 'b', 'm1', 'm2']
    
    param_rows = []
    
    for _, row in df.iterrows():
        model_type = row['model']
        params = logistic_params if model_type == 'logistic' else noisy_or_params
        
        for param in params:
            fit_col = f'fit_{param}'
            init_col = f'init_{param}'
            
            if fit_col in df.columns and pd.notna(row[fit_col]):
                param_rows.append({
                    'agent': row['agent'],
                    'domain': row['domain_clean'],
                    'prompt_category': row['prompt_category'],
                    'model': row['model'],
                    'params_tying': row['params_tying'],
                    'model_label': row['model_label'],
                    'parameter': param,
                    'fitted_value': row[fit_col],
                    'init_value': row[init_col] if pd.notna(row[init_col]) else None,
                    'loss': row['loss'],
                    'aic': row['aic'],
                    'r2': row['r2'],
                    'experiment': row['experiment'],
                    'version': row['version']
                })
    
    return pd.DataFrame(param_rows)

def create_violin_plots(param_df: pd.DataFrame, output_dir: Path, 
                       filter_agents: Optional[List[str]] = None,
                       filter_domains: Optional[List[str]] = None) -> None:
    """Create faceted violin plots for parameter distributions."""
    
    if filter_agents:
        param_df = param_df[param_df['agent'].isin(filter_agents)]
    if filter_domains:
        param_df = param_df[param_df['domain'].isin(filter_domains)]
    
    # Create separate plots for each parameter
    for param in param_df['parameter'].unique():
        param_data = param_df[param_df['parameter'] == param].copy()
        
        if param_data.empty:
            continue
            
        # Create figure with subplots for each model type
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle(f'Parameter {param} Distribution Across Models and Agents', fontsize=16, y=0.98)
        
        for i, model_type in enumerate(['logistic', 'noisy_or']):
            ax = axes[i]
            model_data = param_data[param_data['model'] == model_type]
            
            if model_data.empty:
                ax.text(0.5, 0.5, f'No data for {model_type}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{model_type.capitalize()} Model')
                continue
            
            # Create violin plot
            sns.violinplot(data=model_data, x='params_tying', y='fitted_value', 
                          hue='agent', ax=ax, inner='quart')
            
            ax.set_title(f'{model_type.capitalize()} Model')
            ax.set_xlabel('Parameter Tying Scheme')
            ax.set_ylabel(f'{param} Value')
            
            # Add summary statistics as text
            for tying in model_data['params_tying'].unique():
                tying_data = model_data[model_data['params_tying'] == tying]
                mean_val = tying_data['fitted_value'].mean()
                std_val = tying_data['fitted_value'].std()
                ax.text(tying-3, ax.get_ylim()[1] * 0.95, 
                       f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                       fontsize=8, ha='center')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'violin_{param}_by_model_agent.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_path_png = output_dir / f'violin_{param}_by_model_agent.png'
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved violin plot for {param}: {plot_path}")

def create_parameter_heatmaps(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps showing parameter values across conditions."""
    
    # Create pivot table for heatmap
    # Rows: agents, Columns: model_label_parameter combinations
    pivot_data = param_df.pivot_table(
        index=['agent', 'domain'], 
        columns=['model_label', 'parameter'],
        values='fitted_value',
        aggfunc='mean'  # Average across prompt_categories if multiple
    )
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                center=0.5, ax=ax, cbar_kws={'label': 'Parameter Value'})
    
    ax.set_title('Fitted Parameter Values Across Models and Conditions', fontsize=16)
    ax.set_xlabel('Model × Parameter', fontsize=12)
    ax.set_ylabel('Agent × Domain', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'heatmap_parameters_all_conditions.pdf'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plot_path_png = output_dir / 'heatmap_parameters_all_conditions.png'
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved parameter heatmap: {plot_path}")

def create_parameter_correlation_plots(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create correlation plots between parameters within each model type."""
    
    for model_type in param_df['model'].unique():
        model_data = param_df[param_df['model'] == model_type]
        
        # Pivot to get parameters as columns
        corr_data = model_data.pivot_table(
            index=['agent', 'domain', 'prompt_category', 'params_tying'],
            columns='parameter',
            values='fitted_value'
        ).reset_index()
        
        # Get parameter columns
        param_cols = [col for col in corr_data.columns if col not in 
                     ['agent', 'domain', 'prompt_category', 'params_tying']]
        
        if len(param_cols) < 2:
            continue
            
        # Create correlation matrix
        corr_matrix = corr_data[param_cols].corr()
        
        # Create figure with subplots for each parameter tying scheme
        n_schemes = len(corr_data['params_tying'].unique())
        fig, axes = plt.subplots(1, n_schemes, figsize=(6*n_schemes, 5))
        if n_schemes == 1:
            axes = [axes]
            
        fig.suptitle(f'Parameter Correlations - {model_type.capitalize()} Model', 
                    fontsize=14, y=1.02)
        
        for i, scheme in enumerate(sorted(corr_data['params_tying'].unique())):
            scheme_data = corr_data[corr_data['params_tying'] == scheme]
            scheme_corr = scheme_data[param_cols].corr()
            
            ax = axes[i]
            sns.heatmap(scheme_corr, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, ax=ax, cbar=i==n_schemes-1)
            ax.set_title(f'{scheme}p Model')
            
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'correlation_{model_type}_parameters.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_path_png = output_dir / f'correlation_{model_type}_parameters.png'
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved correlation plot for {model_type}: {plot_path}")

def create_model_comparison_plots(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create plots comparing same parameters across model types."""
    
    # Get parameters that exist in both models
    logistic_params = set(param_df[param_df['model'] == 'logistic']['parameter'].unique())
    noisy_or_params = set(param_df[param_df['model'] == 'noisy_or']['parameter'].unique())
    common_params = logistic_params.intersection(noisy_or_params)
    
    for param in common_params:
        param_data = param_df[param_df['parameter'] == param]
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Parameter {param} Comparison: Logistic vs Noisy-OR', fontsize=16)
        
        for i, scheme in enumerate([3, 4, 5]):
            ax = axes[i]
            scheme_data = param_data[param_data['params_tying'] == scheme]
            
            if scheme_data.empty:
                continue
                
            # Create scatter plot
            logistic_vals = scheme_data[scheme_data['model'] == 'logistic']['fitted_value']
            noisy_or_vals = scheme_data[scheme_data['model'] == 'noisy_or']['fitted_value']
            
            # Match by agent
            merged = scheme_data.pivot_table(
                index=['agent', 'domain', 'prompt_category'],
                columns='model',
                values='fitted_value'
            ).reset_index()
            
            if 'logistic' in merged.columns and 'noisy_or' in merged.columns:
                clean_data = merged.dropna(subset=['logistic', 'noisy_or'])
                
                if not clean_data.empty:
                    ax.scatter(clean_data['logistic'], clean_data['noisy_or'], 
                              alpha=0.7, s=50)
                    
                    # Add diagonal line
                    min_val = min(clean_data['logistic'].min(), clean_data['noisy_or'].min())
                    max_val = max(clean_data['logistic'].max(), clean_data['noisy_or'].max())
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                    
                    # Calculate correlation
                    corr = clean_data['logistic'].corr(clean_data['noisy_or'])
                    ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes)
            
            ax.set_xlabel(f'Logistic {param}')
            ax.set_ylabel(f'Noisy-OR {param}')
            ax.set_title(f'{scheme}p Model')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / f'comparison_{param}_logistic_vs_noisy_or.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plot_path_png = output_dir / f'comparison_{param}_logistic_vs_noisy_or.png'
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison plot for {param}: {plot_path}")

def create_summary_statistics(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create summary statistics table."""
    
    summary_stats = param_df.groupby(['model', 'params_tying', 'parameter']).agg({
        'fitted_value': ['count', 'mean', 'std', 'min', 'max'],
        'loss': 'mean',
        'aic': 'mean',
        'r2': 'mean'
    }).round(4)
    
    summary_stats.columns = ['n_fits', 'mean_param', 'std_param', 'min_param', 'max_param', 
                           'mean_loss', 'mean_aic', 'mean_r2']
    summary_stats = summary_stats.reset_index()
    
    # Save summary
    summary_path = output_dir / 'parameter_summary_statistics.csv'
    summary_stats.to_csv(summary_path, index=False)
    print(f"Saved summary statistics: {summary_path}")
    
    return summary_stats

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-path', type=Path, 
                       default=Path('results/modelfits/combined.csv'),
                       help='Path to combined results CSV')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/parameter_visualizations'),
                       help='Output directory for plots')
    parser.add_argument('--filter-agents', nargs='*',
                       help='Filter to specific agents')
    parser.add_argument('--filter-domains', nargs='*', 
                       help='Filter to specific domains')
    parser.add_argument('--plot-types', nargs='*',
                       choices=['violin', 'heatmap', 'correlation', 'comparison', 'all'],
                       default=['all'],
                       help='Types of plots to create')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {args.results_path}")
    
    # Load and process data
    df = load_combined_results(args.results_path)
    param_df = extract_parameter_data(df)
    
    print(f"Loaded {len(df)} model fits, extracted {len(param_df)} parameter values")
    print(f"Models: {df['model'].unique()}")
    print(f"Parameter schemes: {sorted(df['params_tying'].unique())}")
    print(f"Parameters: {sorted(param_df['parameter'].unique())}")
    
    # Create summary statistics
    summary_stats = create_summary_statistics(param_df, args.output_dir)
    
    # Create plots based on selection
    plot_types = args.plot_types
    if 'all' in plot_types:
        plot_types = ['violin', 'heatmap', 'correlation', 'comparison']
    
    if 'violin' in plot_types:
        print("\nCreating violin plots...")
        create_violin_plots(param_df, args.output_dir, 
                          args.filter_agents, args.filter_domains)
    
    if 'heatmap' in plot_types:
        print("\nCreating parameter heatmaps...")
        create_parameter_heatmaps(param_df, args.output_dir)
    
    if 'correlation' in plot_types:
        print("\nCreating parameter correlation plots...")
        create_parameter_correlation_plots(param_df, args.output_dir)
    
    if 'comparison' in plot_types:
        print("\nCreating model comparison plots...")
        create_model_comparison_plots(param_df, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")
    print("\nSummary of created files:")
    for file_path in sorted(args.output_dir.glob('*')):
        print(f"  {file_path.name}")

if __name__ == '__main__':
    main()
