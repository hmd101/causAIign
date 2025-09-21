#!/usr/bin/env python3
"""
Create separate parameter visualizations for logistic and noisy-OR CBN models.

This script creates dedicated visualizations for each model type instead of 
side-by-side comparisons, allowing for more detailed analysis of each model's
parameter patterns.

Usage:
    python scripts/visualize_parameters_separate_models.py [options]
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_and_prep_data(results_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for analysis."""
    df = pd.read_csv(results_path)
    
    # Create model_label for easier grouping (include learning rate and loss function if not default)
    def create_model_label(row):
        label = f"{row['model']}_{row['params_tying']}p"
        if 'lr' in df.columns and pd.notna(row['lr']) and row['lr'] != 0.1:
            lr_str = f"_lr{row['lr']:g}".replace(".", "p")
            label += lr_str
        if 'loss_function' in df.columns and pd.notna(row['loss_function']) and row['loss_function'] != 'mse':
            label += f"_{row['loss_function']}"
        return label
    
    df['model_label'] = df.apply(create_model_label, axis=1)
    
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
                    'loss_function': row.get('loss_function', 'mse'),  # Default to mse for backward compatibility
                    'lr': row.get('lr', 0.1),  # Default to 0.1 for backward compatibility
                    'aic': row['aic'],
                    'r2': row['r2'],
                    'experiment': row['experiment'],
                    'version': row['version']
                })
    
    param_df = pd.DataFrame(param_rows)
    return df, param_df

def create_logistic_model_dashboard(df: pd.DataFrame, param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive visualization for logistic model."""
    
    # Filter to logistic model only
    df_logistic = df[df['model'] == 'logistic'].copy()
    param_logistic = param_df[param_df['model'] == 'logistic'].copy()
    
    if df_logistic.empty:
        print("No logistic model data found")
        return
    
    # Get metadata for titles and filenames
    experiments = df_logistic['experiment'].unique()
    versions = df_logistic['version'].unique()
    domains = df_logistic['domain_clean'].unique()
    prompt_categories = df_logistic['prompt_category'].unique()
    loss_functions = df_logistic['loss_function'].unique() if 'loss_function' in df_logistic.columns else ['mse']
    
    # Create summary strings for titles and filenames
    exp_str = "_".join(sorted([str(e) for e in experiments]))
    ver_str = "_".join(sorted([str(v) for v in versions]))
    domain_str = "_".join(sorted([str(d) for d in domains if pd.notna(d)]))
    prompt_str = "_".join(sorted([str(p) for p in prompt_categories if pd.notna(p)]))
    loss_str = "_".join(sorted([str(l) for l in loss_functions if pd.notna(l)]))
    
    # Clean up domain string for display
    domain_display = domain_str.replace('all_domains', 'all-domains')
    prompt_display = prompt_str.replace('_', ', ')
    loss_display = loss_str.replace('_', ', ')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    title = f'Logistic CBN Model - Parameter Analysis | Exp: {exp_str}, Ver: {ver_str} | Loss: {loss_display} | Domains: {domain_display} | Prompts: {prompt_display}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Performance by parameter tying scheme
    ax1 = axes[0, 0]
    perf_data = df_logistic.groupby('params_tying').agg({
        'r2': ['mean', 'std'], 
        'aic': ['mean', 'std']
    }).round(3)
    
    x_pos = np.arange(len(perf_data))
    ax1_twin = ax1.twinx()
    
    r2_means = perf_data[('r2', 'mean')]
    r2_stds = perf_data[('r2', 'std')]
    aic_means = perf_data[('aic', 'mean')]
    aic_stds = perf_data[('aic', 'std')]
    
    bars1 = ax1.bar(x_pos - 0.2, r2_means, 0.4, yerr=r2_stds, 
                    label='R²', alpha=0.8, color='skyblue', capsize=5)
    bars2 = ax1_twin.bar(x_pos + 0.2, -aic_means, 0.4, yerr=aic_stds,
                        label='-AIC', alpha=0.8, color='lightcoral', capsize=5)
    
    ax1.set_title('Performance by Parameter Tying')
    ax1.set_xlabel('Parameter Tying Scheme')
    ax1.set_ylabel('R² (higher better)', color='skyblue')
    ax1_twin.set_ylabel('-AIC (higher better)', color='lightcoral')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(idx)}p' for idx in perf_data.index])
    
    # 2. Prior parameters (pC1, pC2) by tying scheme
    ax2 = axes[0, 1]
    prior_data = param_logistic[param_logistic['parameter'].isin(['pC1', 'pC2'])]
    if not prior_data.empty:
        sns.violinplot(data=prior_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax2, palette=['lightblue', 'lightgreen'])
        ax2.set_title('Prior Parameters (pC1, pC2)')
        ax2.set_xlabel('Parameter Tying Scheme')
        ax2.set_ylabel('Prior Probability')
        ax2.legend(title='Parameter')
    
    # 3. Causal strength parameters (w1, w2) by tying scheme
    ax3 = axes[0, 2]
    strength_data = param_logistic[param_logistic['parameter'].isin(['w1', 'w2'])]
    if not strength_data.empty:
        sns.violinplot(data=strength_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax3, palette=['orange', 'red'])
        ax3.set_title('Causal Strength Parameters (w1, w2)')
        ax3.set_xlabel('Parameter Tying Scheme')
        ax3.set_ylabel('Weight Value')
        ax3.legend(title='Parameter')
    
    # 4. Background parameter (w0) by tying scheme
    ax4 = axes[1, 0]
    bg_data = param_logistic[param_logistic['parameter'] == 'w0']
    if not bg_data.empty:
        sns.violinplot(data=bg_data, x='params_tying', y='value', ax=ax4, color='purple')
        ax4.set_title('Background Parameter (w0)')
        ax4.set_xlabel('Parameter Tying Scheme')
        ax4.set_ylabel('Weight Value')
    
    # 5. Parameter correlations
    ax5 = axes[1, 1]
    if not param_logistic.empty:
        pivot_data = param_logistic.pivot_table(
            index=['agent', 'domain', 'params_tying'],
            columns='parameter',
            values='value'
        )
        corr_matrix = pivot_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Parameter Correlations')
    
    # 6. Agent performance ranking
    ax6 = axes[1, 2]
    agent_perf = df_logistic.groupby('agent')['r2'].mean().sort_values(ascending=True)
    if len(agent_perf) > 0:
        # Truncate long agent names for display
        agent_labels = [name[:15] + '...' if len(name) > 15 else name for name in agent_perf.index]
        y_pos = np.arange(len(agent_perf))
        ax6.barh(y_pos, agent_perf.values, color='lightgreen', alpha=0.8)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(agent_labels, fontsize=8)
        ax6.set_xlabel('Mean R²')
        ax6.set_title('Agent Performance Ranking')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'logistic_model_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.pdf'
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    filename_png = f'logistic_model_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.png'
    plot_path_png = output_dir / filename_png
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved logistic model analysis: {plot_path}")

def create_noisy_or_model_dashboard(df: pd.DataFrame, param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create comprehensive visualization for noisy-OR model."""
    
    # Filter to noisy-OR model only
    df_noisy_or = df[df['model'] == 'noisy_or'].copy()
    param_noisy_or = param_df[param_df['model'] == 'noisy_or'].copy()
    
    if df_noisy_or.empty:
        print("No noisy-OR model data found")
        return
    
    # Get metadata for titles and filenames
    experiments = df_noisy_or['experiment'].unique()
    versions = df_noisy_or['version'].unique()
    domains = df_noisy_or['domain_clean'].unique()
    prompt_categories = df_noisy_or['prompt_category'].unique()
    loss_functions = df_noisy_or['loss_function'].unique() if 'loss_function' in df_noisy_or.columns else ['mse']
    
    # Create summary strings for titles and filenames
    exp_str = "_".join(sorted([str(e) for e in experiments]))
    ver_str = "_".join(sorted([str(v) for v in versions]))
    domain_str = "_".join(sorted([str(d) for d in domains if pd.notna(d)]))
    prompt_str = "_".join(sorted([str(p) for p in prompt_categories if pd.notna(p)]))
    loss_str = "_".join(sorted([str(l) for l in loss_functions if pd.notna(l)]))
    
    # Clean up domain string for display
    domain_display = domain_str.replace('all_domains', 'all-domains')
    prompt_display = prompt_str.replace('_', ', ')
    loss_display = loss_str.replace('_', ', ')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    title = f'Noisy-OR CBN Model - Parameter Analysis | Exp: {exp_str}, Ver: {ver_str} | Loss: {loss_display} | Domains: {domain_display} | Prompts: {prompt_display}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # 1. Performance by parameter tying scheme
    ax1 = axes[0, 0]
    perf_data = df_noisy_or.groupby('params_tying').agg({
        'r2': ['mean', 'std'], 
        'aic': ['mean', 'std']
    }).round(3)
    
    x_pos = np.arange(len(perf_data))
    ax1_twin = ax1.twinx()
    
    r2_means = perf_data[('r2', 'mean')]
    r2_stds = perf_data[('r2', 'std')]
    aic_means = perf_data[('aic', 'mean')]
    aic_stds = perf_data[('aic', 'std')]
    
    bars1 = ax1.bar(x_pos - 0.2, r2_means, 0.4, yerr=r2_stds, 
                    label='R²', alpha=0.8, color='skyblue', capsize=5)
    bars2 = ax1_twin.bar(x_pos + 0.2, -aic_means, 0.4, yerr=aic_stds,
                        label='-AIC', alpha=0.8, color='lightcoral', capsize=5)
    
    ax1.set_title('Performance by Parameter Tying')
    ax1.set_xlabel('Parameter Tying Scheme')
    ax1.set_ylabel('R² (higher better)', color='skyblue')
    ax1_twin.set_ylabel('-AIC (higher better)', color='lightcoral')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(idx)}p' for idx in perf_data.index])
    
    # 2. Prior parameters (pC1, pC2) by tying scheme
    ax2 = axes[0, 1]
    prior_data = param_noisy_or[param_noisy_or['parameter'].isin(['pC1', 'pC2'])]
    if not prior_data.empty:
        sns.violinplot(data=prior_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax2, palette=['lightblue', 'lightgreen'])
        ax2.set_title('Prior Parameters (pC1, pC2)')
        ax2.set_xlabel('Parameter Tying Scheme')
        ax2.set_ylabel('Prior Probability')
        ax2.legend(title='Parameter')
    
    # 3. Causal strength parameters (m1, m2) by tying scheme
    ax3 = axes[0, 2]
    strength_data = param_noisy_or[param_noisy_or['parameter'].isin(['m1', 'm2'])]
    if not strength_data.empty:
        sns.violinplot(data=strength_data, x='params_tying', y='value', 
                      hue='parameter', ax=ax3, palette=['orange', 'red'])
        ax3.set_title('Causal Strength Parameters (m1, m2)')
        ax3.set_xlabel('Parameter Tying Scheme')
        ax3.set_ylabel('Strength Probability')
        ax3.legend(title='Parameter')
    
    # 4. Background parameter (b) by tying scheme
    ax4 = axes[1, 0]
    bg_data = param_noisy_or[param_noisy_or['parameter'] == 'b']
    if not bg_data.empty:
        sns.violinplot(data=bg_data, x='params_tying', y='value', ax=ax4, color='purple')
        ax4.set_title('Background Parameter (b)')
        ax4.set_xlabel('Parameter Tying Scheme')
        ax4.set_ylabel('Background Probability')
    
    # 5. Parameter correlations
    ax5 = axes[1, 1]
    if not param_noisy_or.empty:
        pivot_data = param_noisy_or.pivot_table(
            index=['agent', 'domain', 'params_tying'],
            columns='parameter',
            values='value'
        )
        corr_matrix = pivot_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax5, cbar_kws={'shrink': 0.8})
        ax5.set_title('Parameter Correlations')
    
    # 6. Agent performance ranking
    ax6 = axes[1, 2]
    agent_perf = df_noisy_or.groupby('agent')['r2'].mean().sort_values(ascending=True)
    if len(agent_perf) > 0:
        # Truncate long agent names for display
        agent_labels = [name[:15] + '...' if len(name) > 15 else name for name in agent_perf.index]
        y_pos = np.arange(len(agent_perf))
        ax6.barh(y_pos, agent_perf.values, color='lightgreen', alpha=0.8)
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(agent_labels, fontsize=8)
        ax6.set_xlabel('Mean R²')
        ax6.set_title('Agent Performance Ranking')
    
    plt.tight_layout()
    
    # Save plot
    filename = f'noisy_or_model_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.pdf'
    plot_path = output_dir / filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    filename_png = f'noisy_or_model_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.png'
    plot_path_png = output_dir / filename_png
    plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved noisy-OR model analysis: {plot_path}")

def create_individual_parameter_plots(param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create individual plots for each parameter, separated by model."""
    
    # Get unique parameters for each model
    logistic_params = ['pC1', 'pC2', 'w0', 'w1', 'w2']
    noisy_or_params = ['pC1', 'pC2', 'b', 'm1', 'm2']
    
    # Get metadata for titles and filenames
    experiments = param_df['experiment'].unique()
    versions = param_df['version'].unique()
    domains = param_df['domain'].unique()
    prompt_categories = param_df['prompt_category'].unique()
    loss_functions = param_df['loss_function'].unique() if 'loss_function' in param_df.columns else ['mse']
    
    # Create summary strings for titles and filenames
    exp_str = "_".join(sorted([str(e) for e in experiments]))
    ver_str = "_".join(sorted([str(v) for v in versions]))
    domain_str = "_".join(sorted([str(d) for d in domains if pd.notna(d)]))
    prompt_str = "_".join(sorted([str(p) for p in prompt_categories if pd.notna(p)]))
    loss_str = "_".join(sorted([str(l) for l in loss_functions if pd.notna(l)]))
    
    # Clean up domain string for display
    domain_display = domain_str.replace('all_domains', 'all-domains')
    prompt_display = prompt_str.replace('_', ', ')
    loss_display = loss_str.replace('_', ', ')
    
    # Create plots for logistic model parameters
    for param in logistic_params:
        param_data = param_df[(param_df['model'] == 'logistic') & 
                             (param_df['parameter'] == param)]
        
        if param_data.empty:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        title = f'Logistic Model - Parameter {param} | Exp: {exp_str}, Ver: {ver_str} | Loss: {loss_display} | Domains: {domain_display} | Prompts: {prompt_display}'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Violin plot by parameter tying
        ax1 = axes[0]
        sns.violinplot(data=param_data, x='params_tying', y='value', ax=ax1, 
                      color='skyblue', inner='quart')
        ax1.set_title(f'{param} by Parameter Tying')
        ax1.set_xlabel('Parameter Tying Scheme')
        ax1.set_ylabel(f'{param} Value')
        
        # Violin plot by agent with parameter tying indicated
        ax2 = axes[1]
        if not param_data.empty:
            sns.violinplot(data=param_data, x='agent', y='value', hue='params_tying', ax=ax2)
            ax2.set_title(f'{param} by Agent')
            ax2.set_xlabel('Agent')
            ax2.set_ylabel(f'{param} Value')
            ax2.legend(title='Parameter Tying', labels=['3p', '4p', '5p'])
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'logistic_{param}_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.pdf'
        plot_path = output_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        filename_png = f'logistic_{param}_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.png'
        plot_path_png = output_dir / filename_png
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved logistic {param} plot: {plot_path}")
    
    # Create plots for noisy-OR model parameters
    for param in noisy_or_params:
        param_data = param_df[(param_df['model'] == 'noisy_or') & 
                             (param_df['parameter'] == param)]
        
        if param_data.empty:
            continue
            
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        title = f'Noisy-OR Model - Parameter {param} | Exp: {exp_str}, Ver: {ver_str} | Loss: {loss_display} | Domains: {domain_display} | Prompts: {prompt_display}'
        fig.suptitle(title, fontsize=12, fontweight='bold')
        
        # Violin plot by parameter tying
        ax1 = axes[0]
        sns.violinplot(data=param_data, x='params_tying', y='value', ax=ax1, 
                      color='lightcoral', inner='quart')
        ax1.set_title(f'{param} by Parameter Tying')
        ax1.set_xlabel('Parameter Tying Scheme')
        ax1.set_ylabel(f'{param} Value')
        
        # Violin plot by agent with parameter tying indicated
        ax2 = axes[1]
        if not param_data.empty:
            sns.violinplot(data=param_data, x='agent', y='value', hue='params_tying', ax=ax2)
            ax2.set_title(f'{param} by Agent')
            ax2.set_xlabel('Agent')
            ax2.set_ylabel(f'{param} Value')
            ax2.legend(title='Parameter Tying', labels=['3p', '4p', '5p'])
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        filename = f'noisy_or_{param}_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.pdf'
        plot_path = output_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        filename_png = f'noisy_or_{param}_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}_analysis.png'
        plot_path_png = output_dir / filename_png
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved noisy-OR {param} plot: {plot_path}")

def create_model_summary_stats(df: pd.DataFrame, param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create summary statistics for each model separately."""
    
    summaries = {}
    
    for model in ['logistic', 'noisy_or']:
        model_df = df[df['model'] == model]
        model_param_df = param_df[param_df['model'] == model]
        
        if model_df.empty:
            continue
            
        # Performance summary
        perf_summary = model_df.groupby('params_tying').agg({
            'r2': ['count', 'mean', 'std'],
            'aic': ['mean', 'std'],
            'loss': ['mean', 'std']
        }).round(4)
        
        # Parameter summary
        param_summary = model_param_df.groupby(['params_tying', 'parameter']).agg({
            'value': ['count', 'mean', 'std', 'min', 'max']
        }).round(4)
        
        summaries[model] = {
            'performance': perf_summary,
            'parameters': param_summary
        }
        
        # Save individual model summaries
        perf_path = output_dir / f'{model}_performance_summary.csv'
        perf_summary.to_csv(perf_path)
        
        param_path = output_dir / f'{model}_parameter_summary.csv'
        param_summary.to_csv(param_path)
        
        print(f"Saved {model} summaries: {perf_path}, {param_path}")
    
    return summaries

def create_parameter_heatmaps_separate(df: pd.DataFrame, param_df: pd.DataFrame, output_dir: Path) -> None:
    """Create separate parameter heatmaps for logistic and noisy-OR models."""
    
    for model_type in ['logistic', 'noisy_or']:
        model_df = df[df['model'] == model_type].copy()
        model_param_df = param_df[param_df['model'] == model_type].copy()
        
        if model_param_df.empty:
            print(f"No {model_type} model data found for heatmap")
            continue
        
        # Get metadata for titles and filenames
        experiments = model_df['experiment'].unique()
        versions = model_df['version'].unique()
        domains = model_df['domain_clean'].unique()
        prompt_categories = model_df['prompt_category'].unique()
        loss_functions = model_df['loss_function'].unique() if 'loss_function' in model_df.columns else ['mse']
        
        # Create summary strings for titles and filenames
        exp_str = "_".join(sorted([str(e) for e in experiments]))
        ver_str = "_".join(sorted([str(v) for v in versions]))
        domain_str = "_".join(sorted([str(d) for d in domains if pd.notna(d)]))
        prompt_str = "_".join(sorted([str(p) for p in prompt_categories if pd.notna(p)]))
        loss_str = "_".join(sorted([str(l) for l in loss_functions if pd.notna(l)]))
        
        # Clean up domain string for display
        domain_display = domain_str.replace('all_domains', 'all-domains')
        prompt_display = prompt_str.replace('_', ', ')
        loss_display = loss_str.replace('_', ', ')
        
        # Create pivot table for heatmap
        # Rows: agents × domains, Columns: parameter_tying × parameter combinations
        pivot_data = model_param_df.pivot_table(
            index=['agent', 'domain'], 
            columns=['params_tying', 'parameter'],
            values='value',
            aggfunc='mean'  # Average across prompt_categories if multiple
        )
        
        if pivot_data.empty:
            continue
            
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                    center=0.5, ax=ax, cbar_kws={'label': 'Parameter Value'})
        
        # Create title
        title = f'{model_type.replace("_", "-").title()} Model - Parameter Values | Exp: {exp_str}, Ver: {ver_str} | Loss: {loss_display} | Domains: {domain_display} | Prompts: {prompt_display}'
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Parameter Tying × Parameter', fontsize=11)
        ax.set_ylabel('Agent × Domain', fontsize=11)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'heatmap_{model_type}_parameters_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}.pdf'
        plot_path = output_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        filename_png = f'heatmap_{model_type}_parameters_{exp_str}_v{ver_str}_loss-{loss_str}_domains-{domain_str}_prompts-{prompt_str}.png'
        plot_path_png = output_dir / filename_png
        plt.savefig(plot_path_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {model_type} parameter heatmap: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-path', type=Path, 
                       default=Path('results/modelfits/combined.csv'),
                       help='Path to combined results CSV')
    parser.add_argument('--output-dir', type=Path,
                       default=Path('results/parameter_separate_models'),
                       help='Output directory for separate model plots')
    parser.add_argument('--plot-types', nargs='*',
                       choices=['dashboards', 'individual', 'heatmaps', 'summaries', 'all'],
                       default=['all'],
                       help='Types of plots to create')
    parser.add_argument('--filter-prompt-categories', nargs='*',
                       help='Filter to specific prompt categories (e.g., numeric)')
    parser.add_argument('--filter-agents', nargs='*',
                       help='Filter to specific agents')
    parser.add_argument('--filter-domains', nargs='*',
                       help='Filter to specific domains')
    parser.add_argument('--filter-learning-rates', nargs='*', type=float,
                       help='Filter to specific learning rates (e.g., 0.01 0.1 1.0)')
    parser.add_argument('--filter-loss-functions', nargs='*', choices=['mse', 'huber'],
                       help='Filter to specific loss functions (e.g., mse huber)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from: {args.results_path}")
    
    # Load and process data
    df, param_df = load_and_prep_data(args.results_path)
    
    print(f"Loaded {len(df)} model fits, extracted {len(param_df)} parameter values")
    
    # Apply filters if specified
    if args.filter_prompt_categories:
        df = df[df['prompt_category'].isin(args.filter_prompt_categories)]
        param_df = param_df[param_df['prompt_category'].isin(args.filter_prompt_categories)]
        print(f"Filtered to prompt categories: {args.filter_prompt_categories}")
    
    if args.filter_agents:
        df = df[df['agent'].isin(args.filter_agents)]
        param_df = param_df[param_df['agent'].isin(args.filter_agents)]
        print(f"Filtered to agents: {args.filter_agents}")
        
    if args.filter_domains:
        df = df[df['domain_clean'].isin(args.filter_domains)]
        param_df = param_df[param_df['domain'].isin(args.filter_domains)]
        print(f"Filtered to domains: {args.filter_domains}")
    
    if args.filter_learning_rates:
        if 'lr' in df.columns:
            df = df[df['lr'].isin(args.filter_learning_rates)]
            param_df = param_df[param_df['lr'].isin(args.filter_learning_rates)]
            print(f"Filtered to learning rates: {args.filter_learning_rates}")
        else:
            print("Warning: Learning rate column not found in data, skipping LR filter")
    
    if args.filter_loss_functions:
        if 'loss_function' in df.columns:
            df = df[df['loss_function'].isin(args.filter_loss_functions)]
            param_df = param_df[param_df['loss_function'].isin(args.filter_loss_functions)]
            print(f"Filtered to loss functions: {args.filter_loss_functions}")
        else:
            print("Warning: Loss function column not found in data, skipping loss function filter")
    
    print(f"After filtering: {len(df)} model fits, {len(param_df)} parameter values")
    print(f"Models: {df['model'].unique()}")
    print(f"Parameter schemes: {sorted(df['params_tying'].unique())}")
    if 'lr' in df.columns:
        print(f"Learning rates: {sorted(df['lr'].unique())}")
    if 'loss_function' in df.columns:
        print(f"Loss functions: {sorted(df['loss_function'].unique())}")
    
    # Create plots based on selection
    plot_types = args.plot_types
    if 'all' in plot_types:
        plot_types = ['dashboards', 'individual', 'heatmaps', 'summaries']
    
    if 'dashboards' in plot_types:
        print("\nCreating model-specific dashboards...")
        create_logistic_model_dashboard(df, param_df, args.output_dir)
        create_noisy_or_model_dashboard(df, param_df, args.output_dir)
    
    if 'individual' in plot_types:
        print("\nCreating individual parameter plots...")
        create_individual_parameter_plots(param_df, args.output_dir)
    
    if 'heatmaps' in plot_types:
        print("\nCreating separate parameter heatmaps...")
        create_parameter_heatmaps_separate(df, param_df, args.output_dir)
    
    if 'summaries' in plot_types:
        print("\nCreating summary statistics...")
        create_model_summary_stats(df, param_df, args.output_dir)
    
    print(f"\nAll separate model visualizations saved to: {args.output_dir}")
    print("\nCreated files:")
    for file_path in sorted(args.output_dir.glob('*')):
        print(f"  {file_path.name}")

if __name__ == '__main__':
    main()
