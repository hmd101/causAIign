![MyPackage Logo](assets/logo.png)

# Causal Alignment Experiment Workflow

Complete guide for running causal reasoning experiments with LLMs from prompt generation to visualization.

## Overview

This workflow runs experiments comparing how LLMs and humans reason about causal relationships. The pipeline consists of four main steps:

1. **Prompt Generation** - Create prompts for specific research questions
2. **LLM Experiments** - Run prompts through various LLM models  
3. **Data Processing** - Clean, merge, and analyze LLM + human responses
4. **Visualization** - Generate plots and statistical summaries

## Key Concepts

### What is an "Experiment"?

An **experiment** represents a specific research question with a defined set of prompts and LLM configurations. Examples:
- `pilot_study` - Initial testing of prompt styles
- `temperature_comparison` - Testing different LLM temperature settings
- `confidence_analysis` - Comparing confidence vs certainty prompts

### What is `base_dir` (rw17)?

The `base_dir` parameter refers to the **study folder** containing all prompts related to a specific research paper:
- `rw17` - Prompts for Rehder & Waldmann (2017) causal reasoning tasks
- `future_study` - Could contain prompts for different research questions

### New Directory Structure (v2.0)

The system now uses **mode-specific directory organization** for better separation:

```
data/
├── input_llm/rw17/{experiment_name}/           # Input prompts
├── output_llm/{experiment_name}/{model}/       # Raw LLM responses  
└── processed/                                  # Processed data (NEW!)
    ├── humans/rw17/                           # Human-only data
    ├── llm/rw17/{experiment_name}/            # LLM-only data
    └── llm_with_humans/rw17/{experiment_name}/ # Combined data
```


## Complete Workflow

### **Step 1: Prompt Generation**
```bash
python scripts/generate_experiment_prompts.py --version 8 --experiment pilot_study
```

**Parameters:**
- `--version`: Prompt version number (required)
- `--experiment`: Experiment name (required)
- `--base-dir`: Study folder (default: "data/input_llm/rw17")
- `--output-dir`: Custom output directory (optional)

**Output:** `data/input_llm/rw17/pilot_study/8_v_*.csv`

### **Step 2: LLM Experiments**
```bash
python run_experiment.py --version 8 --experiment pilot_study --model gpt-4o
```

**✅ RECOMMENDED: Auto-discovery method**
- `--version`: Version to process (required)
- `--experiment`: Experiment name (required)  
- `--model`: LLM model (required)
- `--base-dir`: Auto-discovery base (default: "data/input_llm/rw17")
- `--temperature`: Temperature setting (default: 0.0)
- `--runs`: Number of runs (default: 1)

**Legacy method (still supported):**
```bash
python run_experiment.py --dataset path/to/file.csv --model gpt-4o --experiment pilot_study
```

**Output:** `data/output_llm/pilot_study/gpt-4o/8_v_single_numeric_response_gpt-4o_0.0_temp.csv`

### **Step 3: Data Processing**

#### **Combined Mode (Default - LLM + Human):**
```bash
python scripts/run_data_pipeline.py --experiment pilot_study --version 8
```

**Output:** `data/processed/llm_with_humans/rw17/pilot_study/`
- `8_v_collider_cleaned_data.csv` - Main combined dataset
- `reasoning_types/8_v_collider_cleaned_data_roman.csv` - With reasoning annotations
- `8_v_humans_avg_equal_sample_size_cogsci.csv` - Aggregated human responses

#### **LLM-Only Mode:**
```bash
python scripts/run_data_pipeline.py --llm-only --experiment pilot_study --version 8
```

**Output:** `data/processed/llm/rw17/pilot_study/`
- `8_v_collider_llm_only.csv` - LLM responses only
- `reasoning_types/8_v_collider_llm_only_roman.csv` - With reasoning annotations

#### **Human-Only Mode:**
```bash
python scripts/run_data_pipeline.py --human-only --human-file rw17_collider_ce.csv
```

**Output:** `data/processed/humans/rw17/`
- `rw17_collider_humans_processed.csv` - Human responses only

**Parameters:**
- `--experiment`: Experiment name (required for LLM/combined modes)
- `--version`: Version to process (required for LLM/combined modes)
- `--llm-only`: Process only LLM data
- `--human-only`: Process only human data
- `--models`: Specific models to include (optional)
- `--output-dir`: Custom output directory (optional)
- `--graph-type`: Causal graph type (default: "collider")

### **Step 4: Visualization**
```bash
python scripts/plot_results.py --version 8 --experiment pilot_study
```

**Parameters:**
- `--version`: Version to plot (required)
- `--experiment`: Experiment name (required)
- `--pipeline-mode`: Data source ("llm_with_humans", "llm", "humans")
- `--graph-type`: Graph type to plot (default: "collider")  
- `--output-dir`: Custom plot directory (optional)
- `--list-available`: Show all available data files

**Output:** `results/plots/pilot_study/v8_pilot_study_collider_*.pdf/png`

## Pipeline Modes

### **1. Combined Mode (llm_with_humans)**
- **Purpose:** Compare LLM and human responses
- **Use case:** Main analysis, statistical comparisons
- **Data:** Both LLM and human responses merged and aligned
- **Directory:** `data/processed/llm_with_humans/rw17/{experiment}/`

### **2. LLM-Only Mode (llm)**  
- **Purpose:** Analyze LLM responses independently
- **Use case:** Model comparison, response analysis
- **Data:** Only LLM responses (multiple models combined)
- **Directory:** `data/processed/llm/rw17/{experiment}/`

### **3. Human-Only Mode (humans)**
- **Purpose:** Process human baseline data
- **Use case:** Human response analysis, baseline establishment
- **Data:** Only human responses
- **Directory:** `data/processed/humans/rw17/`

## File Naming Conventions

### **Input Files:**
`{version}_v_{prompt_category}_LLM_prompting_{graph_type}.csv`
- Example: `8_v_single_numeric_response_LLM_prompting_collider.csv`

### **Output Files:**  
`{version}_v_{prompt_category}_{model}_{temperature}_temp.csv`
- Example: `8_v_single_numeric_response_gpt-4o_0.0_temp.csv`

### **Processed Files:**
- **Combined:** `{version}_v_{graph_type}_cleaned_data.csv`
- **LLM-only:** `{version}_v_{graph_type}_llm_only.csv` 
- **Human-only:** `rw17_{graph_type}_humans_processed.csv`

**⚠️ Important:** These naming conventions are **required** for downstream processing. The `run_data_pipeline.py` script relies on these patterns for auto-discovery and file matching.

## Creating Experiment Configurations

Experiment configurations are defined in `src/causalign/experiment/config/experiment_config.py`:

```python
EXPERIMENTS = {
    "pilot_study": {
        "description": "Initial prompt testing",
        "prompt_styles": ["single_numeric_response"],
        "domains": ["economy", "sociology", "weather"], 
        "tasks": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"],
        "graph_type": "collider"
    },
    "xml_format_study": {
        "description": "XML response format testing",
        "prompt_styles": ["single_numeric_response_xml"],
        "domains": ["economy", "sociology", "weather"],
        "tasks": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"], 
        "graph_type": "collider"
    }
}
```

**Available prompt styles:**
- `single_numeric_response` - Plain numeric (e.g., "75")
- `single_numeric_response_xml` - XML format (e.g., `<response><likelihood>75</likelihood></response>`)
- ` numeric-conf` - Likelihood + certainty
- `CoT` - Likelihood + confidence + reasoning
- `CoT-brief` - Brief reasoning version
- `CoT-moderate` - Moderate reasoning version

## Checking Available Data

```bash
python scripts/plot_results.py --list-available
```

This shows all processed data files organized by pipeline mode and experiment.

## Common Issues & Solutions

### **File Not Found Errors**
-  Check experiment name matches directory structure
-  Verify version numbers!!!
-  Ensure files follow naming conventions

### **Missing Dependencies**  
-  Activate virtual environment: `source ~/.virtualenvs/llm-causality/bin/activate`
-  Install package: `pip install -e .`

### **Data Inconsistencies**
-  Use `--verbose` flag for detailed logging
-  Check pipeline summary files for processing statistics
-  Verify input file formats match expected schema

## Advanced Usage

### **Custom Output Directories**
```bash
python scripts/run_data_pipeline.py --experiment pilot_study --version 8 --output-dir custom/path/
```

### **Specific Model Selection**
```bash  
python scripts/run_data_pipeline.py --experiment pilot_study --version 8 --models gpt-4o claude-3-opus
```

### **Different Graph Types**
```bash
python scripts/run_data_pipeline.py --experiment pilot_study --version 8 --graph-type fork
```

### **Temperature Comparisons**
```bash
# Generate prompts
python scripts/generate_experiment_prompts.py --version 9 --experiment temp_study

# Run with different temperatures
python run_experiment.py --version 9 --experiment temp_study --model gpt-4o --temperature 0.0
python run_experiment.py --version 9 --experiment temp_study --model gpt-4o --temperature 1.0

# Process data
python scripts/run_data_pipeline.py --experiment temp_study --version 9

# Plot results
python scripts/plot_results.py --version 9 --experiment temp_study
```

## Version History

### **v2.0 (Current)**
- ✅ New mode-specific directory structure
- ✅ Pipeline modes (llm_with_humans, llm, humans)
- ✅ Unified script organization in `scripts/` folder
- ✅ Auto-discovery for input files
- ✅ Consistent parameter naming across scripts

### **v1.0 (Legacy)**
- Mixed data storage in single directories
- Manual file path specification
- Inconsistent parameter naming

---

This workflow ensures reproducible, organized experiments while maintaining flexibility for different research questions and analysis approaches.

## Available Experiments

Run `python scripts/generate_experiment_prompts.py --show-experiments` for current list
or use the new wrapper `python scripts/01_prompts/generate_prompts.py --show-experiments`:

- `pilot_study` - Basic numeric responses on collider tasks
- `reasoning_methods` - Compare reasoning approaches  
- `temperature_study` - Effect of temperature on consistency
- `graph_comparison` - Performance across graph structures
- `xml_format_study` - Structured XML output format
- `domain_transfer` - Reasoning across knowledge domains 

## Structured Model Fitting Artifacts (Refactor Phase 2+)

When using the model fitting pipeline (`causalign.analysis.model_fitting.cli` or API), each fit group now produces:

### Per-Group JSON (`fit_<short_spec_hash>_<short_group_hash>.json`)
Contains:
* spec & group hashes (full + short) enabling reproducible joins
* full list of `restarts` (all restarts always captured; no flag required)
* `best_restart_index` chosen by the configured `selection_rule`
* `metrics` block (loss + optional rmse, mae, r2, aic, bic, ece_10bin, cv metrics)
* `ranking` (primary_metric, selection_rule, restart distribution summary for ranking hints)
* `restart_summary` aggregate statistics across restarts:
    - count
    - loss_mean / loss_median / loss_var
    - duration_mean / duration_sum (if durations captured)
    - primary_metric, primary_mean / primary_median / primary_var (when selected metric is recorded per restart)
* `environment` (python, platform, torch version) & provenance metadata

### Parquet Index (`fit_index.parquet`)
Lightweight searchable table with one row per group. Key columns (in addition to identity):
* restart_count
* restart_loss_mean / restart_loss_median / restart_loss_var
* restart_rmse_mean / restart_rmse_median / restart_rmse_var (if rmse collected)
* restart_aic_mean / restart_aic_var (if aic primary)
* primary_metric (string) and primary_selection_rule

Index aggregation prefers values from `restart_summary` (authoritative) and falls back to on-the-fly computations for backward compatibility.

### Selection Rules
Configurable via `RankingPolicy` or CLI options (planned):
* best_loss – pick restart with minimum final loss
* median_loss – pick restart at median loss (robust representative)
* best_primary_metric – pick restart optimizing the chosen primary_metric (max for r2, min for others)

### Why `restart_summary`?
Facilitates downstream model comparison without loading every JSON, enables statistical filtering (e.g., high variance restarts), and standardizes aggregation semantics in one place.

### Querying Examples (Python API)
```python
from pathlib import Path
from causalign.analysis.model_fitting.api import load_index, rank_index

idx = load_index(Path("path/to/output"))
# Rank by AIC
ranked = rank_index(idx, "aic")
# Filter fits with stable restarts
stable = ranked[ranked.restart_loss_var < 1e-3]
```

### Backward Compatibility
Legacy summary CSV remains temporarily; new workflows should rely on JSON + Parquet. Future phases may remove the CSV once downstream consumers migrate.