![MyPackage Logo](assets/logo.png)

# Data Processing Pipeline Guide

This guide explains the new robust data processing pipeline that replaces the manual Jupyter notebook workflow with an automated, configurable system. This pipeline currently both creates 

## Quick Start

### Command Line Usage
```bash
# Run complete pipeline for pilot study, version 6
python scripts/run_data_pipeline.py --experiment pilot_study --version 6

# Process only specific models
python scripts/run_data_pipeline.py --llm-only --models claude-3-opus gpt-4o

# Process only human data
python scripts/run_data_pipeline.py --human-only

# Run with custom options
python scripts/run_data_pipeline.py --no-aggregate --no-reasoning-types --verbose
```

### Programmatic Usage
```python
from src.causalign.data_handling.processors.pipeline import DataProcessingPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    experiment_name="pilot_study",
    version="6",
    graph_type="collider",
    models=["claude-3-opus", "gpt-4o"],
    add_reasoning_types=True,
    aggregate_human_responses=True
)

# Run pipeline
pipeline = DataProcessingPipeline(config)
result_df = pipeline.run_complete_pipeline()
```

##  Data Structure

The pipeline now works with the current file structure:

```
data/
├── input_llm/rw17/           # LLM input prompts (unchanged)
├── output_llm/               # NEW: LLM responses from API
│   ├── pilot_study/
│   └── temp_test/
├── raw/human/rw17/           # Raw human data (unchanged)
├── processed/                # NEW: All processed outputs
│   ├── human/rw17/           # Processed human data
│   └── llm/17_rw/            # Processed LLM data
└── merged_with_human/rw17/   # Combined datasets
```

## 🔧 Pipeline Components

### 1. LLM Response Processor (`LLMResponseProcessor`)
- **Input**: `data/output_llm/{experiment}/{model}/{temperature}/`
- **Function**: Loads and cleans LLM responses, parses XML/CoT formats
- **Output**: Standardized DataFrames with parsed likelihood/confidence values

### 2. Human Data Processor (`HumanDataProcessor`)
- **Input**: `data/raw/human/rw17/` + `data/input_llm/rw17/` (for ID mapping)
- **Function**: Processes raw human data and assigns matching IDs
- **Output**: Cleaned human responses with proper ID alignment

### 3. Data Validators
- **LLMResponseValidator**: Validates LLM data quality
- **HumanResponseValidator**: Validates human data consistency
- **CombinedDataValidator**: Ensures proper data integration

### 4. Pipeline Orchestrator (`DataProcessingPipeline`)
- **Function**: Coordinates the complete workflow
- **Features**: Configurable, logged, validated, robust error handling

## Configuration Options - Detailed Guide

### PipelineConfig Parameters

Each configuration parameter controls a specific aspect of how data is loaded, processed, and output. Here's what each one means:

```python
config = PipelineConfig(
    experiment_name="pilot_study",           # Where to find LLM output data
    version="6",                            # Which version files to process
    graph_type="collider",                  # Type of causal graph structure
    models=["claude-3-opus", "gpt-4o"],     # Which AI models to include
    add_reasoning_types=True,               # Add cognitive reasoning annotations
    aggregate_human_responses=True          # Average multiple human responses
)
```

#### **📂 1. `experiment_name` (str)**

**Purpose**: Specifies which experiment directory to process  
**Maps to**: `data/output_llm/{experiment_name}/`

**Valid Values**:
- `"pilot_study"` → `data/output_llm/pilot_study/`
- `"temp_test"` → `data/output_llm/temp_test/`
- Any directory name under `data/output_llm/`

**Examples**:
```python
# Process pilot study data
experiment_name="pilot_study"

# Process test data  
experiment_name="temp_test"

# Process a specific experiment batch
experiment_name="experiment_2024_batch_1"
```

#### ** 2. `version` (str)**

**Purpose**: Filters which version of prompt files to process  
**File Filter**: Only processes files starting with `{version}_v_`

**Valid Values**:
- `"6"` → Processes files like `6_v_claude-3-opus_...csv`
- `"5"` → Processes files like `5_v_gpt-4o_...csv`
- `None` → Processes ALL version files

**How it works**:
```python
# With version="6"
✅ 6_v_claude-3-opus-20240229_0.0_temp_6_v_numeric_LLM_prompting_collider.csv.csv
❌ 5_v_claude-3-opus-20240229_0.0_temp_5_v_numeric_LLM_prompting_collider.csv.csv
❌ 4_v_claude-3-opus-20240229_0.0_temp_4_v_numeric-conf_LLM_prompting_collider.csv.csv
```

#### **3. `graph_type` (str)**

**Purpose**: Specifies the causal graph structure being studied  
**Used for**: File naming, human data matching, output organization

**Valid Values**:
- `"collider"` → X → Z ← Y (common effect structure)
- `"fork"` → X ← Z → Y (confounding structure)
- `"chain"` → X → Z → Y (mediation structure)
Note: only collider is fully implemented as of now.

#### ** 4. `models` (list[str] or None)**

**Purpose**: Specifies which AI models to include in processing  
**Default**: `None` (processes ALL available models)

**Valid Values**:
- `None` → Process all models found in experiment directory
- `["claude-3-opus"]` → Process only Claude Opus
- `["claude-3-opus", "gpt-4o"]` → Process Claude and GPT-4o
- `["gemini-1.5-pro", "claude-3-opus", "gpt-4o"]` → Process all three

**Model Name Mapping**:
```python
# Directory names → Clean model names
"claude-3-opus-20240229" → "claude-3-opus"
"gemini-2.0-pro-exp-02-05" → "gemini-2.0-pro"
"gpt-4o" → "gpt-4o"
```

#### ** 5. `add_reasoning_types` (bool)**

**Purpose**: Adds cognitive reasoning type annotations to tasks  
**Default**: `True`

**What it does**:
- Maps letter tasks (`a`, `b`, `c`, ..., `k`) to reasoning categories
- Converts tasks to Roman numerals (`VI`, `VII`, `VIII`, ...)
- Adds `reasoning_type` and `RW17_label` columns

**Reasoning Type Mapping**:
```python
# Effect-Present Diagnostic Inference (explaining away)
"a", "b", "c" → "VI", "VII", "VIII" → "Effect-Present Diagnostic Inference"

# Effect-Absent Diagnostic Inference  
"f", "g", "h" → "IX", "X", "XI" → "Effect-Absent Diagnostic Inference"

# Conditional Independence
"d", "e" → "IV", "V" → "Conditional Independence"

# Predictive Inference
"i", "j", "k" → "I", "II", "III" → "Predictive Inference"
```

**Output Columns Added**:
- `reasoning_type`: Cognitive category
- `RW17_label`: Original letter task (`a`, `b`, `c`, ...)
- `task`: Converted to Roman numerals (`I`, `II`, `III`, ...)

**Examples**:
```python
# Add reasoning annotations (recommended)
add_reasoning_types=True
# Result: task="VI", reasoning_type="Effect-Present Diagnostic Inference", RW17_label="a"

# Keep original task labels
add_reasoning_types=False  
# Result: task="a", no reasoning_type column
```

#### **👥 6. `aggregate_human_responses` (bool)**

**Purpose**: Averages multiple human responses per prompt to balance sample sizes  
**Default**: `True`

**Problem it solves**:
- Humans give multiple responses per prompt ID
- LLMs give one response per prompt ID  
- Unbalanced sample sizes affect statistical comparisons

**Effect on Sample Size**:
```python
# With aggregate_human_responses=True
Human: 5440 responses → 336 responses (averaged by prompt ID)
LLM: 720 responses → 720 responses (unchanged)
Final: 1056 total rows

# With aggregate_human_responses=False  
Human: 5440 responses → 5440 responses (all individual responses)
LLM: 720 responses → 720 responses (unchanged)
Final: 6160 total rows
```

**Examples**:
```python
# Balance sample sizes (recommended for statistical analysis)
aggregate_human_responses=True

# Keep all individual human responses (for response variability analysis)
aggregate_human_responses=False
```

#### ** 7. Additional Parameters**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `human_raw_file` | `"rw17_collider_ce.csv"` | Raw human data filename |
| `prompt_mapping_file` | `"6_v_numeric_LLM_prompting_collider.csv"` | ID mapping file |
| `save_intermediate` | `True` | Save processed outputs |

### **Configuration Examples**

#### **Example 1: Standard Analysis**
```python
config = PipelineConfig(
    experiment_name="pilot_study",      # Use pilot study data
    version="6",                       # Process version 6 prompts only
    graph_type="collider",             # Collider graph structure
    models=None,                       # Include all available models
    add_reasoning_types=True,          # Add cognitive reasoning categories
    aggregate_human_responses=True     # Balance sample sizes
)
# Result: 1056 rows, 4 subjects, Roman numerals, reasoning types
```

#### **Example 2: Model Comparison**
```python
config = PipelineConfig(
    experiment_name="pilot_study",
    version="6", 
    graph_type="collider",
    models=["claude-3-opus", "gpt-4o"],  # Compare only these two models
    add_reasoning_types=False,           # Keep original task labels
    aggregate_human_responses=False      # Keep all human responses
)
# Result: More rows, only 3 subjects (2 LLMs + humans), letter tasks
```

#### **Example 3: Single Model Deep Dive**
```python
config = PipelineConfig(
    experiment_name="temp_test",
    version=None,                      # Process all versions
    graph_type="fork", 
    models=["claude-3-opus"],          # Only Claude
    add_reasoning_types=True,
    aggregate_human_responses=True,
    human_raw_file="rw17_fork_cc.csv" # Different human data file
)
# Result: Claude vs humans comparison across all prompt versions
```

## 🔄Prompt Creation Scenarios

The current pipeline supports two different approaches to creating prompts for LLMs:

### **Scenario 1: Human-Study-Based Prompts (Current Implementation)**

**Use Case**: Creating prompts that match existing human experimental data

**Process**:
1. **Start with human data**: Load `data/raw/human/rw17/rw17_collider_ce.csv`
2. **Extract experimental conditions**: Domain, task, counterbalance condition, graph type
3. **Generate prompts for those conditions**: Create prompts that match the human experimental design
4. **Assign sequential IDs**: Use the notebook logic `range(1, len(df) + 1)`
5. **Save prompt file**: Store in `data/input_llm/rw17/`
6. **Run experiments**: Send prompts to LLMs with their assigned IDs
7. **Process responses**: Use current pipeline to merge LLM and human data by ID

**Supported Graph Types**:
- ✅ **Collider**: `rw17_collider_ce.csv` → works with current pipeline
- ✅ **Fork**: `rw17_fork_cc.csv` → change `human_raw_file` in config
- ✅ **Other topologies**: Add new human data file and update config

### **Scenario 2: Abstract Prompts (Requires Extension)**

**Use Case**: Creating prompts without matching human data (e.g., abstract scenarios, novel conditions)

**Current Status**:  **Partially Supported** - requires modifications

**What Works**:
- ✅ LLM processing: `LLMResponseProcessor` can handle any prompt format
- ✅ ID assignment: Can use same sequential logic
- ✅ Response parsing: XML/CoT parsing works regardless of prompt content

**What Needs Extension**:
- ❌ Human data matching: Current pipeline expects human data to exist
- ❌ ID mapping: No way to create prompts without human experimental conditions

**Implementation Options**:

**Option A: Standalone LLM Analysis**
```python
# Process only LLM data without human matching
pipeline = DataProcessingPipeline(config)
llm_only_data = pipeline.run_llm_only_pipeline()
```

**Option B: Extended Pipeline** (Recommended for future)
```python
# New configuration option
config = PipelineConfig(
    mode="abstract_prompts",        # NEW: Skip human data processing
    prompt_source="generated",      # NEW: Use generated rather than human-matched prompts
)
```

### **Current Pipeline Support Summary**

| Scenario | Support Level | Required Changes |
|----------|---------------|------------------|
| **Collider + Human Data** | ✅ **Fully Supported** | None - works out of box |
| **Fork + Human Data** | ✅ **Fully Supported** | Change config files only |
| **Abstract Prompts** | 🟡 **LLM-Only Mode** | Use `run_llm_only_pipeline()` |

## 🛠 Troubleshooting

### Common Issues

**1. File Not Found Errors**
```bash
# Check if experiment directory exists
ls data/output_llm/pilot_study/

# Verify human data file exists
ls data/raw/human/rw17/rw17_collider_ce.csv
```

**2. ID Mapping Issues**
```bash
# Check prompt mapping file
ls data/input_llm/rw17/6_v_numeric_LLM_prompting_collider.csv

# Run human-only pipeline to debug
python scripts/run_data_pipeline.py --human-only --verbose
```

**3. Processing Errors**
```bash
# Run with verbose logging
python scripts/run_data_pipeline.py --verbose

# Check logs
cat data_pipeline.log
```

### Validation Warnings

The pipeline includes data validators that check:
- Required columns present
- Data types correct
- Value ranges reasonable (likelihood 0-100)
- No duplicate IDs within subjects
- Consistent ID counts across subjects

##  Extending the Pipeline

### Adding New Experiments
1. Put LLM outputs in `data/output_llm/{new_experiment}/`
2. Run: `python scripts/run_data_pipeline.py --experiment {new_experiment}`

### Adding New Models
- No code changes needed
- Pipeline automatically detects new model directories

### Adding New Graph Types
1. Update `PipelineConfig.graph_type`
2. Ensure matching human data file exists
3. Update reasoning type mappings if needed

### Custom Processing
```python
# Use individual processors for custom workflows
from src.causalign.data_handling.processors.llm_processor import LLMResponseProcessor
from src.causalign.data_handling.processors.human_processor import HumanDataProcessor

llm_processor = LLMResponseProcessor()
custom_data = llm_processor.process_experiment("my_experiment", version="7")
```

##  Best Practices

1. **Always run with validation**: Let the validators catch data issues early
2. **Use version control**: Commit processed data with clear version tags
3. **Check logs**: Review processing statistics and warnings
4. **Test incrementally**: Use `--llm-only` or `--human-only` for debugging
5. **Backup original data**: Keep raw data files unchanged
6. **Document configurations**: Use descriptive experiment names and document parameter choices

