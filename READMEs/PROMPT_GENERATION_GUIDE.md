![MyPackage Logo](assets/logo.png)

# Prompt Generation Guide

Complete guide for creating prompts for causal reasoning experiments with LLMs.

## Overview

The prompt generation system uses a **Factory Pattern** to create prompts for different experiment types. It supports both human-study-based prompts (matching existing experimental data) and abstract prompts (for pure reasoning experiments).

## Architecture

```
PromptFactory
├── Generators
│   ├── RW17Generator (Human-study-based)
│   └── AbstractGenerator (Abstract/Fantasy domains)
└── Styles
    ├── NumericOnlyStyle
    ├── ConfidenceStyle
    ├── ChainOfThoughtStyle
    └── XML formats
```

## Quick Start

### 1. Generate Prompts for an Experiment

```bash
# Generate prompts for pilot study
python scripts/generate_experiment_prompts.py --experiment pilot_study --version 7
# or via wrapper
python scripts/01_prompts/generate_prompts.py --experiment pilot_study --version 7

# Generate prompts for abstract reasoning
python scripts/generate_experiment_prompts.py --experiment abstract_reasoning --version 9

# Generate prompts with independent causes in collider graphs
python scripts/generate_experiment_prompts.py --experiment pilot_study --version 7 --indep-causes-collider

# List available experiments
python scripts/generate_experiment_prompts.py --list-experiments

# Show experiment details
python scripts/generate_experiment_prompts.py --show-experiments
```

### 2. Available Experiments

Run `python scripts/generate_experiment_prompts.py --show-experiments` for current list:

- `pilot_study` - Basic numeric responses on collider tasks
- `reasoning_methods` - Compare reasoning approaches  
- `temperature_study` - Effect of temperature on consistency
- `graph_comparison` - Performance across graph structures
- `xml_format_study` - Structured XML output format
- `abstract_reasoning` - Abstract domains without human baseline
- `domain_transfer` - Reasoning across knowledge domains

### 3. Independent Causes in Collider Graphs

The `--indep-causes-collider` flag allows you to specify whether causes in collider graphs should be treated as independent variables.

**Usage**:
```bash
# Generate prompts with independent causes
python scripts/generate_experiment_prompts.py --experiment pilot_study --version 7 --indep-causes-collider

# Generate prompts without independent causes (default)
python scripts/generate_experiment_prompts.py --experiment pilot_study --version 8
```

**Effect on Prompt Generation**:

**With `--indep-causes-collider`**:
```
"Here are the causal relationships: 
low interest rates causes high retirement savings. 
small trade deficits causes high retirement savings. 
interest rates and trade deficits are independent, so they can independently cause retirement savings."
```

**Without the flag (default)**:
```
"Here are the causal relationships: 
low interest rates causes high retirement savings. 
small trade deficits causes high retirement savings."
```

**When to Use**:
- **Use the flag** when you want to explicitly state that causes are independent
- **Don't use the flag** when you want standard causal relationship descriptions
- **Research applications**: Useful for studying how independence assumptions affect LLM reasoning

## Prompt Generators

### RW17Generator (Human-Study-Based)

**Purpose**: Replicates human experimental conditions for comparison studies.

**Features**:
- **Domains**: `["economy", "sociology", "weather"]`
- **Graph Types**: `["collider"]` (configurable)
- **Counterbalance**: `["ppp", "pmm", "mmp", "mpm"]`
- **Use Case**: When you want to compare LLM performance against human data

**Example**:
```python
from src.causalign.prompts.generators.prompt_factory import PromptFactory

# Create RW17 generator
generator = PromptFactory.create_generator("rw17", version="7", output_dir="data/input_llm/rw17/")

# Configure for specific graph types
generator = generator.with_graph_types(["collider", "fork"])

# Generate prompts
style = NumericOnlyStyle()
prompts_df, saved_path = generator.generate_and_save(style)
```

### AbstractGenerator (Abstract/Fantasy Domains)

**Purpose**: Creates prompts for domains without human baseline data.

**Features**:
- **Predefined Domains**: `["abstract_math", "fantasy", "symbolic"]`
- **Custom Domains**: User-defined via YAML files
- **Graph Types**: Configurable
- **Use Case**: Test pure reasoning without human comparison

**Example**:
```python
from src.causalign.prompts.generators.abstract_generator import create_custom_abstract_generator

# Create generator with custom domains
generator = create_custom_abstract_generator(
    version="9",
    custom_domain_files=["my_robotics.yaml", "my_biology.yaml"],
    output_dir="my_prompts/"
)

# Generate prompts
style = NumericOnlyStyle()
prompts_df, saved_path = generator.generate_and_save(style)
```

## Prompt Styles

### Available Styles

```python
PROMPT_STYLES = {
    "single_numeric_response": NumericOnlyStyle,           # Just a number 0-100
    "single_numeric_response_xml": XML format,            # Structured XML output
    " numeric-conf": ConfidenceStyle,                 # Number + confidence rating
    "CoT": ChainOfThoughtStyle,        # Number + reasoning
    "CoT-brief": Brief reasoning,      # Limited reasoning
    "CoT-moderate": Moderate reasoning # Medium reasoning
}
```

### Style Descriptions

```bash
# List available styles
python scripts/generate_experiment_prompts.py --list-styles
```

**Output Examples**:

1. **NumericOnlyStyle**: `"45"`
2. **ConfidenceStyle**: `"45 (high confidence)"`
3. **ChainOfThoughtStyle**: `"45 - I think this because..."`

## Creating Custom Abstract Domains

### 1. Create a YAML Domain File

```yaml
# my_robotics.yaml
domain_name: "robotics"

introduction: >
  Robotics engineers study the complex relationships between different systems 
  that enable autonomous robots to function effectively. They examine how various 
  components interact to produce desired robotic behaviors.

variables:
  X:
    name: "sensor accuracy"
    detailed: "Sensor accuracy refers to how precisely the robot's sensors can detect and measure environmental conditions."
    p_values:
      "1": "high"
      "0": "low"
    m_values:
      "1": "low" 
      "0": "high"
      
  Y:
    name: "battery level"
    detailed: "Battery level indicates the amount of electrical energy available to power the robot's operations."
    p_values:
      "1": "full"
      "0": "depleted"
    m_values:
      "1": "depleted"
      "0": "full"
      
  Z:
    name: "task performance"
    detailed: "Task performance measures how successfully the robot completes its assigned objectives."
    p_values:
      "1": "excellent"
      "0": "poor"
    m_values:
      "1": "poor" 
      "0": "excellent"
```

### 2. Use Custom Domain in Experiment

```python
# In experiment_config.py
"custom_robotics": ExperimentConfig(
    name="custom_robotics",
    description="Test causal reasoning in robotics domain",
    prompt_styles=["single_numeric_response"],
    graph_types=["collider"],
    human_data_match=False,  # This triggers AbstractGenerator
    domains=["robotics"],    # Your custom domain
    models=["gpt-4o"],
    temperatures=[0.0],
),
```

### 3. Generate Prompts

```bash
# Generate prompts for custom experiment
python scripts/generate_experiment_prompts.py --experiment custom_robotics --version 10
```

## Predefined Abstract Domains

### Available Domains

1. **abstract_math**: Mathematical properties and relationships
2. **fantasy**: Magical realm with mystical forces
3. **symbolic**: Abstract symbolic logic systems

### Domain Examples

**Abstract Math**:
- Variables: alpha property, beta property, gamma outcome
- Values: strong/weak, present/absent, positive/negative

**Fantasy**:
- Variables: crystal energy, moon phase alignment, spell potency
- Values: radiant/dim, harmonious/discordant, powerful/weak

**Symbolic**:
- Variables: phi condition, psi condition, omega result
- Values: activated/dormant, enabled/disabled, true/false

## File Structure

```
src/causalign/prompts/
├── generators/
│   ├── prompt_factory.py      # Main factory class
│   ├── base_generator.py      # Abstract base class
│   ├── rw17_generator.py      # Human-study generator
│   └── abstract_generator.py  # Abstract domain generator
├── core/
│   ├── abstract_domains.py    # Predefined abstract domains
│   ├── processing.py          # Core prompt generation logic
│   └── constants.py           # Domain components
├── custom_domains/
│   ├── domain_loader.py       # YAML domain loader
│   └── example_robotics.yaml  # Example custom domain
└── styles/
    ├── base_style.py          # Abstract style base
    ├── numeric_only.py        # Numeric response style
    ├── confidence.py          # Confidence rating style
    └── chain_of_thought.py    # Reasoning style
```

## Advanced Usage

### Programmatic Prompt Generation

```python
from src.causalign.prompts.generators.prompt_factory import PromptFactory
from src.causalign.experiment.config.experiment_config import get_experiment_config

# Get experiment configuration
config = get_experiment_config("pilot_study")

# Generate prompts with independent causes
results = PromptFactory.generate_experiment_prompts(
    experiment_config=config,
    version="7",
    output_dir=Path("data/input_llm/rw17/pilot_study/"),
    indep_causes_collider=True  # Enable independent causes
)

# Generate prompts without independent causes (default)
results = PromptFactory.generate_experiment_prompts(
    experiment_config=config,
    version="8",
    output_dir=Path("data/input_llm/rw17/pilot_study/"),
    indep_causes_collider=False  # Default behavior
)

# Process results
for style_name, style_results in results.items():
    for prompts_df, saved_path in style_results:
        print(f"Generated {len(prompts_df)} prompts for {style_name}")
        print(f"Saved to: {saved_path}")
```

### Creating Custom Generators

```python
from src.causalign.prompts.generators.base_generator import BasePromptGenerator

class CustomGenerator(BasePromptGenerator):
    def get_domains(self) -> List[str]:
        return ["my_domain"]
        
    def get_graph_types(self) -> List[str]:
        return ["collider", "chain"]
        
    def get_counterbalance_conditions(self) -> List[str]:
        return ["ppp", "pmm"]

# Register with factory
PromptFactory.GENERATORS["custom"] = CustomGenerator
```

### Batch Processing Multiple Experiments

```python
import subprocess

experiments = ["pilot_study", "reasoning_methods", "abstract_reasoning"]
version = "10"

for experiment in experiments:
    cmd = [
        "python", "scripts/generate_experiment_prompts.py",
        "--experiment", experiment,
        "--version", version
    ]
    subprocess.run(cmd, check=True)
```

## Customizing Prompt Verbalization

### Understanding the Verbalization Flow

The prompt generation system creates natural language prompts through a series of verbalization functions. Here's the complete flow:

```
generate_prompt_dataframe()
├── verbalize_domain_intro()           # Domain introduction text
├── verbalize_variables_section()      # "Some have... Others have..." text
├── verbalize_causal_mechanism()       # Causal relationships
└── verbalize_inference_task()         # Task description and instructions
```

**Key File**: `src/causalign/prompts/core/verbalization.py`

### Where to Change Verbalization

#### 1. Variable Descriptions (`verbalize_variables_section`)

**Location**: Lines 580-621 in `verbalization.py`

**Current Logic**:
```python
if domain_name == "systems":
    variables_text += f"{detailed} Sometimes {name} is {value_1} and sometimes {name} is {value_0}. "
else:
    variables_text += f"{detailed} Some {domain_name} have {value_1} {name}. Others have {value_0} {name}. "
```

**Examples**:
- **Abstract domains**: `"Sometimes A is high and sometimes A is low."`
- **Human domains**: `"Some economies have high interest rates. Others have low interest rates."`

#### 2. Domain-Specific Observation Text (`verbalize_inference_task`)

**Location**: Lines 500-550 in `verbalization.py`

**Current Logic**:
```python
if domain == "weather":
    observation_text = "Suppose that there is a weather system that is known to have "
elif domain == "economy":
    observation_text = "Suppose that the economy is currently known to have "
elif domain == "sociology":
    observation_text = "Suppose that the society you live in currently exhibits the following: "
else:
    observation_text = "You are currently observing: "
```

#### 3. Causal Relationships (`verbalize_causal_mechanism`)

**Location**: Lines 335-400 in `verbalization.py`

**Current Logic**: Creates causal statements like `"high A cause powerful C. strong B cause powerful C."`

**Independent Causes Feature**: When `indep_causes_collider=True`, adds independence explanation:
```python
if indep_causes_collider:
    indep_cause_explanation = f"{x_name} and {y_name} are independent, so they can independently cause {z_name}."
    causal_text += f"{x_z_relation}{x_z_explanation} "
    causal_text += f"{y_z_relation}{y_z_explanation} {indep_cause_explanation} "
else:
    causal_text += f"{x_z_relation}{x_z_explanation} "
    causal_text += f"{y_z_relation}{y_z_explanation} "
```

### Customization Options

#### Option 1: Direct Code Modification (Current Approach)

**Pros**: Simple, immediate effect
**Cons**: Changes affect all users, risk of breaking existing functionality

```python
# In verbalization.py, modify the condition:
if domain_name == "systems":
    variables_text += f"{detailed} Variable {name} can be {value_1} or {value_0}. "
```

#### Option 2: Domain-Specific Overrides

**Pros**: Targeted changes, preserves existing logic
**Cons**: Requires code changes for each domain

```python
# Add domain-specific conditions:
if domain_name == "systems":
    variables_text += f"{detailed} Sometimes {name} is {value_1} and sometimes {name} is {value_0}. "
elif domain_name == "mathematics":
    variables_text += f"{detailed} Variable {name} takes values {value_1} or {value_0}. "
elif domain_name == "physics":
    variables_text += f"{detailed} The {name} parameter can be {value_1} or {value_0}. "
```

#### Option 3: Configuration-Driven Approach (Future TODO)

**Pros**: No code changes needed, flexible, user-friendly
**Cons**: Requires significant refactoring

```yaml
# In domain YAML file:
verbalization:
  variable_description: "The {name} parameter can be {value_1} or {value_0}."
  observation_intro: "Consider a system where you observe:"
  causal_format: "{cause} leads to {effect}."
```

#### Option 4: Function Override in Notebooks (Future TODO)

**Pros**: Experiment-specific changes, no permanent code modification
**Cons**: Requires understanding of function signatures

```python
# In your experiment notebook:
def custom_verbalize_variables_section(domain_dict, row):
    """Custom verbalization for this experiment."""
    # Your custom logic here
    return custom_text

# Override the function temporarily
import src.causalign.prompts.core.verbalization as verbalization
verbalization.verbalize_variables_section = custom_verbalize_variables_section
```

#### Option 5: Plugin System (Future TODO)

**Pros**: Modular, extensible, clean separation
**Cons**: Complex implementation

```python
# Create custom verbalizer plugin
class CustomVerbalizer:
    def verbalize_variables(self, domain_dict, row):
        return custom_text
    
    def verbalize_observation(self, domain_dict, row):
        return custom_text

# Register with factory
PromptFactory.register_verbalizer("custom", CustomVerbalizer())
```

### Common Customization Needs

#### 1. Plural vs Singular Forms

**Problem**: `"Some systems have high A"` vs `"Sometimes A is high"`

**Solutions**:
- **Option 2**: Add domain-specific conditions
- **Option 3**: Configuration-driven approach
- **Option 4**: Function override in notebook

#### 2. Domain-Specific Terminology

**Problem**: Different domains need different language patterns

**Examples**:
- **Mathematics**: `"Variable X takes values..."`
- **Physics**: `"The parameter can be..."`
- **Psychology**: `"The trait varies between..."`

#### 3. Formal vs Informal Tone

**Problem**: Abstract domains might need more formal language

**Solutions**:
- **Option 2**: Domain-specific conditions
- **Option 3**: Configuration with tone settings

### Future TODO: Implementation Recommendations

#### Priority 1: Configuration-Driven Approach

```python
# In domain YAML:
verbalization:
  variable_description_template: "Sometimes {name} is {value_1} and sometimes {name} is {value_0}."
  observation_intro_template: "You are currently observing: {observations}."
  causal_format_template: "{cause} cause {effect}."
```

#### Priority 2: Function Override System

```python
# In experiment notebook:
from src.causalign.prompts.core.verbalization import VerbalizationOverrides

overrides = VerbalizationOverrides()
overrides.set_variable_description(custom_function)
overrides.set_observation_intro(custom_function)

# Use in prompt generation
results = PromptFactory.generate_experiment_prompts(
    experiment_config=config,
    version="7",
    output_dir=output_dir,
    verbalization_overrides=overrides
)
```

#### Priority 3: Plugin Architecture

```python
# Create custom verbalizer
class FormalVerbalizer:
    def verbalize_variables(self, domain_dict, row):
        return formal_variable_text
    
    def verbalize_observation(self, domain_dict, row):
        return formal_observation_text

# Register and use
PromptFactory.register_verbalizer("formal", FormalVerbalizer())
```

### Best Practices for Customization

1. **Start Simple**: Use Option 2 (domain-specific conditions) for quick changes
2. **Document Changes**: Add comments explaining customizations
3. **Test Thoroughly**: Verify changes work with all domain types
4. **Consider Impact**: Changes affect all experiments using that domain
5. **Plan for Future**: Consider implementing configuration-driven approach

## Troubleshooting

### Common Issues

1. **Unknown experiment error**:
   ```bash
   # Check available experiments
   python scripts/generate_experiment_prompts.py --list-experiments
   ```

2. **Custom domain not found**:
   ```bash
   # Check domain file exists and is valid YAML
   cat src/causalign/prompts/custom_domains/my_domain.yaml
   ```

3. **Output directory issues**:
   ```bash
   # Create output directory
   mkdir -p data/input_llm/rw17/my_experiment/
   ```

4. **Independent causes flag not working**:
   ```bash
   # Verify the flag is being passed correctly
   python scripts/generate_experiment_prompts.py --experiment pilot_study --version 7 --indep-causes-collider --help
   
   # Check generated prompts for independence statement
   grep "independent" data/input_llm/rw17/pilot_study/7_v_single_numeric_response_LLM_prompting_collider.csv
   ```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Generate with debug output
results = PromptFactory.generate_experiment_prompts(
    experiment_config=config,
    version="7",
    output_dir=output_dir
)
```

## Integration with Experiment Pipeline

### Complete Workflow

1. **Generate Prompts**:
   ```bash
   python scripts/generate_experiment_prompts.py --experiment pilot_study --version 7
   ```

2. **Run LLM Experiments**:
   ```bash
   python run_experiment.py --version 7 --experiment pilot_study --model gpt-4o
   ```

3. **Process Results**:
   ```bash
   python scripts/run_data_pipeline.py --version 7 --experiment pilot_study
   ```

4. **Visualize Results**:
   ```bash
   python scripts/plot_results.py --version 7 --experiment pilot_study
   ```

## Best Practices

### 1. Version Management
- Use consistent version numbers across the pipeline
- Document version changes in experiment logs
- Use semantic versioning for major changes

### 2. Domain Design
- Keep variable names clear and descriptive
- Ensure logical relationships between variables
- Test domain coherence before large-scale experiments

### 3. Style Selection
- Start with simple styles for initial testing
- Use chain-of-thought for complex reasoning tasks
- Consider XML format for structured analysis

### 4. File Organization
- Use descriptive experiment names
- Organize custom domains in separate directories
- Document domain creation process

## Contributing

### Adding New Domains

1. Create YAML file in `src/causalign/prompts/custom_domains/`
2. Follow the template structure
3. Test with small experiment first
4. Document domain characteristics

### Adding New Styles

1. Extend `BasePromptStyle` class
2. Implement required methods
3. Register in `PromptFactory.PROMPT_STYLES`
4. Add tests and documentation

### Adding New Generators

1. Extend `BasePromptGenerator` class
2. Implement abstract methods
3. Register in `PromptFactory.GENERATORS`
4. Add configuration options

## References

- **Main Workflow**: `README_Workflow.md`
- **Data Processing**: `DATA_PROCESSING_GUIDE.md`
- **API Usage**: `API_USAGE.md`
- **Testing**: `TESTING.md`

## Support

For issues or questions:
1. Check existing documentation
2. Review troubleshooting section
3. Examine example files in `src/causalign/prompts/`
4. Create issue with detailed error information 