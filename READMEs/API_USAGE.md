![MyPackage Logo](assets/logo.png)
# Causal Alignment API Pipeline

A user-friendly command-line tool for running causal reasoning experiments with Large Language Models (LLMs).

##  Quick Start

### 1. Install the Package
This package is still under development but already usable. 
The package is already installed in development mode if you followed the setup instructions:
```bash
pip install -e .
```

### 2. Set Up API Keys

Create a `.env` file **in the project root directory** (same level as `run_experiment.py`) with your API keys:
```bash
# OpenAI API Key (get from: https://platform.openai.com/api-keys)
OPENAI_API_KEY=sk-your-openai-key-here

# Anthropic API Key (get from: https://console.anthropic.com/account/keys)  
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Google AI API Key (get from: https://aistudio.google.com/app/apikey)
GOOGLE_API_KEY=your-google-ai-key-here
```

### 3. Run an Experiment

```bash
# Activate your virtual environment
source ~/.virtualenvs/your-environment/bin/activate

# See available datasets
python run_experiment.py

# Run an experiment
python run_experiment.py \
    --dataset data/input_llm/rw17/6_v_single_numeric_response_LLM_prompting_collider.csv \
    --model gpt-4o \
    --experiment pilot_study \
    --temperature 0.0
```

##  Output Structure

Results are automatically organized in:
```
data/output_llm/
└── experiment-name/
    └── model-name/
        └── temperature-X.X/
            └── results.csv
```

For example:
```
data/output_llm/
└── pilot_study/
    └── gpt-4o/
        └── temperature-0.0/
            └── pilot_study_gpt-4o_0.0_temp_6_v_single_numeric_response_LLM_prompting_collider.csv
```

##  Supported Models 
The current API supports all models from OpenAI, Anthropic, and Google, including OpenAI's reasoning models 

### OpenAI
- `gpt-4o`
- `gpt-4` 
- `gpt-3.5-turbo`
- `gpt-5-nano`
- `o3`
- ...

### Anthropic (Claude)
- `claude-3-opus-20240229`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`
- ...

### Google
- `gemini-1.5-pro`
- `gemini-pro`
- ...

##  Usage Examples

### Basic Usage
```bash
python run_experiment.py \
    --dataset data/input_llm/rw17/6_v_single_numeric_response_LLM_prompting_collider.csv \
    --model gpt-4o \
    --experiment my_experiment
```

### With Custom Temperature
```bash
python run_experiment.py \
    --dataset data/input_llm/rw17/6_v_single_numeric_response_LLM_prompting_collider.csv \
    --model claude-3-opus-20240229 \
    --experiment temperature_test \
    --temperature 0.7
```

### With API Key Override
```bash
python run_experiment.py \
    --dataset data/input_llm/rw17/6_v_single_numeric_response_LLM_prompting_collider.csv \
    --model gemini-1.5-pro \
    --experiment secure_test \
    --api-key your-api-key-here
```

### Multiple Runs
```bash
python run_experiment.py \
    --dataset data/input_llm/rw17/6_v_single_numeric_response_LLM_prompting_collider.csv \
    --model gpt-3.5-turbo \
    --experiment reliability_test \
    --runs 3
```

## 📊 CSV File Structure

### Input CSV Format

Input CSV files **must** contain these exact columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Unique identifier for each prompt | `1, 2, 3...` |
| `prompt` | string | The complete prompt text for the LLM | `"Economists seek to describe..."` |
| `prompt_category` | string | Type of prompt format | `"single_numeric_response"` |
| `graph` | string | Causal graph structure type | `"collider"`, `"fork"`, `"chain"` |
| `domain` | string | Subject domain | `"economy"`, `"biology"` |
| `cntbl_cond` | string | Counterbalance condition | `"ppp"`, `"pmm"`, `"mpm"` |
| `task` | string | Specific inference task | `"a"`, `"b"`, `"c"` |

**Example Input Row:**
```csv
id,prompt,prompt_category,graph,domain,cntbl_cond,task
1,"Economists seek to describe and predict...",single_numeric_response,collider,economy,ppp,a
```

### Output CSV Format

The API automatically generates output files with these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `id` | integer | Same as input | `1, 2, 3...` |
| `response` | string | LLM's raw response | `"<response><likelihood>75</likelihood></response>"` |
| `prompt_category` | string | From input | `"single_numeric_response"` |
| `graph` | string | From input | `"collider"` |
| `domain` | string | From input | `"economy"` |
| `task` | string | From input | `"a"` |
| `cntbl_cond` | string | From input | `"ppp"` |
| `subject` | string | Model name used | `"gpt-4o"` |
| `temperature` | float | Temperature setting | `0.0` |

**Example Output Row:**
```csv
id,response,prompt_category,graph,domain,task,cntbl_cond,subject,temperature
1,"<response><likelihood>75</likelihood></response>",single_numeric_response,collider,economy,a,ppp,gpt-4o,0.0
```

##  Available Datasets (update)

Current datasets in `data/input_llm/rw17/`:
- `6_v_single_numeric_response_LLM_prompting_collider.csv`
- `5_v_single_numeric_response_LLM_prompting_collider.csv` 
- `4_v_numeric-only_LLM_prompting_collider.csv`
- `4_v_CoT_LLM_prompting_collider.csv`
- `4_v_ numeric-conf_LLM_prompting_collider.csv`

##  Command Line Options

```
--dataset, -d       Path to input CSV file (required)
--model, -m         LLM model to use (required)
--experiment, -e    Experiment name (required)
--temperature, -t   Temperature for generation (default: 0.0)
--api-key          API key override
--output-dir       Base output directory (default: data/output_llm)
--runs, -n         Number of prompt repetitions (default: 1)
--verbose, -v      Enable verbose output
--help, -h         Show help message
```

## Next Steps

After running experiments, you can:

1. **Load Results**: Use the analysis notebooks to load and examine results
2. **Compare Models**: Run the same experiment with different models
3. **Statistical Analysis**: Use the correlation analysis tools to compare LLM vs human performance
4. **Visualization**: Generate plots using the visualization components

Most of the analysis options are not part of the package itself but live in the `\scripts` folder on the package root level. 

##  Notes

- API calls are rate-limited (1 second delay between requests) (TODO: some LLM providers support batch API calls, which is half the price)
- Results include metadata (temperature, model, experiment name)
- Failed API calls are logged with error messages
- Temporary files are automatically cleaned up

## Architecture & File Organization

### .env File Location

The `.env` file **must** be placed at the **project root directory** (same level as `run_experiment.py`). 

```
causalign/                    # ← Project root
├── .env                        # ← .env file goes here
├── run_experiment.py           # ← Main CLI script  
├── src/
│   └── causalign/
├── data/
└── ...
```

The script uses `load_dotenv()` which automatically looks for `.env` in the current working directory. Since you run the script from the project root, this is where it expects the `.env` file.

**Alternative:** You can also set environment variables directly in your shell instead of using a `.env` file:
```bash
export OPENAI_API_KEY="your-key-here"
python run_experiment.py --dataset data/input_llm/rw17/dataset.csv --model gpt-4o --experiment test
```

### Why `run_experiment.py` Lives at Project Root

The `run_experiment.py` script is positioned at the **project root** rather than inside `src/causalign/experiment/api/` for several important reasons:

#### 1. **User-Facing Tool**
- This is a **command-line interface** for end users, not internal package code
- Users expect CLI tools at the project root (like `setup.py`, `manage.py` in Django, etc.)
- Easy to find and run: `python run_experiment.py --help`

#### 2. **Separation of Concerns**
```
causalign/
├── run_experiment.py           # ← User interface (CLI)
├── src/causalign/           # ← Package internals
│   └── experiment/api/        # ← API implementation modules
│       ├── api_runner.py      # ← Core experiment logic
│       ├── client.py          # ← LLM client implementations
│       └── data_loader.py     # ← Data loading utilities
```

#### 3. **Dependencies Used**
The CLI script imports and orchestrates these internal modules:

- **`src/causalign/experiment/api/api_runner.py`** - `ExperimentRunner` class
- **`src/causalign/experiment/api/client.py`** - `LLMConfig` class and LLM clients
- **`src/causalign/config/paths.py`** - Path management (via .env loading)

#### 4. **Installation Independence**  
- Works whether package is installed (`pip install -e .`) or not
- Users can run experiments without understanding package internals
- Clean separation between "interface" and "implementation"

This follows Python packaging best practices where:
- **Package code** (in `src/`) contains reusable modules
- **Scripts** (at root) provide user interfaces to that functionality 