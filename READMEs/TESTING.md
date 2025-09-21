
![MyPackage Logo](assets/logo.png)

# Testing Guide for CausalAlign

Below we describe the testing infrastructure for the CausalAlign package.

##  Test Structure

The test suite is organized into several categories to ensure comprehensive coverage:

```
tests/
├── conftest.py              # Shared fixtures and pytest configuration
├── fixtures/                # Test data and mock responses
│   ├── sample_data.csv     # Sample input CSV for testing
│   └── ...                 # Additional test fixtures
├── unit/                   # Unit tests (fast, isolated)
│   ├── test_api_clients.py # Test LLM API client classes
│   ├── test_experiment_runner.py # Test experiment orchestration
│   └── test_data_processing.py # Test data validation/processing
├── integration/            # Integration tests (slower, cross-component)
│   └── test_api_pipeline.py # Test end-to-end API pipeline
├── cli/                    # Command-line interface tests
│   └── test_cli.py         # Test run_experiment.py CLI
└── README.md               # This file
```

## Types of Tests

### **Unit Tests** (`tests/unit/`)
Fast, isolated tests that verify individual components:

- **API Clients**: Test OpenAI, Anthropic, and Gemini client initialization, response generation, error handling, and model validation
- **Experiment Runner**: Test experiment orchestration, data validation, output directory creation, and result processing
- **Data Processing**: Test CSV validation, data transformations, encoding handling, and format compliance

### **Integration Tests** (`tests/integration/`)
Tests that verify components work together correctly:

- **Full Pipeline**: End-to-end testing of the API pipeline with mocked LLM responses
- **Error Handling**: Test graceful degradation when APIs fail
- **Rate Limiting**: Verify proper delays between API calls
- **CSV Format Consistency**: Ensure input/output CSV formats match specifications

### **CLI Tests** (`tests/cli/`)
Tests for the command-line interface:

- **Argument Validation**: Test required/optional arguments
- **Error Messages**: Verify helpful error messages for common mistakes  
- **File Handling**: Test input file validation and output generation
- **API Key Management**: Test environment variable and command-line API key handling

## 🚀 Running Tests

### **Install Test Dependencies**
```bash
pip install -r requirements.txt
```

### **Run All Tests**
```bash
pytest
```

### **Run Specific Test Categories**
```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# CLI tests only  
pytest tests/cli/
```

### **Run Tests with Coverage**
```bash
pytest --cov=src/causalign --cov-report=html
```

### **Run Tests with Specific Markers**
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests  
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

##  Test Configuration

### **pytest.ini Settings**
The test suite is configured with the following settings:

- **Coverage Target**: 80% minimum code coverage
- **Test Discovery**: Automatically finds `test_*.py` files
- **Reporting**: Detailed coverage reports in terminal and HTML
- **Markers**: Organized test categories for selective running

### **Fixtures** (`conftest.py`)
Shared test fixtures provide:

- **Sample Data**: Representative input datasets
- **Mock API Responses**: Simulated LLM responses for all providers
- **Temporary Files**: Safe temporary directories for test outputs  
- **Environment Variables**: Mock API keys and configuration

## Writing New Tests

### **Adding Unit Tests**
Create new test files in `tests/unit/` following the naming convention `test_[module_name].py`:

```python
import pytest
from causalign.your_module import YourClass

class TestYourClass:
    def test_basic_functionality(self):
        """Test basic functionality."""
        obj = YourClass()
        result = obj.some_method()
        assert result == expected_value
    
    def test_error_handling(self):
        """Test error handling."""
        obj = YourClass()
        with pytest.raises(ValueError):
            obj.invalid_operation()
```

### **Adding Integration Tests**
Integration tests should test realistic end-to-end scenarios:

```python
@patch('causalign.experiment.api.llm_clients.OpenAIClient')
def test_full_workflow(self, mock_client, sample_input_data, tmp_path):
    """Test complete workflow from input to output."""
    # Setup mocks
    mock_client.return_value.generate_response.return_value = "Test response"
    
    # Run the workflow
    runner = ExperimentRunner("gpt-3.5-turbo", "test", api_key="key")
    results = runner.run_experiment(sample_input_data)
    
    # Verify results
    assert len(results) == len(sample_input_data)
    # ... additional assertions
```

### **Test Markers**
Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_unit_functionality():
    """Fast unit test."""
    pass

@pytest.mark.integration  
def test_integration_scenario():
    """Integration test."""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Test that takes significant time."""
    pass
```

##  Best Practices

### **Mocking External Dependencies**
Always mock external API calls to avoid:
- Unnecessary API charges
- Test flakiness due to network issues
- Rate limiting during test runs

```python
@patch('causalign.experiment.api.llm_clients.OpenAIClient')
def test_with_mocked_api(self, mock_client):
    mock_client.return_value.generate_response.return_value = "Mocked response"
    # Your test code here
```

### **Using Fixtures**
Leverage shared fixtures for common test data:

```python
def test_data_processing(self, sample_input_data, tmp_path):
    """Use fixtures for consistent test data."""
    # sample_input_data and tmp_path are provided by fixtures
    pass
```

### **Parameterized Tests**
Test multiple scenarios efficiently:

```python
@pytest.mark.parametrize("model,provider", [
    ("gpt-3.5-turbo", "openai"),
    ("claude-3-sonnet-20240229", "anthropic"),
    ("gemini-pro", "gemini")
])
def test_multiple_models(self, model, provider):
    """Test with different model configurations."""
    pass
```

## Debugging Tests

### **Running Individual Tests**
```bash
# Run specific test file
pytest tests/unit/test_api_clients.py

# Run specific test class
pytest tests/unit/test_api_clients.py::TestOpenAIClient

# Run specific test method
pytest tests/unit/test_api_clients.py::TestOpenAIClient::test_init_with_api_key
```

### **Verbose Output**
```bash
# Show detailed output
pytest -v -s

# Show print statements
pytest -s
```

### **Drop into Debugger on Failure**
```bash
pytest --pdb
```

##  Coverage Reports

### **Generate HTML Coverage Report**
```bash
pytest --cov=src/causalign --cov-report=html
open htmlcov/index.html  # View in browser
```

### **Terminal Coverage Report**
```bash
pytest --cov=src/causalign --cov-report=term-missing
```

##  Continuous Integration

The test suite is designed to run in CI/CD environments:

- **Fast Execution**: Unit tests run quickly for rapid feedback
- **No External Dependencies**: All API calls are mocked
- **Deterministic**: Tests produce consistent results across environments (TODO)
- **Comprehensive**: High coverage ensures code quality 