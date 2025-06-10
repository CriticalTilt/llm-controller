# LLMController Test Suite - Setup and Execution Guide

## 📋 Overview

This comprehensive test suite validates the LLMController class functionality with:
- **Unit Tests**: Basic functionality and initialization
- **Mocked Tests**: Behavioral testing without API calls
- **Integration Tests**: Real API testing (requires keys)
- **Performance Tests**: Speed and efficiency validation
- **Behavioral Tests**: Real-world usage patterns

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install testing framework and dependencies
pip install pytest pytest-timeout pytest-mock
pip install langchain anthropic openai python-dotenv

# Optional: For coverage reports
pip install pytest-cov
```

### 2. Set Up Your Environment

Create or update your `.env` file:
```bash
# Required for integration tests
ANTHROPIC_API_KEY=your_anthropic_key_here

# Optional: For provider switching tests
OPENAI_API_KEY=your_openai_key_here
```

### 3. Run Tests

```bash
# Run all tests
pytest test_llm_controller.py -v

# Run with detailed output
pytest test_llm_controller.py -v --tb=short

# Run only unit tests (no API calls)
pytest test_llm_controller.py::TestLLMControllerUnit -v

# Run only mocked tests
pytest test_llm_controller.py::TestLLMControllerMocked -v

# Run integration tests (requires API keys)
pytest test_llm_controller.py::TestLLMControllerIntegration -v
```

## 📊 Test Categories Explained

### Unit Tests (`TestLLMControllerUnit`)
Tests core functionality without external dependencies:
- ✅ Initialization with valid/invalid providers
- ✅ Model switching logic
- ✅ Property access
- ✅ Error handling

**Run with**: `pytest test_llm_controller.py::TestLLMControllerUnit -v`

### Mocked Tests (`TestLLMControllerMocked`)
Tests behavior using mocked API responses:
- ✅ Method delegation to underlying models
- ✅ Invoke/stream/generate functionality
- ✅ Attribute forwarding
- ✅ Chain integration

**Run with**: `pytest test_llm_controller.py::TestLLMControllerMocked -v`

### Integration Tests (`TestLLMControllerIntegration`)
Tests with real API calls (requires API keys):
- ✅ Real Claude API responses
- ✅ Streaming functionality
- ✅ Model switching with live models
- ✅ Response time validation

**Run with**: `pytest test_llm_controller.py::TestLLMControllerIntegration -v`

### Behavioral Tests (`TestLLMControllerBehavioral`)
Tests real-world usage patterns:
- ✅ Error recovery
- ✅ Rapid model switching
- ✅ Thread safety
- ✅ Edge cases

**Run with**: `pytest test_llm_controller.py::TestLLMControllerBehavioral -v`

## 🔧 Advanced Testing Commands

### Run Tests with Coverage
```bash
# Generate coverage report
pytest test_llm_controller.py --cov=llm_controller --cov-report=html

# View coverage in terminal
pytest test_llm_controller.py --cov=llm_controller --cov-report=term-missing
```

### Run Specific Test Methods
```bash
# Test only initialization
pytest test_llm_controller.py::TestLLMControllerUnit::test_initialization_valid_provider -v

# Test only real API calls
pytest test_llm_controller.py::TestLLMControllerIntegration::test_real_invoke -v

# Test performance
pytest test_llm_controller.py::TestPerformance -v
```

### Run Tests with Different Verbosity
```bash
# Minimal output
pytest test_llm_controller.py -q

# Normal output
pytest test_llm_controller.py

# Verbose output
pytest test_llm_controller.py -v

# Very verbose (show all output)
pytest test_llm_controller.py -vv -s
```

## 🎯 Test Configuration Options

### Environment Variables for Testing

```bash
# Required for integration tests
export ANTHROPIC_API_KEY="your_key_here"

# Optional for provider switching tests
export OPENAI_API_KEY="your_key_here"
export HUGGINGFACE_API_KEY="your_key_here"
export XAI_API_KEY="your_key_here"

# Test configuration
export TEST_TIMEOUT=30  # Timeout for long tests
export TEST_MODEL="claude-3-haiku-20240307"  # Faster model for testing
```

### pytest.ini Configuration

Create a `pytest.ini` file in your project root:

```ini
[tool:pytest]
testpaths = .
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --tb=short
    --strict-markers
    --disable-warnings
markers =
    integration: Integration tests requiring API keys
    slow: Slow running tests
    unit: Fast unit tests
    mocked: Tests using mocks
```

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Error: Cannot import LLMController
# Solution: Make sure the class file is in the same directory or update imports
```

#### 2. API Key Issues
```bash
# Error: API key not found
# Solution: Check your .env file and environment variables
echo $ANTHROPIC_API_KEY  # Should show your key
```

#### 3. Timeout Errors
```bash
# Error: Tests timing out
# Solution: Increase timeout or use faster models
pytest test_llm_controller.py --timeout=60
```

#### 4. Mock Issues
```bash
# Error: Mock not working
# Solution: Install pytest-mock
pip install pytest-mock
```

## 📈 Continuous Integration

### GitHub Actions Example

Create `.github/workflows/test.yml`:

```yaml
name: Test LLMController

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.8'
    
    - name: Install dependencies
      run: |
        pip install pytest pytest-mock pytest-cov
        pip install langchain anthropic openai python-dotenv
    
    - name: Run unit tests
      run: pytest test_llm_controller.py::TestLLMControllerUnit -v
    
    - name: Run mocked tests
      run: pytest test_llm_controller.py::TestLLMControllerMocked -v
    
    - name: Run integration tests
      env:
        ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      run: pytest test_llm_controller.py::TestLLMControllerIntegration -v
      if: env.ANTHROPIC_API_KEY != ''
```

## 🎨 Custom Test Scenarios

### Add Your Own Tests

```python
# test_custom.py
import pytest
from llm_controller import LLMController

class TestCustomScenarios:
    def test_my_specific_use_case(self):
        """Test your specific use case"""
        controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
        
        # Your custom test logic here
        response = controller.invoke("Your specific prompt")
        
        # Your assertions
        assert "expected_content" in response.content
```

## 📊 Expected Test Results

### Successful Run Output
```
========================= test session starts =========================
collected 20 items

test_llm_controller.py::TestLLMControllerUnit::test_initialization_valid_provider PASSED
test_llm_controller.py::TestLLMControllerUnit::test_initialization_invalid_provider PASSED
test_llm_controller.py::TestLLMControllerMocked::test_invoke_delegation PASSED
test_llm_controller.py::TestLLMControllerIntegration::test_real_invoke PASSED
...

========================= 20 passed in 15.23s =========================
```

### Performance Benchmarks
- **Unit Tests**: < 1 second total
- **Mocked Tests**: < 5 seconds total
- **Integration Tests**: 10-30 seconds (depends on API speed)
- **Full Suite**: 15-45 seconds

## 🔍 Next Steps

1. **Run the basic tests** to validate your setup
2. **Add your API keys** to enable integration tests
3. **Customize tests** for your specific use cases
4. **Set up CI/CD** for automated testing
5. **Monitor performance** over time

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify your environment setup
3. Run tests with `-vv -s` for detailed output
4. Check API key permissions and quotas

Happy testing! 🚀