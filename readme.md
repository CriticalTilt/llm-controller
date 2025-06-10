# LLMController - Universal LangChain Model Switcher

A unified interface for seamlessly switching between different LLM providers while maintaining full LangChain compatibility. Switch from OpenAI to Claude to Ollama with just one line of code!

## 🚀 Quick Start

```python
from llm_controller import LLMController

# Initialize with Claude
llm = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
response = llm.invoke("Hello!")

# Switch to OpenAI
llm.switch_model(llm="gpt-4", provider="openai")
response = llm.invoke("Same interface, different model!")
```

## 🎯 Why LLMController?

### The Problem
- Different LLM providers have different APIs and interfaces
- Switching between models requires code changes throughout your application
- Testing multiple models means rewriting chains, agents, and pipelines
- No unified way to compare responses across providers

### The Solution
**LLMController** provides a single, consistent interface that:
- ✅ Works with **all LangChain features** (chains, agents, streaming, etc.)
- ✅ Supports **runtime model switching** without code changes
- ✅ Maintains **full compatibility** with existing LangChain code
- ✅ Enables **easy A/B testing** between different models
- ✅ Provides **fallback mechanisms** for reliability

📊 Comparison Table:
| Feature | Base Models | LLMController |
|---------|-------------|---------------|
| **Setup** | Different code for each provider | One unified interface |
| **Switching** | Rewrite code | `llm.switch_model()` |
| **A/B Testing** | Manual setup for each | Loop through providers |
| **Fallbacks** | Try/catch for each model | Automatic fallback chain |
| **Cost Optimization** | Manual cost tracking | Built-in cost-aware switching |
| **Environment Config** | Hard-coded models | Dynamic model selection |
| **Chain Compatibility** | Provider-specific chains | Universal chains |
| **Error Handling** | Provider-specific errors | Unified error handling |

### 🎯 When You DON'T Need LLMController:

Single provider: If you only ever use OpenAI
Static setup: If you never switch models
Simple scripts: One-off tasks with no complexity
Learning: When understanding base LangChain concepts

### 🎯 When LLMController is Essential:

Multi-provider apps: Using OpenAI, Claude, and local models
Production systems: Need fallbacks and reliability
Research/testing: Comparing different models
Cost optimization: Switching based on budget
Dynamic apps: Model choice depends on user/task
Enterprise: Need provider flexibility for compliance

### 💡 The Big Picture:
Base models = Individual tools (screwdriver, hammer, wrench)
LLMController = Swiss Army knife that contains all tools with one interface
You could carry individual tools, but the Swiss Army knife is more convenient when you need multiple tools and want to switch between them quickly!

## Use Cases
## 🎯 **Core Value: Runtime Provider Switching**

### **Without LLMController (Traditional Approach):**
```python
# You have to decide upfront and rewrite code to switch
claude_model = ChatAnthropic(model="claude-3-sonnet", api_key="...")
openai_model = ChatOpenAI(model="gpt-4", api_key="...")

# Different interfaces, parameters, and setup for each
claude_response = claude_model.invoke("Hello")
openai_response = openai_model.invoke("Hello")

# To switch providers, you need to change your code
# chain = prompt | claude_model | parser  # Hard-coded to Claude
# chain = prompt | openai_model | parser  # Need to rewrite
```

### **With LLMController:**
```python
# One interface for all providers
llm = LLMController(llm="claude-3-sonnet", provider="claude")
response = llm.invoke("Hello")

# Runtime switching without code changes
llm.switch_model(llm="gpt-4", provider="openai")
response = llm.invoke("Hello")  # Same code, different model!

# Chains work seamlessly across providers
chain = prompt | llm | parser  # Works with ANY provider
```

## 🚀 **Specific Benefits Over Base Models:**

### **1. Provider Abstraction & Consistency**
```python
# Base models have different parameter names and patterns:
claude = ChatAnthropic(model="claude-3-sonnet", api_key=anthropic_key)
openai = ChatOpenAI(model="gpt-4", api_key=openai_key)  
ollama = Ollama(model="llama2", base_url="http://localhost:11434")

# LLMController unifies them:
llm = LLMController(llm="claude-3-sonnet", provider="claude")
llm.switch_model(llm="gpt-4", provider="openai")
llm.switch_model(llm="llama2", provider="ollama")
# Same interface, different providers!
```

### **2. A/B Testing & Model Comparison**
```python
def compare_models(question):
    results = {}
    llm = LLMController(llm="claude-3-sonnet", provider="claude")
    
    # Test multiple models with same code
    models = [
        ("claude", "claude-3-sonnet"),
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"),
        ("ollama", "llama2")
    ]
    
    for provider, model in models:
        llm.switch_model(llm=model, provider=provider)
        results[f"{provider}_{model}"] = llm.invoke(question)
    
    return results

# Compare 4 different models with 6 lines of code!
results = compare_models("Explain quantum computing")
```

### **3. Graceful Fallbacks & Error Recovery**
```python
def robust_query(question):
    llm = LLMController(llm="gpt-4", provider="openai")
    
    # Try premium model first, fallback to cheaper options
    fallbacks = [
        ("openai", "gpt-4"),
        ("openai", "gpt-3.5-turbo"),
        ("claude", "claude-3-haiku"),
        ("ollama", "llama2")  # Free local fallback
    ]
    
    for provider, model in fallbacks:
        try:
            llm.switch_model(llm=model, provider=provider)
            return llm.invoke(question)
        except Exception as e:
            print(f"Failed with {provider}/{model}: {e}")
            continue
    
    raise Exception("All models failed")

# Automatic fallback chain!
response = robust_query("Help me with this code")
```

### **4. Dynamic Configuration Based on Task**
```python
class AdaptiveLLM:
    def __init__(self):
        self.controller = LLMController(llm="gpt-3.5-turbo", provider="openai")
    
    def query(self, text, task_type="general"):
        # Choose optimal model based on task
        if task_type == "creative":
            self.controller.switch_model("claude-3-sonnet", "claude")
        elif task_type == "coding":
            self.controller.switch_model("gpt-4", "openai")
        elif task_type == "simple":
            self.controller.switch_model("gpt-3.5-turbo", "openai")
        elif task_type == "local":
            self.controller.switch_model("llama2", "ollama")
        
        return self.controller.invoke(text)

# Automatically choose the best model for each task
adaptive = AdaptiveLLM()
creative_response = adaptive.query("Write a poem", "creative")
code_response = adaptive.query("Debug this function", "coding")
quick_response = adaptive.query("What's 2+2?", "simple")
```

### **5. Cost Optimization**
```python
class CostOptimizedLLM:
    def __init__(self):
        self.controller = LLMController(llm="gpt-3.5-turbo", provider="openai")
        self.costs = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3-sonnet": 0.015,
            "claude-3-haiku": 0.0025,
            "llama2": 0.0  # Local
        }
    
    def query(self, text, max_cost=0.01):
        # Choose cheapest model under budget
        affordable_models = [
            (model, cost) for model, cost in self.costs.items() 
            if cost <= max_cost
        ]
        
        if not affordable_models:
            raise ValueError(f"No models available under ${max_cost}")
        
        # Pick the most expensive model we can afford (best quality)
        best_model = max(affordable_models, key=lambda x: x[1])[0]
        
        # Set provider based on model
        provider = "openai" if "gpt" in best_model else "claude" if "claude" in best_model else "ollama"
        self.controller.switch_model(best_model, provider)
        
        return self.controller.invoke(text)

# Automatic cost optimization!
cost_optimizer = CostOptimizedLLM()
cheap_response = cost_optimizer.query("Simple question", max_cost=0.003)
premium_response = cost_optimizer.query("Complex analysis", max_cost=0.05)
```

### **6. Environment-Aware Deployment**
```python
def create_production_llm():
    """Create LLM based on environment"""
    if os.getenv("ENVIRONMENT") == "production":
        # Use reliable, fast models in production
        return LLMController(llm="claude-3-haiku", provider="claude")
    elif os.getenv("ENVIRONMENT") == "development":
        # Use local models for development
        return LLMController(llm="llama2", provider="ollama")
    else:
        # Use premium models for research
        return LLMController(llm="claude-3-sonnet", provider="claude")

# Same code, different models based on environment
llm = create_production_llm()
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    LLMController                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            Unified Interface                        │    │
│  │  • invoke()  • stream()  • batch()  • chains       │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                 │
│  ┌─────────────────────────┼─────────────────────────────┐  │
│  │         Provider Factory & Router                   │  │
│  └─────────────────────────┼─────────────────────────────┘  │
│                           │                                 │
│  ┌─────────┬──────────┬────┼────┬──────────┬─────────────┐   │
│  │ OpenAI  │  Claude  │ Grok   │  Ollama  │ HuggingFace │   │
│  │ GPT-4   │ Sonnet   │ Beta   │  Llama2  │   Models    │   │
│  │ GPT-3.5 │ Haiku    │        │ Mistral  │    etc.     │   │
│  └─────────┴──────────┴────────┴──────────┴─────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 How It Works

### 1. **Delegation Pattern**
LLMController acts as a **transparent proxy** that delegates all method calls to the currently active model:

```python
# When you call:
response = llm.invoke("Hello")

# LLMController does:
# 1. Routes to current model (e.g., Claude)
# 2. Calls claude_model.invoke("Hello")
# 3. Returns response unchanged
```

### 2. **Provider Factory System**
Each provider has a dedicated factory method that handles the specifics:

```python
def _create_claude_model(self, model_name: str):
    return ChatAnthropic(
        model=model_name,
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.7
    )

def _create_openai_model(self, model_name: str):
    return ChatOpenAI(
        model=model_name,
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )
```

### 3. **LangChain Runnable Interface**
Implements the full `Runnable` interface for seamless pipeline integration:

```python
# Full LangChain compatibility
prompt | llm | output_parser  # ✅ Works!
chain = RunnableLambda(preprocess) | llm | postprocess  # ✅ Works!
```

### 4. **Dynamic Model Switching**
Runtime switching without breaking existing chains:

```python
llm = LLMController(llm="claude-3-sonnet", provider="claude")
chain = prompt | llm | parser

# Later, switch the model but keep the same chain
llm.switch_model(llm="gpt-4", provider="openai")
# Chain still works, now using GPT-4!
```

## 📚 Supported Providers

| Provider | Models | Status | API Key Required |
|----------|---------|---------|------------------|
| **OpenAI** | GPT-4, GPT-3.5-turbo, etc. | ✅ Full Support | `OPENAI_API_KEY` |
| **Anthropic (Claude)** | Claude-3 (Opus, Sonnet, Haiku) | ✅ Full Support | `ANTHROPIC_API_KEY` |
| **Grok (X.AI)** | Grok-beta | ✅ Full Support | `XAI_API_KEY` |
| **Ollama** | Llama2, Mistral, CodeLlama, etc. | ✅ Full Support | None (local) |
| **Hugging Face** | Any HF model | ✅ Basic Support | `HUGGINGFACE_API_KEY` |

## 🛠️ Installation

```bash
# Core dependencies
pip install langchain-core langchain-community
pip install langchain-openai langchain-anthropic
pip install python-dotenv

# Optional: For specific providers
pip install transformers  # For Hugging Face
# Ollama: Install separately from https://ollama.ai
```

## 📖 Usage Examples

### Basic Usage

```python
from llm_controller import LLMController

# Initialize
llm = LLMController(llm="claude-3-sonnet-20240229", provider="claude")

# Simple query
response = llm.invoke("Explain quantum computing")
print(response.content)

# Switch models
llm.switch_model(llm="gpt-4", provider="openai")
response = llm.invoke("Same question, different model")
```

### With LangChain Chains

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Create a chain
prompt = ChatPromptTemplate.from_template("Explain {topic} simply")
chain = prompt | llm | StrOutputParser()

# Use the chain
result = chain.invoke({"topic": "machine learning"})

# Switch model mid-conversation
llm.switch_model(llm="claude-3-haiku-20240307", provider="claude")
# Same chain, now using Haiku for faster responses
```

### Streaming Responses

```python
# Streaming works the same across all providers
for chunk in llm.stream("Write a poem about AI"):
    print(chunk.content, end="", flush=True)
```

### Batch Processing

```python
# Process multiple inputs
inputs = ["Explain AI", "What is ML?", "Define NLP"]
responses = llm.batch(inputs)

for i, response in enumerate(responses):
    print(f"Q: {inputs[i]}")
    print(f"A: {response.content}\n")
```

### A/B Testing Different Models

```python
def compare_models(question, models):
    results = {}
    
    for provider, model in models.items():
        llm.switch_model(llm=model, provider=provider)
        response = llm.invoke(question)
        results[f"{provider}_{model}"] = response.content
    
    return results

# Compare responses
models = {
    "claude": "claude-3-sonnet-20240229",
    "openai": "gpt-4",
    "ollama": "llama2"
}

results = compare_models("What is the meaning of life?", models)
for model, response in results.items():
    print(f"\n{model}: {response[:100]}...")
```

### With LangChain Agents

```python
from langchain.agents import create_react_agent
from langchain.tools import DuckDuckGoSearchRun

# Create tools
search = DuckDuckGoSearchRun()
tools = [search]

# Create agent with LLMController
agent = create_react_agent(llm, tools, prompt_template)

# Switch to different model for different tasks
llm.switch_model(llm="gpt-4", provider="openai")  # Complex reasoning
result = agent.invoke({"input": "Research the latest AI trends"})

llm.switch_model(llm="claude-3-haiku", provider="claude")  # Fast responses
result = agent.invoke({"input": "Quick weather check"})
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file:

```bash
# Required for respective providers
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
XAI_API_KEY=your_grok_key_here
HUGGINGFACE_API_KEY=your_hf_key_here

# Optional configurations
OLLAMA_BASE_URL=http://localhost:11434  # Default Ollama URL
DEFAULT_TEMPERATURE=0.7
DEFAULT_MAX_TOKENS=1000
```

### Custom Model Configurations

```python
# Initialize with custom parameters
llm = LLMController(
    llm="claude-3-sonnet-20240229",
    provider="claude",
    temperature=0.9,
    max_tokens=2000
)

# Or configure after creation
llm._current_model.temperature = 0.5
```

## 🔍 Advanced Features

### Model Information

```python
# Get current model details
info = llm.current_model_info
print(f"Provider: {info['provider']}")
print(f"Model: {info['model']}")
print(f"Type: {info['type']}")
```

### Error Handling and Fallbacks

```python
def robust_query(question, fallback_models):
    for provider, model in fallback_models:
        try:
            llm.switch_model(llm=model, provider=provider)
            return llm.invoke(question)
        except Exception as e:
            print(f"Failed with {provider}/{model}: {e}")
            continue
    raise Exception("All models failed")

# Define fallback hierarchy
fallbacks = [
    ("claude", "claude-3-sonnet-20240229"),
    ("openai", "gpt-3.5-turbo"),
    ("ollama", "llama2")
]

response = robust_query("Explain AI", fallbacks)
```

### Custom Provider Integration

```python
# Extend LLMController for custom providers
class CustomLLMController(LLMController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_configs["custom_provider"] = self._create_custom_model
    
    def _create_custom_model(self, model_name: str):
        # Implement your custom provider
        return YourCustomModel(model=model_name)
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest test_llm_controller.py -v

# Run only unit tests (no API calls)
pytest test_llm_controller.py::TestLLMControllerUnit -v

# Run integration tests (requires API keys)
pytest test_llm_controller.py::TestLLMControllerIntegration -v
```

### Performance Testing

```python
import time

def benchmark_models(question, models, iterations=3):
    results = {}
    
    for provider, model in models.items():
        times = []
        llm.switch_model(llm=model, provider=provider)
        
        for _ in range(iterations):
            start = time.time()
            response = llm.invoke(question)
            end = time.time()
            times.append(end - start)
        
        results[f"{provider}_{model}"] = {
            "avg_time": sum(times) / len(times),
            "response_length": len(response.content)
        }
    
    return results
```

## 🚨 Common Issues & Solutions

### Import Errors

```bash
# Error: Cannot import LLMController
# Solution: Check LangChain installation
pip install langchain-core langchain-openai langchain-anthropic

# Error: Runnable not found
# Solution: Update LangChain
pip install --upgrade langchain-core
```

### API Key Issues

```python
# Check API keys are loaded
import os
print("OpenAI:", "✓" if os.getenv("OPENAI_API_KEY") else "✗")
print("Anthropic:", "✓" if os.getenv("ANTHROPIC_API_KEY") else "✗")

# Load .env file explicitly
from dotenv import load_dotenv
load_dotenv()
```

### Model Switching Issues

```python
# Issue: Chain breaks after switching
# Solution: Ensure model compatibility
try:
    llm.switch_model(llm="new-model", provider="new-provider")
    test_response = llm.invoke("test")
except Exception as e:
    print(f"Model switch failed: {e}")
    # Fallback to previous model
```

## 🔮 Future Enhancements

- [ ] **Automatic Fallbacks**: Intelligent provider switching on failures
- [ ] **Cost Optimization**: Route to cheapest model for simple queries
- [ ] **Response Caching**: Cache responses to reduce API calls
- [ ] **Model Analytics**: Track usage, performance, and costs
- [ ] **Async Operations**: Full async/await support
- [ ] **Plugin System**: Easy custom provider integration

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Add tests**: Ensure your changes are well-tested
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**

### Development Setup

```bash
# Clone and setup
git clone https://github.com/joshuamschultz/llm-controller.git
cd llm-controller

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** - For the amazing framework
- **Anthropic** - For Claude API
- **OpenAI** - For GPT models
- **Ollama** - For local model serving
- **Community Contributors** - For feedback and improvements

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/joshuamschultz/llm-controller/issues)
- **Documentation**: [Wiki](https://github.com/joshuamschultz/llm-controller/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/joshuamschultz/llm-controller/discussions)

---

**Made with ❤️ for the LangChain community**

*Simplifying LLM provider switching, one model at a time.*