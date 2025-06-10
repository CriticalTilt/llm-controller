# llm_controller.py
"""
LLMController - Universal LangChain Model Switcher
A unified interface for seamlessly switching between different LLM providers
while maintaining full LangChain compatibility.

Usage:
    from llm_controller import LLMController
    
    # Initialize with any provider
    llm = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
    response = llm.invoke("Hello!")
    
    # Switch providers seamlessly
    llm.switch_model(llm="gpt-4", provider="openai")
    response = llm.invoke("Same interface, different model!")
"""

import os
from typing import Dict, Any, Optional, Union, List, Iterator
import requests
import json

# Import LangChain components with fallbacks for different versions
try:
    # Try new LangChain structure first (0.1+)
    from langchain_community.llms import Ollama
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_core.language_models.base import BaseLanguageModel
    from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.language_models.llms import LLM
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.runnables import Runnable
    LANGCHAIN_NEW_STRUCTURE = True
except ImportError:
    # Fallback to legacy structure
    try:
        from langchain.llms import Ollama
        from langchain.chat_models import ChatOpenAI, ChatAnthropic
        from langchain.schema import BaseMessage, AIMessage, HumanMessage
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        from langchain.llms.base import LLM
        from langchain.chat_models.base import BaseChatModel
        try:
            from langchain_core.runnables import Runnable
        except ImportError:
            # Create a minimal Runnable base class for older versions
            class Runnable:
                def __init__(self, **kwargs):
                    pass
        BaseLanguageModel = BaseChatModel
        LANGCHAIN_NEW_STRUCTURE = False
    except ImportError as e:
        raise ImportError(
            "LangChain is required. Install with: "
            "pip install langchain-core langchain-openai langchain-anthropic"
        ) from e


class SimpleChain:
    """
    Simple chain implementation for fallback when RunnableSequence fails
    Provides basic pipeline functionality for older LangChain versions
    """
    
    def __init__(self, first, last):
        self.first = first
        self.last = last
    
    def invoke(self, input, config=None, **kwargs):
        """Run the chain: first component then second component"""
        try:
            intermediate = self.first.invoke(input, config, **kwargs)
            return self.last.invoke(intermediate, config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Chain execution failed: {e}") from e
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Async version of invoke"""
        try:
            intermediate = await self.first.ainvoke(input, config, **kwargs)
            return await self.last.ainvoke(intermediate, config, **kwargs)
        except Exception as e:
            raise RuntimeError(f"Async chain execution failed: {e}") from e
    
    def __or__(self, other):
        """Support chaining multiple components"""
        return SimpleChain(self, other)
    
    def __repr__(self):
        return f"SimpleChain({self.first} | {self.last})"


class LLMController(Runnable):
    """
    A unified LLM controller that provides seamless switching between providers
    while maintaining full LangChain compatibility for invoke(), LangGraph, etc.
    
    Supports: OpenAI, Anthropic (Claude), Grok (X.AI), Ollama, Hugging Face
    
    Example:
        controller = LLMController(llm="claude-3-sonnet-20240229", provider="claude")
        response = controller.invoke("Hello world!")
        
        # Switch providers
        controller.switch_model(llm="gpt-4", provider="openai")
        response = controller.invoke("Same interface!")
    """
    
    def __init__(self, llm: str = "gpt-3.5-turbo", provider: str = "openai", **kwargs):
        """
        Initialize the LLM Controller
        
        Args:
            llm: Model name (e.g., "claude-3-sonnet-20240229", "gpt-4")
            provider: Provider name ("claude", "openai", "ollama", "grok", "huggingface")
            **kwargs: Additional parameters passed to underlying models
        """
        # Only pass kwargs to super().__init__ if it accepts them
        try:
            # Check if parent class __init__ accepts parameters
            import inspect
            if hasattr(super(), '__init__'):
                sig = inspect.signature(super().__init__)
                if len(sig.parameters) > 1 or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                    super().__init__(**kwargs)
                else:
                    super().__init__()
        except (TypeError, ValueError):
            # Fallback: just call parent __init__ without parameters
            if hasattr(super(), '__init__'):
                super().__init__()
        
        self.llm_name = llm
        self.provider = provider
        self._model_kwargs = kwargs
        
        # Provider factory mapping
        self.model_configs = {
            "openai": self._create_openai_model,
            "claude": self._create_claude_model,
            "anthropic": self._create_claude_model,  # Alias for claude
            "grok": self._create_grok_model,
            "xai": self._create_grok_model,  # Alias for grok
            "ollama": self._create_ollama_model,
            "huggingface": self._create_huggingface_model,
            "hf": self._create_huggingface_model,  # Alias for huggingface
        }
        
        self._current_model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on provider and llm name"""
        if self.provider not in self.model_configs:
            available_providers = ", ".join(self.model_configs.keys())
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. "
                f"Available providers: {available_providers}"
            )
        
        try:
            self._current_model = self.model_configs[self.provider](self.llm_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def _create_openai_model(self, model_name: str) -> BaseLanguageModel:
        """Create OpenAI model"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "api_key": api_key,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Only add max_tokens if it's specified and not None
        max_tokens = self._model_kwargs.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add other parameters, excluding the ones we've already handled
        for k, v in self._model_kwargs.items():
            if k not in ["temperature", "max_tokens"] and v is not None:
                params[k] = v
        
        return ChatOpenAI(**params)
    
    def _create_claude_model(self, model_name: str) -> BaseLanguageModel:
        """Create Anthropic Claude model"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "api_key": api_key,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Only add max_tokens if it's specified and not None
        max_tokens = self._model_kwargs.get("max_tokens")
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        # Add other parameters, excluding the ones we've already handled
        for k, v in self._model_kwargs.items():
            if k not in ["temperature", "max_tokens"] and v is not None:
                params[k] = v
        
        return ChatAnthropic(**params)
    
    def _create_ollama_model(self, model_name: str) -> BaseLanguageModel:
        """Create Ollama model (local)"""
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Build parameters, excluding None values
        params = {
            "model": model_name,
            "base_url": base_url,
            "temperature": self._model_kwargs.get("temperature", 0.7),
        }
        
        # Add other parameters, excluding temperature and None values
        for k, v in self._model_kwargs.items():
            if k != "temperature" and v is not None:
                params[k] = v
        
        return Ollama(**params)
    
    def _create_grok_model(self, model_name: str) -> BaseLanguageModel:
        """Create Grok (X.AI) model"""
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required")
        
        return GrokChatModel(
            model_name=model_name,
            api_key=api_key,
            **self._model_kwargs
        )
    
    def _create_huggingface_model(self, model_name: str) -> BaseLanguageModel:
        """Create Hugging Face model"""
        api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY environment variable is required")
        
        return HuggingFaceChatModel(
            model_name=model_name,
            api_key=api_key,
            **self._model_kwargs
        )
    
    def switch_model(self, llm: str, provider: str = None, **kwargs):
        """
        Switch to a different model/provider
        
        Args:
            llm: New model name
            provider: New provider (optional, keeps current if not specified)
            **kwargs: Additional model parameters to update
        """
        if provider:
            self.provider = provider
        self.llm_name = llm
        
        # Update model kwargs if provided
        if kwargs:
            self._model_kwargs.update(kwargs)
        
        self._initialize_model()
    
    # Core Runnable interface methods
    def invoke(self, input, config=None, **kwargs):
        """
        Invoke the current model
        
        Args:
            input: Input prompt or messages
            config: Optional configuration
            **kwargs: Additional parameters
            
        Returns:
            Model response
        """
        try:
            response = self._current_model.invoke(input, config, **kwargs)
            
            # Handle Ollama's string response - wrap it in AIMessage for consistency
            if self.provider == "ollama" and isinstance(response, str):
                return AIMessage(content=response)
            
            return response
        except Exception as e:
            raise RuntimeError(
                f"Error invoking {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def ainvoke(self, input, config=None, **kwargs):
        """Async version of invoke"""
        try:
            if hasattr(self._current_model, 'ainvoke'):
                response = await self._current_model.ainvoke(input, config, **kwargs)
                
                # Handle Ollama's string response - wrap it in AIMessage for consistency
                if self.provider == "ollama" and isinstance(response, str):
                    return AIMessage(content=response)
                
                return response
            else:
                # Fallback for models that don't support async
                import asyncio
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: self.invoke(input, config, **kwargs)
                )
        except Exception as e:
            raise RuntimeError(
                f"Error in async invoke for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def stream(self, input, config=None, **kwargs) -> Iterator[Any]:
        """
        Stream responses from the current model
        
        Args:
            input: Input prompt or messages
            config: Optional configuration
            **kwargs: Additional parameters
            
        Yields:
            Response chunks
        """
        try:
            for chunk in self._current_model.stream(input, config, **kwargs):
                # Handle Ollama's string chunks - wrap them in AIMessage for consistency
                if self.provider == "ollama" and isinstance(chunk, str):
                    yield AIMessage(content=chunk)
                else:
                    yield chunk
        except Exception as e:
            raise RuntimeError(
                f"Error streaming from {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def astream(self, input, config=None, **kwargs):
        """Async version of stream"""
        try:
            if hasattr(self._current_model, 'astream'):
                async for chunk in self._current_model.astream(input, config, **kwargs):
                    yield chunk
            else:
                # Fallback: convert sync stream to async
                for chunk in self.stream(input, config, **kwargs):
                    yield chunk
        except Exception as e:
            raise RuntimeError(
                f"Error in async stream for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    def batch(self, inputs, config=None, **kwargs):
        """
        Process multiple inputs in batch
        
        Args:
            inputs: List of inputs
            config: Optional configuration
            **kwargs: Additional parameters
            
        Returns:
            List of responses
        """
        try:
            if hasattr(self._current_model, 'batch'):
                responses = self._current_model.batch(inputs, config, **kwargs)
                
                # Handle Ollama's string responses - wrap them in AIMessage for consistency
                if self.provider == "ollama":
                    processed_responses = []
                    for response in responses:
                        if isinstance(response, str):
                            processed_responses.append(AIMessage(content=response))
                        else:
                            processed_responses.append(response)
                    return processed_responses
                
                return responses
            else:
                # Fallback: process one by one
                return [self.invoke(input, config, **kwargs) for input in inputs]
        except Exception as e:
            raise RuntimeError(
                f"Error in batch processing for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    async def abatch(self, inputs, config=None, **kwargs):
        """Async version of batch"""
        try:
            if hasattr(self._current_model, 'abatch'):
                return await self._current_model.abatch(inputs, config, **kwargs)
            else:
                # Fallback: process all async
                import asyncio
                tasks = [self.ainvoke(input, config, **kwargs) for input in inputs]
                return await asyncio.gather(*tasks)
        except Exception as e:
            raise RuntimeError(
                f"Error in async batch for {self.provider} model '{self.llm_name}': {e}"
            ) from e
    
    # Pipeline operator support for LangChain chains
    def __or__(self, other):
        """Support for | operator: controller | parser"""
        try:
            # Try to use LangChain's RunnableSequence if available
            if LANGCHAIN_NEW_STRUCTURE:
                from langchain_core.runnables import RunnableSequence
                return RunnableSequence(first=self, last=other)
            else:
                return SimpleChain(self, other)
        except (ImportError, Exception):
            # Fall back to SimpleChain if RunnableSequence fails
            return SimpleChain(self, other)
    
    def __ror__(self, other):
        """Support for | operator: prompt | controller"""
        try:
            # Try to use LangChain's RunnableSequence if available
            if LANGCHAIN_NEW_STRUCTURE:
                from langchain_core.runnables import RunnableSequence
                return RunnableSequence(first=other, last=self)
            else:
                return SimpleChain(other, self)
        except (ImportError, Exception):
            # Fall back to SimpleChain if RunnableSequence fails
            return SimpleChain(other, self)
    
    # Method delegation for backwards compatibility
    def __getattr__(self, name):
        """
        Delegate any missing methods to the current model
        This ensures compatibility with all LangChain features
        """
        if self._current_model and hasattr(self._current_model, name):
            attr = getattr(self._current_model, name)
            # If it's a method, wrap it to maintain error context
            if callable(attr):
                def wrapper(*args, **kwargs):
                    try:
                        return attr(*args, **kwargs)
                    except Exception as e:
                        raise RuntimeError(
                            f"Error calling {name} on {self.provider} model: {e}"
                        ) from e
                return wrapper
            return attr
        
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
    
    @property
    def current_model_info(self) -> Dict[str, str]:
        """Get current model information"""
        return {
            "provider": self.provider,
            "model": self.llm_name,
            "type": type(self._current_model).__name__,
            "langchain_structure": "new" if LANGCHAIN_NEW_STRUCTURE else "legacy"
        }
    
    @property
    def _llm_type(self) -> str:
        """Return the type of LLM for LangChain compatibility"""
        return f"llm_controller_{self.provider}"
    
    def __repr__(self):
        return f"LLMController(provider='{self.provider}', model='{self.llm_name}')"
    
    def __str__(self):
        return f"LLMController[{self.provider}:{self.llm_name}]"


class GrokChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Grok (X.AI) API"""
    
    def __init__(self, model_name: str = "grok-beta", api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.api_url = "https://api.x.ai/v1/chat/completions"
        
        if not self.api_key:
            raise ValueError("XAI_API_KEY is required for Grok models")
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        # Convert LangChain messages to API format
        api_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                api_messages.append({"role": "assistant", "content": msg.content})
            else:
                api_messages.append({"role": "user", "content": str(msg.content)})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": api_messages,
            "model": self.model_name,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Grok API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Grok API response format: {e}") from e
    
    @property
    def _llm_type(self) -> str:
        return "grok_chat"


class HuggingFaceChatModel(BaseChatModel):
    """Custom LangChain chat model wrapper for Hugging Face Inference API"""
    
    def __init__(self, model_name: str, api_key: str = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY is required for Hugging Face models")
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Any:
        from langchain_core.outputs import ChatGeneration, ChatResult
        
        # Combine messages into a single prompt for non-chat models
        prompt = "\n".join([f"{msg.__class__.__name__}: {msg.content}" for msg in messages])
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 256),
                "temperature": kwargs.get("temperature", 0.7),
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            else:
                content = str(result)
            
            message = AIMessage(content=content)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Hugging Face API request failed: {e}") from e
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Hugging Face API response format: {e}") from e
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_chat"


class AdaptiveLLM:
    """
    Dynamic LLM that adapts model selection based on task type.
    Automatically chooses the optimal model for different types of tasks:
    - Creative tasks -> Claude Sonnet (best for creative writing)
    - Coding tasks -> GPT-4 (excellent for code generation/debugging)
    - Simple tasks -> GPT-3.5-turbo (fast and cost-effective)
    - Local tasks -> Ollama models (private and free)
    """
    
    def __init__(self, default_model: str = "gpt-3.5-turbo", default_provider: str = "openai", **kwargs):
        """
        Initialize AdaptiveLLM with a default model configuration
        
        Args:
            default_model: Default model to use when task type is not specified
            default_provider: Default provider for the default model
            **kwargs: Additional parameters passed to LLMController
        """
        self.controller = LLMController(llm=default_model, provider=default_provider, **kwargs)
        
        # Task-to-model mapping - can be customized
        self.task_models = {
            "creative": ("claude-3-sonnet-20240229", "claude"),
            "coding": ("gpt-4", "openai"),
            "simple": ("gpt-3.5-turbo", "openai"),
            "local": ("llama2", "ollama"),
            "general": (default_model, default_provider)
        }
    
    def query(self, text: str, task_type: str = "general", **kwargs):
        """
        Query with automatic model selection based on task type
        
        Args:
            text: Input text/prompt
            task_type: Type of task ("creative", "coding", "simple", "local", "general")
            **kwargs: Additional parameters passed to invoke()
            
        Returns:
            Model response
        """
        if task_type in self.task_models:
            model, provider = self.task_models[task_type]
            self.controller.switch_model(model, provider)
        else:
            # Use general model for unknown task types
            model, provider = self.task_models["general"]
            self.controller.switch_model(model, provider)
        
        return self.controller.invoke(text, **kwargs)
    
    def set_task_model(self, task_type: str, model: str, provider: str):
        """
        Customize the model used for a specific task type
        
        Args:
            task_type: Task type to configure
            model: Model name to use for this task type
            provider: Provider name for the model
        """
        self.task_models[task_type] = (model, provider)
    
    def get_current_model_info(self) -> Dict[str, str]:
        """Get information about the currently active model"""
        return self.controller.current_model_info
    
    def __getattr__(self, name):
        """Delegate other methods to the underlying controller"""
        return getattr(self.controller, name)


class CostOptimizedLLM:
    """
    Cost-aware LLM that automatically selects the most expensive model within budget.
    Maintains a cost database and routes queries to the best model you can afford.
    """
    
    def __init__(self, default_model: str = "gpt-3.5-turbo", default_provider: str = "openai", **kwargs):
        """
        Initialize CostOptimizedLLM
        
        Args:
            default_model: Default model to use
            default_provider: Default provider
            **kwargs: Additional parameters passed to LLMController
        """
        self.controller = LLMController(llm=default_model, provider=default_provider, **kwargs)
        
        # Cost per 1K tokens (approximate, in USD) - update with current pricing
        self.costs = {
            # OpenAI models
            "gpt-4": 0.03,
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            
            # Anthropic models
            "claude-3-opus-20240229": 0.075,
            "claude-3-sonnet-20240229": 0.015,
            "claude-3-haiku-20240307": 0.0025,
            
            # Local/Free models
            "llama2": 0.0,
            "mistral": 0.0,
            "codellama": 0.0,
            
            # Grok
            "grok-beta": 0.01,  # Estimated
        }
        
        # Provider mapping for cost optimization
        self.model_providers = {
            # OpenAI
            "gpt-4": "openai",
            "gpt-4-turbo": "openai", 
            "gpt-3.5-turbo": "openai",
            
            # Anthropic
            "claude-3-opus-20240229": "claude",
            "claude-3-sonnet-20240229": "claude",
            "claude-3-haiku-20240307": "claude",
            
            # Ollama (local)
            "llama2": "ollama",
            "mistral": "ollama",
            "codellama": "ollama",
            
            # Grok
            "grok-beta": "grok",
        }
    
    def query(self, text: str, max_cost: float = 0.01, **kwargs):
        """
        Query using the best model within the specified budget
        
        Args:
            text: Input text/prompt
            max_cost: Maximum cost per 1K tokens (USD)
            **kwargs: Additional parameters passed to invoke()
            
        Returns:
            Model response
            
        Raises:
            ValueError: If no models are available within budget
        """
        # Find affordable models
        # Special case: max_cost of 0.0 means no budget, so no models are affordable
        if max_cost <= 0.0:
            affordable_models = []
        else:
            affordable_models = [
                (model, cost) for model, cost in self.costs.items() 
                if cost <= max_cost and model in self.model_providers
            ]
        
        if not affordable_models:
            available_costs = [f"{model}: ${cost}" for model, cost in self.costs.items()]
            raise ValueError(
                f"No models available under ${max_cost}. "
                f"Available models and costs: {available_costs}"
            )
        
        # Pick the most expensive model we can afford (best quality within budget)
        best_model, best_cost = max(affordable_models, key=lambda x: x[1])
        provider = self.model_providers[best_model]
        
        # Switch to the optimal model
        self.controller.switch_model(best_model, provider)
        
        return self.controller.invoke(text, **kwargs)
    
    def estimate_cost(self, text: str, model: str = None) -> float:
        """
        Estimate the cost for a query with a specific model
        
        Args:
            text: Input text to estimate cost for
            model: Model name (uses current model if not specified)
            
        Returns:
            Estimated cost in USD
        """
        if model is None:
            model = self.controller.llm_name
        
        if model not in self.costs:
            return 0.0
        
        # Rough token estimation (1 token ≈ 4 characters)
        estimated_tokens = len(text) / 4
        cost_per_1k = self.costs[model]
        
        return (estimated_tokens / 1000) * cost_per_1k
    
    def get_affordable_models(self, max_cost: float) -> List[Dict[str, Any]]:
        """
        Get list of models within budget, sorted by quality (cost)
        
        Args:
            max_cost: Maximum cost per 1K tokens
            
        Returns:
            List of model information dictionaries
        """
        affordable = []
        for model, cost in self.costs.items():
            if cost <= max_cost and model in self.model_providers:
                affordable.append({
                    "model": model,
                    "provider": self.model_providers[model],
                    "cost_per_1k": cost,
                    "quality_rank": cost  # Higher cost = better quality (generally)
                })
        
        # Sort by quality (cost) descending
        return sorted(affordable, key=lambda x: x["quality_rank"], reverse=True)
    
    def update_cost(self, model: str, cost_per_1k: float):
        """
        Update the cost for a specific model
        
        Args:
            model: Model name
            cost_per_1k: Cost per 1K tokens in USD
        """
        self.costs[model] = cost_per_1k
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the currently active model including cost"""
        info = self.controller.current_model_info
        model = info["model"]
        info["cost_per_1k"] = self.costs.get(model, 0.0)
        return info
    
    def __getattr__(self, name):
        """Delegate other methods to the underlying controller"""
        return getattr(self.controller, name)


def create_production_llm(**kwargs) -> LLMController:
    """
    Create LLM configuration based on environment variables.
    Automatically selects appropriate models for different deployment environments:
    
    - PRODUCTION: Reliable, fast models (Claude Haiku)
    - DEVELOPMENT: Local models for cost-free development (Ollama)  
    - STAGING: Balanced performance and cost (GPT-3.5-turbo)
    - RESEARCH: Premium models for best results (Claude Sonnet)
    
    Environment Variables:
        ENVIRONMENT: deployment environment (production/development/staging/research)
        LLM_MODEL: override model selection
        LLM_PROVIDER: override provider selection
        
    Args:
        **kwargs: Additional parameters passed to LLMController
        
    Returns:
        Configured LLMController instance
        
    Example:
        # Set environment
        os.environ["ENVIRONMENT"] = "production"
        llm = create_production_llm()
        
        # Override with specific model
        os.environ["LLM_MODEL"] = "gpt-4"
        os.environ["LLM_PROVIDER"] = "openai"
        llm = create_production_llm()
    """
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    # Allow manual override via environment variables
    override_model = os.getenv("LLM_MODEL")
    override_provider = os.getenv("LLM_PROVIDER")
    
    if override_model and override_provider:
        return LLMController(llm=override_model, provider=override_provider, **kwargs)
    
    # Environment-based model selection
    env_configs = {
        "production": {
            "model": "claude-3-haiku-20240307",
            "provider": "claude",
            "reason": "Fast, reliable, and cost-effective for production workloads"
        },
        "development": {
            "model": "llama2", 
            "provider": "ollama",
            "reason": "Local model for cost-free development and testing"
        },
        "staging": {
            "model": "gpt-3.5-turbo",
            "provider": "openai", 
            "reason": "Good balance of performance and cost for staging"
        },
        "research": {
            "model": "claude-3-sonnet-20240229",
            "provider": "claude",
            "reason": "Premium model for research and high-quality outputs"
        },
        "testing": {
            "model": "gpt-3.5-turbo",
            "provider": "openai",
            "reason": "Consistent and reliable for automated testing"
        }
    }
    
    # Default to development if environment not recognized
    config = env_configs.get(environment, env_configs["development"])
    
    try:
        controller = LLMController(
            llm=config["model"], 
            provider=config["provider"], 
            **kwargs
        )
        
        # Log the configuration choice (optional)
        if os.getenv("DEBUG", "").lower() in ["true", "1", "yes"]:
            print(f"Environment: {environment}")
            print(f"Selected: {config['provider']}/{config['model']}")
            print(f"Reason: {config['reason']}")
        
        return controller
        
    except Exception as e:
        # Fallback to a basic configuration if the environment-specific one fails
        print(f"Warning: Failed to create {environment} LLM ({e}), falling back to basic config")
        return LLMController(llm="gpt-3.5-turbo", provider="openai", **kwargs)


def create_adaptive_llm(**kwargs) -> AdaptiveLLM:
    """
    Convenience function to create an AdaptiveLLM instance
    
    Args:
        **kwargs: Parameters passed to AdaptiveLLM constructor
        
    Returns:
        Configured AdaptiveLLM instance
    """
    return AdaptiveLLM(**kwargs)


def create_cost_optimized_llm(**kwargs) -> CostOptimizedLLM:
    """
    Convenience function to create a CostOptimizedLLM instance
    
    Args:
        **kwargs: Parameters passed to CostOptimizedLLM constructor
        
    Returns:
        Configured CostOptimizedLLM instance
    """
    return CostOptimizedLLM(**kwargs)


# Convenience functions for quick setup
def create_controller(provider: str, model: str = None, **kwargs) -> LLMController:
    """
    Convenience function to create a controller with sensible defaults
    
    Args:
        provider: Provider name ("claude", "openai", "ollama", etc.)
        model: Model name (uses provider default if not specified)
        **kwargs: Additional parameters
    
    Returns:
        Configured LLMController
    """
    # Provider defaults
    defaults = {
        "claude": "claude-3-sonnet-20240229",
        "openai": "gpt-3.5-turbo",
        "ollama": "llama2",
        "grok": "grok-beta",
        "huggingface": "microsoft/DialoGPT-medium"
    }
    
    if not model:
        model = defaults.get(provider)
        if not model:
            raise ValueError(f"No default model for provider '{provider}'. Please specify a model.")
    
    return LLMController(llm=model, provider=provider, **kwargs)


# Example usage
if __name__ == "__main__":
    print("LLMController - Universal LangChain Model Switcher")
    print("=" * 60)
    
    try:
        # Example with environment variables
        from dotenv import load_dotenv
        load_dotenv()
        
        print("\n1. Basic LLMController Example")
        print("-" * 40)
        controller = create_controller("claude", temperature=0.8)
        print(f"Created: {controller}")
        print(f"Model info: {controller.current_model_info}")
        
        # Test basic functionality
        response = controller.invoke("Say hello in one word.")
        print(f"Response: {response.content}")
        
        # Switch models
        controller.switch_model("gpt-3.5-turbo", "openai")
        print(f"Switched to: {controller}")
        
        print("\n2. AdaptiveLLM Example")
        print("-" * 40)
        adaptive = create_adaptive_llm()
        print("AdaptiveLLM created - automatically chooses optimal models for tasks")
        
        # Test different task types
        creative_response = adaptive.query("Write a haiku about programming", "creative")
        print(f"Creative task (using {adaptive.get_current_model_info()['model']}): {creative_response.content[:100]}...")
        
        coding_response = adaptive.query("Fix this Python code: def hello() print('hi')", "coding")
        print(f"Coding task (using {adaptive.get_current_model_info()['model']}): {coding_response.content[:100]}...")
        
        simple_response = adaptive.query("What's 2+2?", "simple")
        print(f"Simple task (using {adaptive.get_current_model_info()['model']}): {simple_response.content[:50]}...")
        
        print("\n3. CostOptimizedLLM Example")
        print("-" * 40)
        cost_optimizer = create_cost_optimized_llm()
        print("CostOptimizedLLM created - automatically stays within budget")
        
        # Test with different budgets
        affordable_models = cost_optimizer.get_affordable_models(max_cost=0.01)
        print(f"Models under $0.01: {[m['model'] for m in affordable_models[:3]]}")
        
        cheap_response = cost_optimizer.query("Simple question: What is AI?", max_cost=0.003)
        info = cost_optimizer.get_current_model_info()
        print(f"Budget $0.003 used: {info['model']} (${info['cost_per_1k']}/1k tokens)")
        
        premium_response = cost_optimizer.query("Complex analysis of machine learning", max_cost=0.05)
        info = cost_optimizer.get_current_model_info()
        print(f"Budget $0.05 used: {info['model']} (${info['cost_per_1k']}/1k tokens)")
        
        print("\n4. Environment-Aware Deployment Example")
        print("-" * 40)
        
        # Test different environments
        import os
        
        # Simulate production environment
        os.environ["ENVIRONMENT"] = "production"
        prod_llm = create_production_llm()
        print(f"Production LLM: {prod_llm.current_model_info}")
        
        # Simulate development environment
        os.environ["ENVIRONMENT"] = "development"
        dev_llm = create_production_llm()
        print(f"Development LLM: {dev_llm.current_model_info}")
        
        # Override with specific model
        os.environ["LLM_MODEL"] = "gpt-4"
        os.environ["LLM_PROVIDER"] = "openai"
        override_llm = create_production_llm()
        print(f"Override LLM: {override_llm.current_model_info}")
        
        print("\n✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("Make sure you have API keys set in your environment variables:")
        print("- OPENAI_API_KEY (for OpenAI models)")
        print("- ANTHROPIC_API_KEY (for Claude models)")
        print("- Or use Ollama for local models (no API key needed)")