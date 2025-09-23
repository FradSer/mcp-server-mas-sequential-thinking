"""Modernized configuration management with dependency injection and clean abstractions."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Type, runtime_checkable

from agno.models.anthropic import Claude
from agno.models.base import Model
from agno.models.deepseek import DeepSeek
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter


class GitHubOpenAI(OpenAIChat):
    """OpenAI provider configured for GitHub Models API with enhanced validation."""

    @staticmethod
    def _validate_github_token(token: str) -> None:
        """Validate GitHub token format with comprehensive checks."""
        if not token:
            raise ValueError("GitHub token is required but not provided")

        # Valid GitHub token prefixes
        valid_prefixes = ("ghp_", "github_pat_", "gho_", "ghu_")

        if not any(token.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid GitHub token format. Token must start with one of: {', '.join(valid_prefixes)}"
            )

        # Length validation for classic tokens
        if token.startswith("ghp_") and len(token) != 40:
            raise ValueError(
                "Invalid GitHub classic PAT length. Expected 40 characters."
            )

    def __init__(self, **kwargs) -> None:
        # Set GitHub Models configuration
        kwargs.setdefault("base_url", "https://models.github.ai/inference")

        if "api_key" not in kwargs:
            kwargs["api_key"] = os.environ.get("GITHUB_TOKEN")

        api_key = kwargs.get("api_key")
        if not api_key:
            raise ValueError(
                "GitHub token is required but not found in GITHUB_TOKEN environment variable"
            )

        self._validate_github_token(api_key)
        super().__init__(**kwargs)


@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for model provider and settings."""

    provider_class: Type[Model]
    enhanced_model_id: str
    standard_model_id: str
    api_key: Optional[str] = None

    def create_enhanced_model(self) -> Model:
        """Create enhanced model instance (used for complex synthesis like Blue Hat)."""
        # Enable prompt caching for Anthropic models
        if self.provider_class == Claude:
            return self.provider_class(
                id=self.enhanced_model_id,
                cache_system_prompt=True,  # Enable Anthropic prompt caching
            )
        return self.provider_class(id=self.enhanced_model_id)

    def create_standard_model(self) -> Model:
        """Create standard model instance (used for individual hat processing)."""
        # Enable prompt caching for Anthropic models
        if self.provider_class == Claude:
            return self.provider_class(
                id=self.standard_model_id,
                cache_system_prompt=True,  # Enable Anthropic prompt caching
            )
        return self.provider_class(id=self.standard_model_id)


@runtime_checkable
class ConfigurationStrategy(Protocol):
    """Protocol defining configuration strategy interface."""

    def get_config(self) -> ModelConfig:
        """Return model configuration for this strategy."""
        ...

    def get_required_environment_variables(self) -> Dict[str, bool]:
        """Return required environment variables and whether they're optional."""
        ...


class BaseProviderStrategy(ABC):
    """Abstract base strategy with common functionality."""

    @property
    @abstractmethod
    def provider_class(self) -> Type[Model]:
        """Return the provider model class."""

    @property
    @abstractmethod
    def default_enhanced_model(self) -> str:
        """Return default enhanced model ID (for complex synthesis)."""

    @property
    @abstractmethod
    def default_standard_model(self) -> str:
        """Return default standard model ID (for individual processing)."""

    @property
    @abstractmethod
    def api_key_name(self) -> Optional[str]:
        """Return API key environment variable name."""

    def _get_env_with_fallback(self, env_var: str, fallback: str) -> str:
        """Get environment variable with fallback to default."""
        value = os.environ.get(env_var, "").strip()
        return value if value else fallback

    def get_config(self) -> ModelConfig:
        """Get complete configuration with environment overrides."""
        prefix = self.__class__.__name__.replace("Strategy", "").upper()

        enhanced_model = self._get_env_with_fallback(
            f"{prefix}_ENHANCED_MODEL_ID", self.default_enhanced_model
        )
        standard_model = self._get_env_with_fallback(
            f"{prefix}_STANDARD_MODEL_ID", self.default_standard_model
        )

        # Get API key with enhanced validation and None handling
        api_key: Optional[str] = None
        if self.api_key_name:
            api_key = os.environ.get(self.api_key_name, "").strip()
            api_key = api_key if api_key else None

        return ModelConfig(
            provider_class=self.provider_class,
            enhanced_model_id=enhanced_model,
            standard_model_id=standard_model,
            api_key=api_key,
        )

    def get_required_environment_variables(self) -> Dict[str, bool]:
        """Return required environment variables."""
        env_vars: Dict[str, bool] = {}

        if self.api_key_name:
            env_vars[self.api_key_name] = False  # Required

        # Enhanced/standard environment variables (optional)
        prefix = self.__class__.__name__.replace("Strategy", "").upper()
        env_vars[f"{prefix}_ENHANCED_MODEL_ID"] = True  # Optional
        env_vars[f"{prefix}_STANDARD_MODEL_ID"] = True  # Optional

        return env_vars


# Concrete strategy implementations
class DeepSeekStrategy(BaseProviderStrategy):
    """DeepSeek provider strategy."""

    provider_class = DeepSeek
    default_enhanced_model = "deepseek-chat"
    default_standard_model = "deepseek-chat"
    api_key_name = "DEEPSEEK_API_KEY"


class GroqStrategy(BaseProviderStrategy):
    """Groq provider strategy."""

    provider_class = Groq
    default_enhanced_model = "deepseek-r1-distill-llama-70b"
    default_standard_model = "qwen/qwen3-32b"
    api_key_name = "GROQ_API_KEY"


class OpenRouterStrategy(BaseProviderStrategy):
    """OpenRouter provider strategy."""

    provider_class = OpenRouter
    default_enhanced_model = "deepseek/deepseek-chat-v3-0324"
    default_standard_model = "deepseek/deepseek-r1"
    api_key_name = "OPENROUTER_API_KEY"


class OllamaStrategy(BaseProviderStrategy):
    """Ollama provider strategy (no API key required)."""

    provider_class = Ollama
    default_enhanced_model = "devstral:24b"
    default_standard_model = "devstral:24b"
    api_key_name = None


class GitHubStrategy(BaseProviderStrategy):
    """GitHub Models provider strategy."""

    @property
    def provider_class(self) -> Type[Model]:
        return GitHubOpenAI

    @property
    def default_enhanced_model(self) -> str:
        return "openai/gpt-5"

    @property
    def default_standard_model(self) -> str:
        return "openai/gpt-5-min"

    @property
    def api_key_name(self) -> str:
        return "GITHUB_TOKEN"


class AnthropicStrategy(BaseProviderStrategy):
    """Anthropic Claude provider strategy with prompt caching enabled."""

    @property
    def provider_class(self) -> Type[Model]:
        return Claude

    @property
    def default_enhanced_model(self) -> str:
        return "claude-3-5-sonnet-20241022"

    @property
    def default_standard_model(self) -> str:
        return "claude-3-5-haiku-20241022"

    @property
    def api_key_name(self) -> str:
        return "ANTHROPIC_API_KEY"


class ConfigurationManager:
    """Manages configuration strategies with dependency injection."""

    def __init__(self) -> None:
        self._strategies = {
            "deepseek": DeepSeekStrategy(),
            "groq": GroqStrategy(),
            "openrouter": OpenRouterStrategy(),
            "ollama": OllamaStrategy(),
            "github": GitHubStrategy(),
            "anthropic": AnthropicStrategy(),
        }
        self._default_strategy = "deepseek"

    def register_strategy(self, name: str, strategy: ConfigurationStrategy) -> None:
        """Register a new configuration strategy."""
        self._strategies[name] = strategy

    def get_strategy(
        self, provider_name: Optional[str] = None
    ) -> ConfigurationStrategy:
        """Get strategy for specified provider."""
        if provider_name is None:
            provider_name = os.environ.get("LLM_PROVIDER", self._default_strategy)

        provider_name = provider_name.lower()

        if provider_name not in self._strategies:
            available = list(self._strategies.keys())
            raise ValueError(
                f"Unknown provider: {provider_name}. Available: {available}"
            )

        return self._strategies[provider_name]

    def get_model_config(self, provider_name: Optional[str] = None) -> ModelConfig:
        """Get model configuration using dependency injection."""
        strategy = self.get_strategy(provider_name)
        return strategy.get_config()

    def validate_environment(
        self, provider_name: Optional[str] = None
    ) -> Dict[str, str]:
        """Validate environment variables and return missing required ones."""
        strategy = self.get_strategy(provider_name)
        required_vars = strategy.get_required_environment_variables()

        missing: Dict[str, str] = {}
        for var_name, is_optional in required_vars.items():
            if not is_optional and not os.environ.get(var_name):
                missing[var_name] = "Required but not set"

        # Check EXA API key for research functionality (optional)
        # Note: EXA tools will be disabled if key is not provided
        exa_key = os.environ.get("EXA_API_KEY")
        if not exa_key:
            # Don't fail startup - just log warning that research will be disabled
            pass

        return missing

    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return list(self._strategies.keys())


# Singleton instance for dependency injection
_config_manager = ConfigurationManager()


# Public API functions
def get_model_config(provider_name: Optional[str] = None) -> ModelConfig:
    """Get model configuration using modernized configuration management."""
    return _config_manager.get_model_config(provider_name)


def check_required_api_keys(provider_name: Optional[str] = None) -> List[str]:
    """Check for missing required API keys."""
    missing_vars = _config_manager.validate_environment(provider_name)
    return list(missing_vars.keys())


def register_provider_strategy(name: str, strategy: ConfigurationStrategy) -> None:
    """Register a custom provider strategy."""
    _config_manager.register_strategy(name, strategy)


def get_available_providers() -> List[str]:
    """Get list of available providers."""
    return _config_manager.get_available_providers()
