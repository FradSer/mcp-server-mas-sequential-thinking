"""Comprehensive tests for the configuration module."""

from unittest.mock import MagicMock

import pytest

from mcp_server_mas_sequential_thinking.modernized_config import (
    DeepSeekStrategy,
    GitHubOpenAI,
    GitHubStrategy,
    GroqStrategy,
    ModelConfig,
    OllamaStrategy,
    OpenRouterStrategy,
    check_required_api_keys,
    get_model_config,
)
from tests.helpers.mocks import MockEnvironment


class TestProviderStrategies:
    """Test all provider strategy implementations."""

    @pytest.mark.parametrize(
        ("provider_name", "strategy_class"),
        [
            ("deepseek", DeepSeekStrategy),
            ("groq", GroqStrategy),
            ("openrouter", OpenRouterStrategy),
            ("ollama", OllamaStrategy),
            ("github", GitHubStrategy),
        ],
    )
    def test_strategy_configuration(self, provider_name, strategy_class):
        """Test strategy configuration consistency."""
        strategy = strategy_class()
        config = strategy.get_config()

        assert isinstance(config, ModelConfig)
        assert config.provider_class is not None
        assert config.team_model_id is not None
        assert config.agent_model_id is not None

    def test_environment_variable_precedence(self):
        """Test environment variable override precedence."""
        strategy = DeepSeekStrategy()

        with MockEnvironment(
            {
                "DEEPSEEK_TEAM_MODEL_ID": "custom-team-model",
                "DEEPSEEK_AGENT_MODEL_ID": "custom-agent-model",
                "DEEPSEEK_API_KEY": "custom-key",
            }
        ):
            config = strategy.get_config()
            assert config.team_model_id == "custom-team-model"
            assert config.agent_model_id == "custom-agent-model"
            assert config.api_key == "custom-key"

    def test_fallback_to_defaults(self):
        """Test fallback to default values when env vars are empty."""
        strategy = DeepSeekStrategy()

        with MockEnvironment(
            {
                "DEEPSEEK_TEAM_MODEL_ID": "",  # Empty string should fallback
                "DEEPSEEK_AGENT_MODEL_ID": "",  # Empty string should fallback
                "DEEPSEEK_API_KEY": "",  # Empty string should be None
            }
        ):
            config = strategy.get_config()
            assert config.team_model_id == strategy.default_team_model
            assert config.agent_model_id == strategy.default_agent_model
            assert config.api_key is None

    def test_strategy_registry_completeness(self):
        """Test that all strategies are registered correctly."""
        expected_providers = ["deepseek", "groq", "openrouter", "ollama", "github"]

        for provider in expected_providers:
            assert provider in STRATEGIES
            assert isinstance(STRATEGIES[provider], type(STRATEGIES[provider]))


class TestAPIKeyValidation:
    """Test API key requirement checking."""

    def test_missing_api_keys_detection(self):
        """Test detection of missing required API keys."""
        with MockEnvironment({"LLM_PROVIDER": "deepseek"}):
            missing_keys = check_required_api_keys()
            assert "DEEPSEEK_API_KEY" in missing_keys
            assert "EXA_API_KEY" in missing_keys

    def test_provider_specific_key_requirements(self):
        """Test provider-specific API key requirements."""
        with MockEnvironment({"LLM_PROVIDER": "github"}):
            missing_keys = check_required_api_keys()
            assert "GITHUB_TOKEN" in missing_keys
            assert "EXA_API_KEY" in missing_keys

    def test_ollama_no_api_key_required(self):
        """Test that Ollama doesn't require API key."""
        with MockEnvironment({"LLM_PROVIDER": "ollama", "EXA_API_KEY": "test-exa-key"}):
            missing_keys = check_required_api_keys()
            assert "OLLAMA_API_KEY" not in missing_keys
            assert len(missing_keys) == 0

    def test_all_keys_present(self):
        """Test when all required keys are present."""
        with MockEnvironment(
            {
                "LLM_PROVIDER": "deepseek",
                "DEEPSEEK_API_KEY": "test-key",
                "EXA_API_KEY": "test-exa-key",
            }
        ):
            missing_keys = check_required_api_keys()
            assert len(missing_keys) == 0


class TestGitHubProvider:
    """Test GitHub Models provider implementation."""

    @pytest.mark.parametrize(
        "valid_token",
        [
            "ghp_1234567890123456789012345678901234567890",  # Classic PAT (40 chars)
            "github_pat_11ABCDEFG0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
            "gho_16C7e42F292c6912E7710c838347Ae178B4a",  # OAuth token
            "ghu_16C7e42F292c6912E7710c838347Ae178B4a",  # User-to-server token
        ],
    )
    def test_valid_github_token_formats(self, valid_token):
        """Test validation of valid GitHub token formats."""
        with MockEnvironment({"GITHUB_TOKEN": valid_token}):
            # Should not raise an exception
            provider = GitHubOpenAI()
            assert provider.api_key == valid_token

    @pytest.mark.parametrize(
        "invalid_token",
        [
            "invalid_token_format",
            "sk-1234567890123456789012345678901234567890",  # OpenAI format
            "ghp_123",  # Too short for classic PAT
            "",  # Empty
            "ghp_12345678901234567890123456789012345678901",  # Too long for classic PAT
        ],
    )
    def test_invalid_github_token_formats(self, invalid_token):
        """Test validation of invalid GitHub token formats."""
        with MockEnvironment({"GITHUB_TOKEN": invalid_token}):
            with pytest.raises(ValueError):
                GitHubOpenAI()

    def test_github_token_missing(self):
        """Test handling of missing GitHub token."""
        with MockEnvironment({}):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubOpenAI()

    def test_github_provider_base_url(self):
        """Test that GitHub provider uses correct base URL."""
        with MockEnvironment(
            {"GITHUB_TOKEN": "ghp_1234567890123456789012345678901234567890"}
        ):
            provider = GitHubOpenAI()
            assert provider.base_url == "https://models.github.ai/inference"


class TestModelConfigurationFlow:
    """Test the complete model configuration flow."""

    def test_get_model_config_default_provider(self):
        """Test getting model config with default provider."""
        with MockEnvironment(
            {"DEEPSEEK_API_KEY": "test-key", "EXA_API_KEY": "test-exa-key"}
        ):
            config = get_model_config()
            assert config.provider_class == DeepSeekStrategy.provider_class
            assert config.team_model_id == "deepseek-chat"
            assert config.agent_model_id == "deepseek-chat"

    def test_get_model_config_specific_provider(self):
        """Test getting model config with specific provider."""
        with MockEnvironment(
            {
                "LLM_PROVIDER": "groq",
                "GROQ_API_KEY": "test-key",
                "EXA_API_KEY": "test-exa-key",
            }
        ):
            config = get_model_config()
            assert config.provider_class == GroqStrategy.provider_class
            assert "deepseek-r1-distill-llama-70b" in config.team_model_id
            assert "qwen-2.5-32b" in config.agent_model_id

    def test_get_model_config_invalid_provider_fallback(self):
        """Test fallback to default provider for invalid provider name."""
        with MockEnvironment(
            {"LLM_PROVIDER": "invalid_provider", "DEEPSEEK_API_KEY": "test-key"}
        ):
            config = get_model_config()
            # Should fallback to deepseek
            assert config.provider_class == DeepSeekStrategy.provider_class

    def test_provider_case_insensitive(self):
        """Test that provider names are case insensitive."""
        with MockEnvironment(
            {"LLM_PROVIDER": "DEEPSEEK", "DEEPSEEK_API_KEY": "test-key"}
        ):
            config = get_model_config()
            assert config.provider_class == DeepSeekStrategy.provider_class


class TestProviderStrategyDetails:
    """Test specific details of each provider strategy."""

    def test_deepseek_strategy_details(self):
        """Test DeepSeek strategy configuration."""
        strategy = DeepSeekStrategy()
        assert strategy.default_team_model == "deepseek-chat"
        assert strategy.default_agent_model == "deepseek-chat"
        assert strategy.api_key_name == "DEEPSEEK_API_KEY"

    def test_groq_strategy_details(self):
        """Test Groq strategy configuration."""
        strategy = GroqStrategy()
        assert "deepseek-r1-distill-llama-70b" in strategy.default_team_model
        assert "qwen-2.5-32b" in strategy.default_agent_model
        assert strategy.api_key_name == "GROQ_API_KEY"

    def test_openrouter_strategy_details(self):
        """Test OpenRouter strategy configuration."""
        strategy = OpenRouterStrategy()
        assert "deepseek/deepseek-chat-v3-0324" in strategy.default_team_model
        assert "deepseek/deepseek-r1" in strategy.default_agent_model
        assert strategy.api_key_name == "OPENROUTER_API_KEY"

    def test_ollama_strategy_details(self):
        """Test Ollama strategy configuration."""
        strategy = OllamaStrategy()
        assert "devstral:24b" in strategy.default_team_model
        assert "devstral:24b" in strategy.default_agent_model
        assert strategy.api_key_name is None

    def test_github_strategy_details(self):
        """Test GitHub strategy configuration."""
        strategy = GitHubStrategy()
        assert "openai/gpt-5" in strategy.default_team_model
        assert "openai/gpt-5-min" in strategy.default_agent_model
        assert strategy.api_key_name == "GITHUB_TOKEN"


class TestModelConfigDataclass:
    """Test ModelConfig dataclass functionality."""

    def test_model_config_creation(self):
        """Test ModelConfig creation with all fields."""
        config = ModelConfig(
            provider_class=MagicMock,
            team_model_id="test-team-model",
            agent_model_id="test-agent-model",
            api_key="test-key",
        )

        assert config.provider_class == MagicMock
        assert config.team_model_id == "test-team-model"
        assert config.agent_model_id == "test-agent-model"
        assert config.api_key == "test-key"

    def test_model_config_immutability(self):
        """Test that ModelConfig is immutable."""
        config = ModelConfig(
            provider_class=MagicMock,
            team_model_id="test-team-model",
            agent_model_id="test-agent-model",
        )

        with pytest.raises(
            Exception
        ):  # dataclass(frozen=True) should prevent modification
            config.team_model_id = "modified-model"

    def test_model_config_optional_api_key(self):
        """Test ModelConfig with optional API key."""
        config = ModelConfig(
            provider_class=MagicMock,
            team_model_id="test-team-model",
            agent_model_id="test-agent-model",
        )

        assert config.api_key is None
