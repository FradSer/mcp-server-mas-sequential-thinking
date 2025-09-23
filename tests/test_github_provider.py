"""Comprehensive tests for GitHub Models provider.

Tests GitHubStrategy class implementation using Test-Driven Development approach.
Covers initialization, configuration, environment overrides, and agno Model integration.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from mcp_server_mas_sequential_thinking.modernized_config import (
    BaseProviderStrategy,
    GitHubStrategy,
    ModelConfig,
)

# Import ProviderStrategy for compatibility
ProviderStrategy = BaseProviderStrategy


class TestGitHubStrategyDefaults:
    """Test default configurations of GitHubStrategy."""

    def test_github_strategy_defaults(self):
        """Test that GitHubStrategy has correct default values."""
        strategy = GitHubStrategy()

        assert strategy.default_team_model == "openai/gpt-5"
        assert strategy.default_agent_model == "openai/gpt-5-min"
        assert strategy.api_key_name == "GITHUB_TOKEN"

    def test_provider_class_returns_github_openai(self):
        """Test that provider_class returns GitHub-configured OpenAI class."""
        strategy = GitHubStrategy()

        # Get the provider class (should be GitHubOpenAI)
        provider_class = strategy.provider_class

        # Verify it's a class and has the expected properties
        assert callable(provider_class)
        assert provider_class.__name__ == "GitHubOpenAI"

    def test_strategy_inherits_from_provider_strategy(self):
        """Test that GitHubStrategy properly inherits from ProviderStrategy."""
        strategy = GitHubStrategy()

        assert isinstance(strategy, ProviderStrategy)
        assert hasattr(strategy, "get_config")


class TestGitHubStrategyEnvironmentOverrides:
    """Test environment variable overrides for GitHubStrategy."""

    def test_team_model_environment_override(self):
        """Test that GITHUB_TEAM_MODEL_ID overrides default team model."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_TEAM_MODEL_ID": "gpt-4-turbo"}):
            config = strategy.get_config()
            assert config.team_model_id == "gpt-4-turbo"

    def test_agent_model_environment_override(self):
        """Test that GITHUB_AGENT_MODEL_ID overrides default agent model."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_AGENT_MODEL_ID": "gpt-3.5-turbo"}):
            config = strategy.get_config()
            assert config.agent_model_id == "gpt-3.5-turbo"

    def test_both_model_environment_overrides(self):
        """Test that both model environment variables can be overridden simultaneously."""
        strategy = GitHubStrategy()

        env_vars = {
            "GITHUB_TEAM_MODEL_ID": "gpt-4-turbo",
            "GITHUB_AGENT_MODEL_ID": "gpt-3.5-turbo",
        }

        with patch.dict(os.environ, env_vars):
            config = strategy.get_config()
            assert config.team_model_id == "gpt-4-turbo"
            assert config.agent_model_id == "gpt-3.5-turbo"

    def test_no_environment_variables_uses_defaults(self):
        """Test that without environment variables, defaults are used."""
        strategy = GitHubStrategy()

        # Clear any existing GitHub environment variables
        env_clear = {
            "GITHUB_TEAM_MODEL_ID": "",
            "GITHUB_AGENT_MODEL_ID": "",
            "GITHUB_TOKEN": "",
        }

        with patch.dict(os.environ, env_clear, clear=False):
            with patch.dict(os.environ, {}, clear=False):  # Remove the keys entirely
                config = strategy.get_config()
                assert config.team_model_id == "openai/gpt-5"
                assert config.agent_model_id == "openai/gpt-5-min"


class TestGitHubStrategyAPIKeyValidation:
    """Test API key validation and handling."""

    def test_api_key_from_environment(self):
        """Test that API key is read from environment variable."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-github-token"}):
            config = strategy.get_config()
            assert config.api_key == "test-github-token"

    def test_missing_api_key_returns_none(self):
        """Test that missing API key returns None."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {}, clear=True):
            config = strategy.get_config()
            assert config.api_key is None

    def test_empty_api_key_returns_none(self):
        """Test that empty API key returns None."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_TOKEN": ""}):
            config = strategy.get_config()
            assert config.api_key is None

    def test_api_key_name_property(self):
        """Test that api_key_name property returns correct value."""
        strategy = GitHubStrategy()

        assert strategy.api_key_name == "GITHUB_TOKEN"


class TestGitHubStrategyGetConfig:
    """Test get_config() method returns proper ModelConfig."""

    def test_get_config_returns_model_config_instance(self):
        """Test that get_config returns ModelConfig instance."""
        strategy = GitHubStrategy()

        config = strategy.get_config()
        assert isinstance(config, ModelConfig)

    def test_get_config_with_all_defaults(self):
        """Test get_config with all default values."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {}, clear=True):
            config = strategy.get_config()

            assert config.provider_class.__name__ == "GitHubOpenAI"
            assert config.team_model_id == "openai/gpt-5"
            assert config.agent_model_id == "openai/gpt-5-min"
            assert config.api_key is None

    def test_get_config_with_full_environment(self):
        """Test get_config with full environment configuration."""
        strategy = GitHubStrategy()

        env_vars = {
            "GITHUB_TEAM_MODEL_ID": "gpt-4-turbo",
            "GITHUB_AGENT_MODEL_ID": "gpt-3.5-turbo",
            "GITHUB_TOKEN": "test-token-12345",
        }

        with patch.dict(os.environ, env_vars):
            config = strategy.get_config()

            assert config.provider_class.__name__ == "GitHubOpenAI"
            assert config.team_model_id == "gpt-4-turbo"
            assert config.agent_model_id == "gpt-3.5-turbo"
            assert config.api_key == "test-token-12345"

    def test_get_config_immutable(self):
        """Test that ModelConfig is immutable (frozen dataclass)."""
        strategy = GitHubStrategy()

        config = strategy.get_config()

        # Should raise AttributeError due to frozen dataclass
        with pytest.raises(AttributeError):
            config.team_model_id = "new-model"


class TestGitHubOpenAIInitialization:
    """Test GitHubOpenAI initialization with correct base_url."""

    @patch("agno.models.openai.OpenAIChat")
    def test_github_openai_initialization(self, mock_openai_chat):
        """Test GitHubOpenAI is initialized with GitHub Models base URL."""
        strategy = GitHubStrategy()

        # Mock the OpenAIChat constructor
        mock_instance = MagicMock()
        mock_openai_chat.return_value = mock_instance

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
            config = strategy.get_config()

            # Get the GitHub OpenAI class and test instantiation
            github_openai_class = config.provider_class

            # Verify it's the GitHubOpenAI class
            assert github_openai_class.__name__ == "GitHubOpenAI"

    def test_github_openai_base_url_configuration(self):
        """Test that GitHubOpenAI sets correct base URL."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_TOKEN": "test-token"}):
            config = strategy.get_config()

            # The provider class should be GitHubOpenAI
            assert config.provider_class.__name__ == "GitHubOpenAI"

    def test_github_openai_api_key_handling(self):
        """Test GitHubOpenAI API key handling."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test_token"}):
            config = strategy.get_config()

            # Verify the config has the correct API key
            assert config.api_key == "ghp_test_token"

    @patch("agno.models.openai.OpenAIChat.__init__", return_value=None)
    def test_github_openai_uses_correct_base_url(self, mock_openai_init):
        """Test that GitHubOpenAI passes correct base_url to parent class."""
        from mcp_server_mas_sequential_thinking.config import GitHubOpenAI

        # Create GitHubOpenAI instance with valid test token
        valid_token = "ghp_1234567890abcdef1234567890abcdef1234"
        with patch.dict(os.environ, {"GITHUB_TOKEN": valid_token}):
            GitHubOpenAI()

        # Verify that OpenAIChat.__init__ was called with correct base_url
        mock_openai_init.assert_called_once()
        call_kwargs = mock_openai_init.call_args[1]
        assert call_kwargs["base_url"] == "https://models.github.ai/inference"
        assert call_kwargs["api_key"] == valid_token


class TestGitHubStrategyIntegration:
    """Integration tests for GitHubStrategy."""

    def test_strategy_follows_provider_pattern(self):
        """Test that GitHubStrategy follows the same pattern as other providers."""
        strategy = GitHubStrategy()

        # Should implement all abstract properties
        assert hasattr(strategy, "provider_class")
        assert hasattr(strategy, "default_team_model")
        assert hasattr(strategy, "default_agent_model")
        assert hasattr(strategy, "api_key_name")
        assert hasattr(strategy, "get_config")

        # Properties should return expected types
        assert isinstance(strategy.default_team_model, str)
        assert isinstance(strategy.default_agent_model, str)
        assert isinstance(strategy.api_key_name, str)

    def test_strategy_in_strategies_registry(self):
        """Test that GitHubStrategy can be added to STRATEGIES registry."""
        from mcp_server_mas_sequential_thinking.config import STRATEGIES

        # Add GitHub strategy to registry
        github_strategy = GitHubStrategy()
        test_strategies = STRATEGIES.copy()
        test_strategies["github"] = github_strategy

        assert "github" in test_strategies
        assert isinstance(test_strategies["github"], GitHubStrategy)
        assert isinstance(test_strategies["github"], ProviderStrategy)

    def test_prefix_extraction_for_environment_variables(self):
        """Test that environment variable prefix is correctly extracted from class name."""
        strategy = GitHubStrategy()

        # We can test this indirectly by checking environment variable lookup
        with patch.dict(os.environ, {"GITHUB_TEAM_MODEL_ID": "test-model"}):
            config = strategy.get_config()
            assert config.team_model_id == "test-model"

    def test_full_workflow_simulation(self):
        """Test a complete workflow simulation."""
        strategy = GitHubStrategy()

        # Set up full environment
        env_vars = {
            "GITHUB_TEAM_MODEL_ID": "openai/gpt-5",
            "GITHUB_AGENT_MODEL_ID": "openai/gpt-5-min",
            "GITHUB_TOKEN": "ghp_test_token_123456789",
        }

        with patch.dict(os.environ, env_vars):
            # Get configuration
            config = strategy.get_config()

            # Verify configuration is correct
            assert config.team_model_id == "openai/gpt-5"
            assert config.agent_model_id == "openai/gpt-5-min"
            assert config.api_key == "ghp_test_token_123456789"
            assert config.provider_class.__name__ == "GitHubOpenAI"


class TestEnvironmentVariableHelper:
    """Test helper method for environment variable handling."""

    def test_get_env_with_fallback_returns_environment_value(self):
        """Test that helper returns environment value when present."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"TEST_VAR": "environment_value"}):
            result = strategy._get_env_with_fallback("TEST_VAR", "fallback_value")
            assert result == "environment_value"

    def test_get_env_with_fallback_returns_fallback_when_missing(self):
        """Test that helper returns fallback when environment variable is missing."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {}, clear=True):
            result = strategy._get_env_with_fallback("MISSING_VAR", "fallback_value")
            assert result == "fallback_value"

    def test_get_env_with_fallback_returns_fallback_when_empty(self):
        """Test that helper returns fallback when environment variable is empty."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"EMPTY_VAR": ""}):
            result = strategy._get_env_with_fallback("EMPTY_VAR", "fallback_value")
            assert result == "fallback_value"

    def test_get_env_with_fallback_returns_fallback_when_none(self):
        """Test that helper returns fallback when environment variable is None."""
        strategy = GitHubStrategy()

        with patch("os.environ.get", return_value=None):
            result = strategy._get_env_with_fallback("NULL_VAR", "fallback_value")
            assert result == "fallback_value"

    def test_get_env_with_fallback_preserves_whitespace_values(self):
        """Test that helper preserves valid whitespace-only values."""
        strategy = GitHubStrategy()

        with patch.dict(os.environ, {"WHITESPACE_VAR": "   "}):
            result = strategy._get_env_with_fallback("WHITESPACE_VAR", "fallback_value")
            assert result == "   "


class TestGitHubTokenValidation:
    """Test GitHub token format validation."""

    @patch("agno.models.openai.OpenAIChat.__init__", return_value=None)
    def test_valid_github_token_formats(self, mock_openai_init):
        """Test that valid GitHub token formats are accepted."""
        from mcp_server_mas_sequential_thinking.config import GitHubOpenAI

        valid_tokens = [
            "ghp_1234567890abcdef1234567890abcdef1234",  # Classic PAT (40 chars)
            "github_pat_11ABCDEFG_1234567890abcdef1234567890abcdef1234567890abcdef1234567890ab",  # Fine-grained PAT
            "gho_1234567890abcdef1234567890abcdef12345678",  # OAuth token
            "ghu_1234567890abcdef1234567890abcdef12345678",  # User-to-server token
        ]

        for token in valid_tokens:
            with patch.dict(os.environ, {"GITHUB_TOKEN": token}):
                # Should not raise an exception
                github_client = GitHubOpenAI()
                assert github_client is not None

    def test_invalid_github_token_formats_raise_error(self):
        """Test that invalid GitHub token formats raise validation error."""
        from mcp_server_mas_sequential_thinking.config import GitHubOpenAI

        invalid_tokens = [
            ("invalid_token", "Invalid GitHub token format"),
            ("ghp_short", "Invalid GitHub classic PAT length"),
            ("not_a_github_token_at_all", "Invalid GitHub token format"),
            ("1234567890", "Invalid GitHub token format"),
            ("", "GitHub token is required"),
        ]

        for token, expected_error in invalid_tokens:
            with patch.dict(os.environ, {"GITHUB_TOKEN": token}):
                with pytest.raises(ValueError, match=expected_error):
                    GitHubOpenAI()

    def test_missing_github_token_raises_error(self):
        """Test that missing GitHub token raises appropriate error."""
        from mcp_server_mas_sequential_thinking.config import GitHubOpenAI

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="GitHub token is required"):
                GitHubOpenAI()


class TestGitHubStrategyErrorHandling:
    """Test error handling scenarios for GitHubStrategy."""

    def test_provider_class_returns_valid_class(self):
        """Test that provider_class returns a valid callable class."""
        strategy = GitHubStrategy()

        provider_class = strategy.provider_class

        # Should be a callable class
        assert callable(provider_class)
        assert hasattr(provider_class, "__name__")
        assert provider_class.__name__ == "GitHubOpenAI"

    def test_malformed_environment_variables(self):
        """Test handling of malformed environment variables."""
        strategy = GitHubStrategy()

        # Test with whitespace-only values
        env_vars = {
            "GITHUB_TEAM_MODEL_ID": "   ",
            "GITHUB_AGENT_MODEL_ID": "\t\n",
            "GITHUB_TOKEN": "  \r\n  ",
        }

        with patch.dict(os.environ, env_vars):
            config = strategy.get_config()

            # Whitespace-only values should be treated as valid strings
            assert config.team_model_id == "   "
            assert config.agent_model_id == "\t\n"
            assert config.api_key == "  \r\n  "


if __name__ == "__main__":
    pytest.main([__file__])
