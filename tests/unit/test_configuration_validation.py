"""Unit tests for configuration validation functionality.

Tests the configuration validation system to ensure proper API key
validation and environment configuration.
"""

import os
import pytest
from unittest.mock import patch

from mcp_server_mas_sequential_thinking.config.modernized_config import (
    ConfigurationManager,
    GitHubOpenAI,
    validate_configuration_comprehensive,
)


class TestConfigurationValidation:
    """Test suite for configuration validation."""

    def setup_method(self):
        """Set up test environment."""
        self.config_manager = ConfigurationManager()

    def test_valid_github_token_formats(self):
        """Test that valid GitHub token formats are accepted."""
        valid_tokens = [
            "ghp_" + "a" * 36,  # Classic PAT
            "github_pat_" + "a" * 80,  # Fine-grained PAT
            "gho_" + "a" * 36,  # OAuth token
        ]

        for token in valid_tokens:
            # Should not raise an exception
            GitHubOpenAI._validate_github_token(token)

    def test_invalid_github_token_formats(self):
        """Test that invalid GitHub token formats are rejected."""
        invalid_tokens = [
            "",  # Empty
            "invalid_token",  # Wrong prefix
            "ghp_short",  # Too short
            "ghp_" + "test" * 10,  # Contains 'test'
            "ghp_" + "a" * 5,  # Too short
        ]

        for token in invalid_tokens:
            with pytest.raises(ValueError):
                GitHubOpenAI._validate_github_token(token)

    def test_api_key_validation_patterns(self):
        """Test API key validation for different providers."""
        test_cases = [
            # (key_name, valid_key, should_pass)
            ("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 50, True),
            ("ANTHROPIC_API_KEY", "invalid_key", False),
            ("GROQ_API_KEY", "sk-" + "a" * 40, True),
            ("GROQ_API_KEY", "invalid_key", False),
            ("DEEPSEEK_API_KEY", "sk-" + "a" * 40, True),
        ]

        for key_name, key_value, should_pass in test_cases:
            if should_pass:
                # Should not raise an exception
                self.config_manager._validate_api_key_format(key_name, key_value)
            else:
                with pytest.raises(ValueError):
                    self.config_manager._validate_api_key_format(key_name, key_value)

    def test_placeholder_detection(self):
        """Test that placeholder API keys are detected."""
        placeholder_keys = [
            "test_api_key",
            "demo_token_here",
            "your_api_key_here",
            "example_key",
            "placeholder_token",
        ]

        for placeholder in placeholder_keys:
            with pytest.raises(ValueError, match="placeholder"):
                self.config_manager._validate_api_key_format("TEST_API_KEY", placeholder)

    def test_key_length_validation(self):
        """Test that API keys meet minimum length requirements."""
        short_key = "abc123"  # Too short
        with pytest.raises(ValueError, match="too short"):
            self.config_manager._validate_api_key_format("TEST_API_KEY", short_key)

    def test_character_validation(self):
        """Test that API keys contain only allowed characters."""
        invalid_chars_key = "sk-valid_key_with@#$%invalid_chars"
        with pytest.raises(ValueError, match="invalid characters"):
            self.config_manager._validate_api_key_format("TEST_API_KEY", invalid_chars_key)

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_required_keys(self):
        """Test detection of missing required API keys."""
        # Test with deepseek provider (requires DEEPSEEK_API_KEY)
        result = validate_configuration_comprehensive("deepseek")
        assert "DEEPSEEK_API_KEY" in result
        assert "Required but not set" in result["DEEPSEEK_API_KEY"]

    @patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-" + "a" * 40}, clear=True)
    def test_valid_configuration(self):
        """Test that valid configuration passes validation."""
        result = validate_configuration_comprehensive("deepseek")
        # Should not include DEEPSEEK_API_KEY in errors if valid
        assert "DEEPSEEK_API_KEY" not in result or "Required but not set" not in result.get("DEEPSEEK_API_KEY", "")

    def test_get_available_providers(self):
        """Test that available providers are returned correctly."""
        providers = self.config_manager.get_available_providers()
        expected_providers = ["deepseek", "groq", "openrouter", "ollama", "github", "anthropic"]

        for provider in expected_providers:
            assert provider in providers

    def test_provider_strategy_validation(self):
        """Test that provider strategies validate correctly."""
        # Test with valid provider
        strategy = self.config_manager.get_strategy("deepseek")
        assert strategy is not None

        # Test with invalid provider
        with pytest.raises(ValueError, match="Unknown provider"):
            self.config_manager.get_strategy("invalid_provider")

    @patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_" + "test" * 9}, clear=True)
    def test_invalid_github_token_in_environment(self):
        """Test that invalid GitHub tokens in environment are caught."""
        result = validate_configuration_comprehensive("github")
        assert "GITHUB_TOKEN" in result
        # Should detect the test pattern as invalid


class TestConfigurationEdgeCases:
    """Test edge cases for configuration validation."""

    def test_empty_environment_variable(self):
        """Test handling of empty environment variables."""
        config_manager = ConfigurationManager()

        with patch.dict(os.environ, {"TEST_API_KEY": ""}, clear=True):
            # Empty string should be treated as missing
            config_manager._validate_api_key_format("TEST_API_KEY", "")

    def test_whitespace_only_keys(self):
        """Test handling of whitespace-only API keys."""
        config_manager = ConfigurationManager()

        with pytest.raises(ValueError):
            config_manager._validate_api_key_format("TEST_API_KEY", "   ")

    def test_extremely_long_keys(self):
        """Test handling of extremely long API keys."""
        config_manager = ConfigurationManager()

        # Very long but valid key should pass basic validation
        long_key = "sk-" + "a" * 200
        config_manager._validate_api_key_format("TEST_API_KEY", long_key)

    def test_unicode_in_keys(self):
        """Test that unicode characters in API keys are handled."""
        config_manager = ConfigurationManager()

        unicode_key = "sk-test_key_with_ñ_and_émojis_🔑"
        with pytest.raises(ValueError, match="invalid characters"):
            config_manager._validate_api_key_format("TEST_API_KEY", unicode_key)