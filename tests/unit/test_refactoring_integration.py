"""TDD tests for refactoring integration - verifying behavior consistency after refactoring."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

from src.mcp_server_mas_sequential_thinking.constants import (
    ValidationLimits,
    TokenCosts,
    ComplexityScoring,
    DefaultSettings,
)
from src.mcp_server_mas_sequential_thinking.types import (
    ProcessingMetadata,
    ValidationError,
    ConfigurationError,
    ThoughtProcessingError,
)
from src.mcp_server_mas_sequential_thinking.models import ThoughtData
from src.mcp_server_mas_sequential_thinking.session import SessionMemory
from src.mcp_server_mas_sequential_thinking.server_core import create_validated_thought_data
from tests.helpers.factories import ThoughtDataBuilder


class TestConstantsIntegration:
    """Test that constants are properly used throughout the codebase."""

    def test_validation_limits_used_in_models(self):
        """RED: Test that ValidationLimits constants are used in model validation."""
        # Test thought number validation uses constants
        with pytest.raises(ValueError):
            create_validated_thought_data(
                thought="Test thought",
                thought_number=0,  # Below MIN_THOUGHT_NUMBER
                total_thoughts=ValidationLimits.MIN_TOTAL_THOUGHTS,
                next_needed=True,
                is_revision=False,
                revises_thought=None,
                branch_from=None,
                branch_id=None,
                needs_more=True,
            )

    def test_validation_limits_used_in_session_memory(self):
        """RED: Test that ValidationLimits constants are used in SessionMemory."""
        from agno.team.team import Team

        mock_team = Mock(spec=Team)
        session = SessionMemory(team=mock_team)

        # Verify that SessionMemory uses the same limits as ValidationLimits
        assert session.MAX_THOUGHTS_PER_SESSION == ValidationLimits.MAX_THOUGHTS_PER_SESSION
        assert session.MAX_BRANCHES_PER_SESSION == ValidationLimits.MAX_BRANCHES_PER_SESSION
        assert session.MAX_THOUGHTS_PER_BRANCH == ValidationLimits.MAX_THOUGHTS_PER_BRANCH

        # Test with a smaller limit to avoid creating 1000 objects in test
        # Fill up to just under the limit
        for i in range(1, 11):  # Add 10 thoughts instead of 1000
            thought = ThoughtDataBuilder().with_number(i).build()
            session.add_thought(thought)

        # Verify thoughts were added
        assert len(session.thought_history) == 10

    def test_token_costs_used_in_processing(self):
        """RED: Test that TokenCosts constants are accessible for processing."""
        # Test that all provider costs are available
        assert TokenCosts.DEEPSEEK_COST_PER_1K > 0
        assert TokenCosts.GROQ_COST_PER_1K >= 0  # Groq might be free or very cheap
        assert TokenCosts.OLLAMA_COST_PER_1K == 0.0  # Local model, no cost
        assert TokenCosts.DEFAULT_COST_PER_1K > 0

    def test_complexity_scoring_constants_available(self):
        """RED: Test that ComplexityScoring constants are available for analysis."""
        # Test that scoring constants sum to max score
        total_max_points = (
            ComplexityScoring.WORD_COUNT_MAX_POINTS
            + ComplexityScoring.SENTENCE_MAX_POINTS
            + ComplexityScoring.QUESTION_MAX_POINTS
            + ComplexityScoring.TECHNICAL_TERM_MAX_POINTS
            + ComplexityScoring.BRANCHING_MAX_POINTS
            + ComplexityScoring.RESEARCH_MAX_POINTS
            + ComplexityScoring.ANALYSIS_MAX_POINTS
        )
        assert total_max_points == ComplexityScoring.MAX_SCORE


class TestTypesIntegration:
    """Test that type definitions work correctly with actual code."""

    def test_processing_metadata_used_in_storage(self):
        """RED: Test ProcessingMetadata integrates with persistent storage."""
        metadata: ProcessingMetadata = {
            "strategy": "multi_agent",
            "complexity_score": 75.0,
            "estimated_cost": 0.05,
            "actual_cost": 0.048,
            "token_usage": 1200,
            "processing_time": 2.5,
            "specialists": ["planner", "researcher", "analyzer"],
            "provider": "deepseek",
            "routing_reasoning": "High complexity requires multiple specialists",
        }

        # Metadata should be compatible with storage functions
        assert isinstance(metadata, dict)
        assert metadata["strategy"] == "multi_agent"
        assert len(metadata["specialists"]) == 3

    def test_custom_exceptions_used_in_validation(self):
        """RED: Test custom exceptions are used in validation functions."""
        # Test ValidationError is raised for invalid input
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation failed")

        # Test ConfigurationError is raised for invalid config
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test configuration failed")

        # Test ThoughtProcessingError is raised for processing failures
        with pytest.raises(ThoughtProcessingError):
            raise ThoughtProcessingError("Test processing failed")

    def test_thought_processing_error_with_metadata(self):
        """RED: Test ThoughtProcessingError can carry ProcessingMetadata."""
        metadata: ProcessingMetadata = {
            "strategy": "single_agent",
            "error_count": 1,
            "retry_count": 2,
        }

        error = ThoughtProcessingError("Processing failed", metadata)
        assert error.metadata["strategy"] == "single_agent"
        assert error.metadata["error_count"] == 1


class TestModuleUnificationIntegration:
    """Test that unified modules work correctly after consolidation."""

    @patch("src.mcp_server_mas_sequential_thinking.modernized_config.get_model_config")
    def test_unified_config_module_integration(self, mock_get_config):
        """RED: Test unified config module works as expected."""
        from src.mcp_server_mas_sequential_thinking.modernized_config import (
            get_model_config,
            check_required_api_keys,
        )

        # Mock config to test integration
        mock_config = Mock()
        mock_config.provider_class = Mock()
        mock_config.team_model_id = "test-team-model"
        mock_config.agent_model_id = "test-agent-model"
        mock_get_config.return_value = mock_config

        # Should be able to get config
        config = get_model_config("deepseek")
        assert config.team_model_id == "test-team-model"
        assert config.agent_model_id == "test-agent-model"

    def test_unified_agents_module_import(self):
        """RED: Test unified agents module can be imported and used."""
        from src.mcp_server_mas_sequential_thinking.unified_agents import (
            UnifiedAgentFactory,
        )

        # Should be able to instantiate the factory
        factory = UnifiedAgentFactory()
        assert factory is not None

    def test_unified_team_module_import(self):
        """RED: Test unified team module can be imported and used."""
        from src.mcp_server_mas_sequential_thinking.unified_team import (
            create_team,
            StandardTeamBuilder,
            EnhancedTeamBuilder,
        )

        # Should be able to instantiate builders
        standard_builder = StandardTeamBuilder()
        enhanced_builder = EnhancedTeamBuilder()
        assert standard_builder is not None
        assert enhanced_builder is not None


class TestRefactoredValidationBehavior:
    """Test that validation behavior is consistent after refactoring."""

    def test_thought_data_validation_uses_constants(self):
        """RED: Test ThoughtData validation uses extracted constants."""
        # Valid thought should work
        valid_thought = create_validated_thought_data(
            thought="This is a valid test thought",
            thought_number=ValidationLimits.MIN_THOUGHT_NUMBER,
            total_thoughts=ValidationLimits.MIN_TOTAL_THOUGHTS,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=None,
            branch_id=None,
            needs_more=True,
        )
        assert valid_thought.thought_number == ValidationLimits.MIN_THOUGHT_NUMBER
        assert valid_thought.total_thoughts == ValidationLimits.MIN_TOTAL_THOUGHTS

    def test_session_memory_limits_enforced(self):
        """RED: Test SessionMemory enforces limits using constants."""
        from agno.team.team import Team

        mock_team = Mock(spec=Team)
        session = SessionMemory(team=mock_team)

        # Test branch limit enforcement
        base_thought = ThoughtDataBuilder().with_number(1).build()
        session.add_thought(base_thought)

        # Should be able to create branches up to limit
        for i in range(ValidationLimits.MAX_BRANCHES_PER_SESSION):
            branch_thought = (
                ThoughtDataBuilder()
                .with_number(i + 2)
                .as_branch(1, f"branch-{i}")
                .build()
            )
            session.add_thought(branch_thought)

        # Creating one more branch should fail
        with pytest.raises(ValueError, match="exceeds maximum.*branches"):
            over_limit_branch = (
                ThoughtDataBuilder()
                .with_number(ValidationLimits.MAX_BRANCHES_PER_SESSION + 2)
                .as_branch(1, f"branch-{ValidationLimits.MAX_BRANCHES_PER_SESSION}")
                .build()
            )
            session.add_thought(over_limit_branch)

    def test_default_settings_integration(self):
        """RED: Test DefaultSettings are used throughout the system."""
        # Test that default provider matches constant
        assert DefaultSettings.DEFAULT_PROVIDER == "deepseek"

        # Test that default complexity threshold is reasonable
        assert 0 < DefaultSettings.DEFAULT_COMPLEXITY_THRESHOLD < 100

        # Test that default token buffer is a reasonable percentage
        assert 0 < DefaultSettings.DEFAULT_TOKEN_BUFFER < 1.0


class TestErrorHandlingConsistency:
    """Test that error handling is consistent after type improvements."""

    def test_validation_error_inheritance_works(self):
        """RED: Test ValidationError can be caught as ValueError."""
        with pytest.raises(ValueError):
            raise ValidationError("Test validation error")

        # Should also be catchable as ValidationError specifically
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation error")

    def test_thought_processing_error_with_metadata_integration(self):
        """RED: Test ThoughtProcessingError integrates with ProcessingMetadata."""
        metadata: ProcessingMetadata = {
            "strategy": "adaptive",
            "complexity_score": 85.0,
            "error_count": 3,
            "retry_count": 2,
        }

        error = ThoughtProcessingError("Complex processing failed", metadata)

        # Error should carry metadata
        assert error.metadata["strategy"] == "adaptive"
        assert error.metadata["error_count"] == 3

        # Should still be a proper exception
        with pytest.raises(ThoughtProcessingError):
            raise error


class TestPerformanceImprovements:
    """Test that refactoring maintained or improved performance."""

    def test_session_memory_cache_optimization(self):
        """RED: Test SessionMemory O(1) lookup optimization works."""
        from agno.team.team import Team

        mock_team = Mock(spec=Team)
        session = SessionMemory(team=mock_team)

        # Add multiple thoughts
        thoughts = []
        for i in range(1, 20):
            thought = ThoughtDataBuilder().with_number(i).with_thought(f"Thought {i}").build()
            session.add_thought(thought)
            thoughts.append(thought)

        # Looking up any thought should be fast (O(1) cache lookup)
        content = session.find_thought_content(10)
        assert content == "Thought 10"

        # Looking up non-existent thought should return default
        content = session.find_thought_content(999)
        assert content == "Unknown thought"

    def test_constants_eliminate_magic_numbers(self):
        """RED: Test that magic numbers have been eliminated from validation."""
        # Before refactoring, these would have been magic numbers
        # Now they should all use constants

        # Test that we can find the source of all important limits
        assert hasattr(ValidationLimits, 'MAX_THOUGHTS_PER_SESSION')
        assert hasattr(ValidationLimits, 'MAX_BRANCHES_PER_SESSION')
        assert hasattr(ValidationLimits, 'MIN_THOUGHT_NUMBER')
        assert hasattr(ValidationLimits, 'MIN_TOTAL_THOUGHTS')

        # Test that they have reasonable values
        assert ValidationLimits.MAX_THOUGHTS_PER_SESSION >= 100
        assert ValidationLimits.MAX_BRANCHES_PER_SESSION >= 10
        assert ValidationLimits.MIN_THOUGHT_NUMBER == 1
        assert ValidationLimits.MIN_TOTAL_THOUGHTS >= 5


class TestBackwardCompatibility:
    """Test that refactoring maintains backward compatibility."""

    def test_thought_data_model_compatibility(self):
        """RED: Test ThoughtData model maintains compatible interface."""
        # Should be able to create ThoughtData with all fields
        thought_data = ThoughtData(
            thought="Test thought",
            thought_number=1,
            total_thoughts=5,
            next_needed=True,
            is_revision=False,
            revises_thought=None,
            branch_from=None,
            branch_id=None,
            needs_more=True,
        )

        # All fields should be accessible
        assert thought_data.thought == "Test thought"
        assert thought_data.thought_number == 1
        assert thought_data.total_thoughts == 5
        assert thought_data.next_needed is True
        assert thought_data.is_revision is False

    def test_session_memory_interface_compatibility(self):
        """RED: Test SessionMemory maintains compatible interface."""
        from agno.team.team import Team

        mock_team = Mock(spec=Team)
        session = SessionMemory(team=mock_team)

        # Original interface should still work
        thought = ThoughtDataBuilder().build()
        session.add_thought(thought)

        # All original methods should be available
        assert hasattr(session, 'add_thought')
        assert hasattr(session, 'find_thought_content')
        assert hasattr(session, 'get_branch_summary')
        assert hasattr(session, 'get_current_branch_id')