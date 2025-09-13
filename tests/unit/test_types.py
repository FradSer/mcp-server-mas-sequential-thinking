"""TDD tests for types module - validating type definitions and protocols."""

import pytest
from typing import get_type_hints, get_origin, get_args
from unittest.mock import Mock

from src.mcp_server_mas_sequential_thinking.types import (
    ProcessingMetadata,
    SessionStats,
    ComplexityMetrics,
    ModelProvider,
    AgentFactory,
    TeamBuilder,
    ValidationError,
    ConfigurationError,
    ThoughtProcessingError,
    TeamCreationError,
    ThoughtNumber,
    BranchId,
)


class TestProcessingMetadataTypedDict:
    """Test ProcessingMetadata TypedDict structure and typing."""

    def test_processing_metadata_structure_optional_fields(self):
        """RED: Test ProcessingMetadata allows partial data (total=False)."""
        # Should be able to create with only some fields
        metadata: ProcessingMetadata = {"strategy": "single_agent"}
        assert metadata["strategy"] == "single_agent"

        # Should be able to create with all fields
        full_metadata: ProcessingMetadata = {
            "strategy": "multi_agent",
            "complexity_score": 75.5,
            "estimated_cost": 0.05,
            "actual_cost": 0.048,
            "token_usage": 1200,
            "processing_time": 2.5,
            "specialists": ["planner", "researcher"],
            "provider": "deepseek",
            "routing_reasoning": "High complexity requires multiple agents",
            "error_count": 0,
            "retry_count": 1,
        }
        assert len(full_metadata) == 11

    def test_processing_metadata_type_annotations(self):
        """RED: Test ProcessingMetadata has correct type annotations."""
        hints = get_type_hints(ProcessingMetadata)

        # Test string fields
        assert hints["strategy"] == str
        assert hints["provider"] == str
        assert hints["routing_reasoning"] == str

        # Test numeric fields
        assert hints["complexity_score"] == float
        assert hints["estimated_cost"] == float
        assert hints["actual_cost"] == float
        assert hints["processing_time"] == float
        assert hints["token_usage"] == int
        assert hints["error_count"] == int
        assert hints["retry_count"] == int

        # Test list field
        assert get_origin(hints["specialists"]) == list
        assert get_args(hints["specialists"])[0] == str

    def test_processing_metadata_empty_creation(self):
        """RED: Test ProcessingMetadata can be created empty."""
        metadata: ProcessingMetadata = {}
        assert isinstance(metadata, dict)
        assert len(metadata) == 0


class TestSessionStatsTypedDict:
    """Test SessionStats TypedDict structure and typing."""

    def test_session_stats_structure(self):
        """RED: Test SessionStats TypedDict structure."""
        stats: SessionStats = {
            "total_thoughts": 50,
            "total_cost": 0.125,
            "total_tokens": 25000,
            "average_processing_time": 1.8,
            "error_rate": 0.02,
            "successful_thoughts": 49,
            "failed_thoughts": 1,
        }
        assert stats["total_thoughts"] == 50
        assert stats["error_rate"] == 0.02

    def test_session_stats_type_annotations(self):
        """RED: Test SessionStats has correct type annotations."""
        hints = get_type_hints(SessionStats)

        assert hints["total_thoughts"] == int
        assert hints["successful_thoughts"] == int
        assert hints["failed_thoughts"] == int
        assert hints["total_tokens"] == int

        assert hints["total_cost"] == float
        assert hints["average_processing_time"] == float
        assert hints["error_rate"] == float

    def test_session_stats_partial_creation(self):
        """RED: Test SessionStats allows partial data."""
        partial_stats: SessionStats = {"total_thoughts": 10}
        assert partial_stats["total_thoughts"] == 10


class TestComplexityMetricsTypedDict:
    """Test ComplexityMetrics TypedDict structure and typing."""

    def test_complexity_metrics_required_fields(self):
        """RED: Test ComplexityMetrics requires all fields (total=True)."""
        # This should work with all fields
        metrics: ComplexityMetrics = {
            "word_count": 150,
            "sentence_count": 8,
            "question_count": 2,
            "technical_terms": 5,
            "has_branching": False,
            "has_research_keywords": True,
            "has_analysis_keywords": True,
            "overall_score": 67.5,
        }
        assert metrics["word_count"] == 150
        assert metrics["has_research_keywords"] is True

    def test_complexity_metrics_type_annotations(self):
        """RED: Test ComplexityMetrics has correct type annotations."""
        hints = get_type_hints(ComplexityMetrics)

        # Integer fields
        assert hints["word_count"] == int
        assert hints["sentence_count"] == int
        assert hints["question_count"] == int
        assert hints["technical_terms"] == int

        # Boolean fields
        assert hints["has_branching"] == bool
        assert hints["has_research_keywords"] == bool
        assert hints["has_analysis_keywords"] == bool

        # Float field
        assert hints["overall_score"] == float


class TestProtocols:
    """Test Protocol classes for type checking."""

    def test_model_provider_protocol(self):
        """RED: Test ModelProvider protocol defines required methods."""
        # Create a mock that satisfies the protocol
        mock_provider = Mock()
        mock_provider.id = "test-model"
        mock_provider.cost_per_token = 0.0001

        # Should be able to use as ModelProvider type
        provider: ModelProvider = mock_provider
        assert provider.id == "test-model"
        assert provider.cost_per_token == 0.0001

    def test_agent_factory_protocol(self):
        """RED: Test AgentFactory protocol defines required methods."""
        mock_factory = Mock()
        mock_factory.create_team_agents.return_value = {"agent1": Mock(), "agent2": Mock()}

        # Should be able to use as AgentFactory type
        factory: AgentFactory = mock_factory
        agents = factory.create_team_agents(Mock(), "standard")
        assert len(agents) == 2

    def test_team_builder_protocol(self):
        """RED: Test TeamBuilder protocol defines required methods."""
        mock_builder = Mock()
        mock_builder.build_team.return_value = Mock()

        # Should be able to use as TeamBuilder type
        builder: TeamBuilder = mock_builder
        team = builder.build_team(Mock(), Mock())
        assert team is not None


class TestCustomExceptions:
    """Test custom exception classes."""

    def test_validation_error_inheritance(self):
        """RED: Test ValidationError inherits from ValueError."""
        error = ValidationError("Test validation error")
        assert isinstance(error, ValueError)
        assert str(error) == "Test validation error"

    def test_configuration_error_inheritance(self):
        """RED: Test ConfigurationError inherits from Exception."""
        error = ConfigurationError("Test configuration error")
        assert isinstance(error, Exception)
        assert str(error) == "Test configuration error"

    def test_thought_processing_error_inheritance(self):
        """RED: Test ThoughtProcessingError inherits from Exception."""
        error = ThoughtProcessingError("Test processing error")
        assert isinstance(error, Exception)
        assert str(error) == "Test processing error"

    def test_team_creation_error_inheritance(self):
        """RED: Test TeamCreationError inherits from Exception."""
        error = TeamCreationError("Test team creation error")
        assert isinstance(error, Exception)
        assert str(error) == "Test team creation error"

    def test_custom_exceptions_can_be_raised_and_caught(self):
        """RED: Test custom exceptions can be raised and caught."""
        with pytest.raises(ValidationError):
            raise ValidationError("Test validation")

        with pytest.raises(ConfigurationError):
            raise ConfigurationError("Test configuration")

        with pytest.raises(ThoughtProcessingError):
            raise ThoughtProcessingError("Test processing")

        with pytest.raises(TeamCreationError):
            raise TeamCreationError("Test team creation")


class TestTypeAliases:
    """Test type aliases for better type safety."""

    def test_thought_number_alias(self):
        """RED: Test ThoughtNumber type alias works correctly."""
        # ThoughtNumber should be an alias for int
        thought_num: ThoughtNumber = 42
        assert isinstance(thought_num, int)
        assert thought_num == 42

    def test_branch_id_alias(self):
        """RED: Test BranchId type alias works correctly."""
        # BranchId should be an alias for str
        branch: BranchId = "main-branch"
        assert isinstance(branch, str)
        assert branch == "main-branch"

    def test_type_aliases_in_function_signatures(self):
        """RED: Test type aliases can be used in function signatures."""
        def process_thought(thought_number: ThoughtNumber, branch_id: BranchId) -> str:
            return f"Processing thought {thought_number} in branch {branch_id}"

        result = process_thought(5, "feature-branch")
        assert result == "Processing thought 5 in branch feature-branch"


class TestTypeCompatibility:
    """Test type compatibility and runtime behavior."""

    def test_processing_metadata_runtime_validation(self):
        """RED: Test ProcessingMetadata behaves as expected at runtime."""
        # Test with valid data
        metadata: ProcessingMetadata = {
            "strategy": "single_agent",
            "complexity_score": 45.0,
            "token_usage": 800,
        }

        # Should behave like a regular dict
        assert "strategy" in metadata
        assert metadata.get("complexity_score") == 45.0
        assert metadata.get("nonexistent_key") is None

    def test_typed_dict_vs_regular_dict(self):
        """RED: Test TypedDict behaves like regular dict at runtime."""
        metadata: ProcessingMetadata = {"strategy": "test"}
        regular_dict = {"strategy": "test"}

        # Both should have same runtime behavior
        assert metadata["strategy"] == regular_dict["strategy"]
        assert len(metadata) == len(regular_dict)
        assert list(metadata.keys()) == list(regular_dict.keys())

    def test_protocol_duck_typing(self):
        """RED: Test protocols work with duck typing."""
        class MockModelProvider:
            def __init__(self):
                self.id = "mock-model"
                self.cost_per_token = 0.0001

        # Should work with Protocol due to duck typing
        mock = MockModelProvider()
        provider: ModelProvider = mock
        assert provider.id == "mock-model"


class TestTypeHintsConsistency:
    """Test that type hints are consistent across the module."""

    def test_all_typed_dicts_have_annotations(self):
        """RED: Test all TypedDict classes have proper annotations."""
        typed_dicts = [ProcessingMetadata, SessionStats, ComplexityMetrics]

        for typed_dict in typed_dicts:
            hints = get_type_hints(typed_dict)
            assert len(hints) > 0, f"{typed_dict.__name__} should have type annotations"

    def test_exception_classes_proper_inheritance(self):
        """RED: Test all custom exceptions have proper inheritance."""
        exceptions = [
            ValidationError,
            ConfigurationError,
            ThoughtProcessingError,
            TeamCreationError,
        ]

        for exception_class in exceptions:
            # All should inherit from Exception
            assert issubclass(exception_class, Exception)

            # Should be instantiable with a message
            instance = exception_class("test message")
            assert str(instance) == "test message"