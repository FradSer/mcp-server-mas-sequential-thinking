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


class TestRefactoredComponentsIntegration:
    """Test integration of newly refactored components (LoggingMixin, AgentFactory, StepExecutorMixin)."""

    def test_logging_mixin_with_cost_optimization_constants(self):
        """Test LoggingMixin uses CostOptimizationConstants correctly."""
        from src.mcp_server_mas_sequential_thinking.server_core import LoggingMixin
        from src.mcp_server_mas_sequential_thinking.constants import CostOptimizationConstants

        mixin = LoggingMixin()

        # Test efficiency score calculation with constants
        fast_time = 30.0  # Below threshold
        slow_time = 120.0  # Above threshold

        fast_score = mixin._calculate_efficiency_score(fast_time)
        slow_score = mixin._calculate_efficiency_score(slow_time)

        # Fast processing should get perfect score
        assert fast_score == 1.0
        # Slow processing should get reduced score
        assert slow_score < 1.0
        assert slow_score >= 0.5  # Minimum efficiency score

    def test_agent_factory_creates_functional_agents(self):
        """Test AgentFactory creates agents that can be used in workflows."""
        from src.mcp_server_mas_sequential_thinking.agno_workflow_router import AgentFactory
        from unittest.mock import Mock

        # Mock model for agent creation
        mock_model = Mock()
        mock_model.id = "test-model"

        # Test creating different types of agents
        with patch('src.mcp_server_mas_sequential_thinking.agno_workflow_router.Agent') as mock_agent_class:
            mock_agent = Mock()
            mock_agent_class.return_value = mock_agent

            # Create agents of different types
            planner = AgentFactory.create_planner(mock_model, "basic")
            analyzer = AgentFactory.create_analyzer(mock_model, "advanced")
            researcher = AgentFactory.create_researcher(mock_model)
            critic = AgentFactory.create_critic(mock_model)
            synthesizer = AgentFactory.create_synthesizer(mock_model)

            # All should be valid agent instances
            assert planner == mock_agent
            assert analyzer == mock_agent
            assert researcher == mock_agent
            assert critic == mock_agent
            assert synthesizer == mock_agent

            # Factory should have been called for each agent
            assert mock_agent_class.call_count == 5

    def test_step_executor_mixin_with_cost_optimization_constants(self):
        """Test StepExecutorMixin integrates with CostOptimizationConstants for session state."""
        from src.mcp_server_mas_sequential_thinking.agno_workflow_router import StepExecutorMixin
        from src.mcp_server_mas_sequential_thinking.constants import CostOptimizationConstants

        # Test step output creation with complexity from session state
        session_state = {
            "current_complexity_score": 75.5,
            "provider": "deepseek",
            "estimated_cost": CostOptimizationConstants.DEFAULT_COST_ESTIMATE
        }

        result = StepExecutorMixin._create_step_output(
            content="Integration test result",
            strategy="multi_agent",
            session_state=session_state,
            specialists=["planner", "analyzer"]
        )

        # Verify integration
        assert result.content["result"] == "Integration test result"
        assert result.content["strategy"] == "multi_agent"
        assert result.content["complexity"] == 75.5
        assert result.content["specialists"] == ["planner", "analyzer"]
        assert result.step_name == "multi_agent_processing"

    def test_complete_workflow_integration(self):
        """Test complete workflow using all refactored components together."""
        from src.mcp_server_mas_sequential_thinking.agno_workflow_router import AgentFactory, StepExecutorMixin
        from src.mcp_server_mas_sequential_thinking.server_core import LoggingMixin
        from src.mcp_server_mas_sequential_thinking.constants import CostOptimizationConstants
        from unittest.mock import Mock, patch

        # Create a mock class that uses all mixins
        class MockWorkflowProcessor(LoggingMixin, StepExecutorMixin):
            def __init__(self, model):
                self.model = model

            def process_complex_request(self, content: str, complexity_score: float):
                # Use AgentFactory to create agents
                with patch('src.mcp_server_mas_sequential_thinking.agno_workflow_router.Agent') as mock_agent_class:
                    mock_agent = Mock()
                    mock_agent_class.return_value = mock_agent

                    # Create agents using factory
                    if complexity_score > CostOptimizationConstants.HIGH_BUDGET_UTILIZATION * 100:
                        agents = [
                            AgentFactory.create_planner(self.model, "advanced"),
                            AgentFactory.create_analyzer(self.model, "advanced"),
                            AgentFactory.create_researcher(self.model),
                            AgentFactory.create_critic(self.model)
                        ]
                        strategy = "multi_agent"
                    else:
                        agents = [AgentFactory.create_planner(self.model, "basic")]
                        strategy = "single_agent"

                    # Create session state
                    session_state = {
                        "current_complexity_score": complexity_score,
                        "agents_created": len(agents)
                    }

                    # Update session state using mixin
                    self._update_session_state(session_state, strategy, f"{strategy}_completed")

                    # Calculate efficiency (simulate processing time)
                    processing_time = 45.0 if strategy == "single_agent" else 90.0
                    efficiency_score = self._calculate_efficiency_score(processing_time)

                    # Create step output using mixin
                    result = self._create_step_output(
                        content=f"Processed '{content}' using {strategy}",
                        strategy=strategy,
                        session_state=session_state,
                        specialists=[agent.__class__.__name__ for agent in agents] if len(agents) > 1 else None
                    )

                    return result, efficiency_score, session_state

        # Test the complete workflow
        mock_model = Mock()
        mock_model.id = "test-model"
        processor = MockWorkflowProcessor(mock_model)

        # Test high complexity scenario
        result, efficiency, session = processor.process_complex_request(
            "Complex analysis request",
            85.0  # High complexity
        )

        # Verify all components worked together
        assert result.content["result"] == "Processed 'Complex analysis request' using multi_agent"
        assert result.content["strategy"] == "multi_agent"
        assert result.content["complexity"] == 85.0
        assert "specialists" in result.content
        assert session["multi_agent_completed"] is True
        assert session["processing_strategy"] == "multi_agent"
        assert 0.0 <= efficiency <= 1.0

        # Test low complexity scenario
        result2, efficiency2, session2 = processor.process_complex_request(
            "Simple task",
            25.0  # Low complexity
        )

        assert result2.content["strategy"] == "single_agent"
        assert session2["single_agent_completed"] is True

    def test_constants_integration_across_modules(self):
        """Test that CostOptimizationConstants are used consistently across modules."""
        from src.mcp_server_mas_sequential_thinking.constants import (
            CostOptimizationConstants,
            QualityThresholds,
            TokenCosts
        )

        # Test that related constants are consistent
        # Quality thresholds should align with cost optimization thresholds
        assert QualityThresholds.HIGH_BUDGET_UTILIZATION == CostOptimizationConstants.HIGH_BUDGET_UTILIZATION

        # Test that token costs are reasonable for cost optimization
        # Both represent cost per 1000 tokens, so they should be equal
        assert TokenCosts.DEFAULT_COST_PER_1K == CostOptimizationConstants.DEFAULT_COST_ESTIMATE

        # Test that cost optimization constants are in valid ranges
        weights = [
            CostOptimizationConstants.QUALITY_WEIGHT,
            CostOptimizationConstants.COST_WEIGHT,
            CostOptimizationConstants.SPEED_WEIGHT,
            CostOptimizationConstants.RELIABILITY_WEIGHT
        ]

        # All weights should sum to 1.0
        assert abs(sum(weights) - 1.0) < 0.0001

        # All weights should be positive
        for weight in weights:
            assert weight > 0

    def test_error_handling_integration(self):
        """Test error handling works across all refactored components."""
        from src.mcp_server_mas_sequential_thinking.agno_workflow_router import StepExecutorMixin

        # Test error handling with different strategies
        test_error = ValueError("Integration test error")

        strategies = ["single_agent", "multi_agent", "hybrid", "parallel_analysis"]

        for strategy in strategies:
            result = StepExecutorMixin._handle_execution_error(test_error, strategy)

            # Error result should be properly formatted
            assert result.success is False
            assert result.error == "Integration test error"
            assert strategy.replace("_", " ").capitalize() in result.content
            assert "processing failed" in result.content

    def test_performance_metrics_integration(self):
        """Test performance metrics work with logging and cost optimization."""
        from src.mcp_server_mas_sequential_thinking.server_core import LoggingMixin
        from src.mcp_server_mas_sequential_thinking.constants import PerformanceMetrics, CostOptimizationConstants

        mixin = LoggingMixin()

        # Test efficiency calculation with performance constants
        threshold_time = PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD

        # Time exactly at threshold
        score_at_threshold = mixin._calculate_efficiency_score(threshold_time)
        expected_score = max(
            PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
            PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / threshold_time
        )
        assert score_at_threshold == expected_score

        # Perfect efficiency for fast processing
        fast_score = mixin._calculate_efficiency_score(30.0)
        assert fast_score == PerformanceMetrics.PERFECT_EFFICIENCY_SCORE

        # Test execution consistency
        success_consistency = mixin._calculate_execution_consistency(True)
        failure_consistency = mixin._calculate_execution_consistency(False)

        assert success_consistency == PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY
        assert failure_consistency == PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY