"""
Tests for Agno-compliant workflow router.

Tests the AgnoCompliantRouter implementation and its integration with the ThoughtProcessor.
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from src.mcp_server_mas_sequential_thinking.agno_workflow_router import (
    AgnoWorkflowRouter,
    WorkflowResult,
)
from src.mcp_server_mas_sequential_thinking.models import ThoughtData
from src.mcp_server_mas_sequential_thinking.routing.complexity_types import (
    ComplexityLevel,
)


@pytest.fixture
def mock_model_config():
    """Mock model configuration."""
    mock_config = Mock()
    mock_model = Mock()
    mock_model.id = "test-model"
    mock_model.__class__.__name__ = "TestModel"
    mock_config.create_team_model.return_value = mock_model
    return mock_config


@pytest.fixture
def sample_thought_data():
    """Sample thought data for testing."""
    return ThoughtData(
        thought="What are the implications of artificial intelligence on society?",
        thoughtNumber=1,
        totalThoughts=5,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )


@pytest.fixture
def agno_router(mock_model_config):
    """Create AgnoWorkflowRouter instance for testing."""
    with patch(
        "src.mcp_server_mas_sequential_thinking.agno_workflow_router.get_model_config"
    ) as mock_get_config:
        mock_get_config.return_value = mock_model_config
        return AgnoWorkflowRouter()


class TestAgnoWorkflowRouter:
    """Test cases for AgnoWorkflowRouter."""

    def test_initialization(self, agno_router):
        """Test router initialization."""
        assert agno_router is not None
        assert agno_router.complexity_router is not None
        assert agno_router.workflow is not None
        assert agno_router.single_agent_step is not None
        assert agno_router.hybrid_team_step is not None
        assert agno_router.full_team_step is not None
        # Note: parallel_analysis_step removed in simplification

    def test_complexity_selector_simple(self, agno_router):
        """Test complexity selector for simple thoughts."""
        from agno.workflow.types import StepInput

        # Create a shared mutable dictionary for session_state
        session_state = {}

        # Mock StepInput with simple thought
        input_data = Mock(spec=StepInput)
        input_data.input = {
            "thought": "Hello world",
            "thought_number": 1,
            "total_thoughts": 5,
        }
        input_data.session_state = session_state

        # Call selector
        result = agno_router._original_complexity_selector(input_data)

        # Should return single agent step for simple thought
        assert len(result) == 1
        assert result[0] == agno_router.single_agent_step

        # Note: Selector doesn't modify session_state, cache is set by executors
        # Just verify the correct routing decision was made


    def test_complexity_selector_moderate(self, agno_router):
        """Test complexity selector for moderate complexity thoughts."""
        from agno.workflow.types import StepInput

        # Create a shared mutable dictionary for session_state
        session_state = {}

        # Mock StepInput with moderate complexity thought
        input_data = Mock(spec=StepInput)
        input_data.input = {
            "thought": "What are the technical implications of machine learning algorithms in healthcare diagnostics?",
            "thought_number": 1,
            "total_thoughts": 5,
        }
        input_data.session_state = session_state

        # Call selector
        result = agno_router._original_complexity_selector(input_data)

        # Should return hybrid team step for moderate thought
        assert len(result) == 1
        assert result[0] == agno_router.hybrid_team_step

        # Note: Selector doesn't modify session_state, cache is set by executors
        # Just verify the correct routing decision was made

    def test_complexity_selector_complex(self, agno_router):
        """Test complexity selector for complex thoughts."""
        from agno.workflow.types import StepInput

        # Create a shared mutable dictionary for session_state
        session_state = {}

        # Mock StepInput with complex thought (not quite highly complex)
        input_data = Mock(spec=StepInput)
        input_data.input = {
            "thought": "Analyze the multifaceted implications of artificial intelligence on economic systems, considering technological unemployment and wealth distribution.",
            "thought_number": 1,
            "total_thoughts": 5,
        }
        input_data.session_state = session_state

        # Call selector
        result = agno_router._original_complexity_selector(input_data)

        # Note: Selector doesn't modify session_state, cache is set by executors
        # Just verify the correct routing decision was made
        # Should return either hybrid or multi_agent for complex thought
        assert len(result) == 1
        assert result[0] in [agno_router.hybrid_team_step, agno_router.full_team_step]

    def test_complexity_selector_highly_complex(self, agno_router):
        """Test complexity selector for highly complex thoughts."""
        from agno.workflow.types import StepInput

        # Create a shared mutable dictionary for session_state
        session_state = {}

        # Mock StepInput with highly complex thought
        input_data = Mock(spec=StepInput)
        input_data.input = {
            "thought": """Analyze the multifaceted implications of artificial intelligence on economic systems,
            considering technological unemployment, wealth distribution, regulatory frameworks, ethical considerations,
            international competitiveness, and long-term societal transformation patterns. What are the potential
            policy interventions needed? How might different stakeholders respond? What are the unintended consequences?
            Consider the impact on labor markets, education systems, social safety nets, and democratic institutions.""",
            "thought_number": 1,
            "total_thoughts": 5,
        }
        input_data.session_state = session_state

        # Call selector
        result = agno_router._original_complexity_selector(input_data)

        # Should return full team step for highly complex thought (simplified architecture)
        assert len(result) == 1
        assert result[0] == agno_router.full_team_step

        # Note: Selector doesn't modify session_state, cache is set by executors
        # Just verify the correct routing decision was made

    def test_complexity_selector_error_handling(self, agno_router):
        """Test complexity selector error handling."""
        from agno.workflow.types import StepInput

        # Mock StepInput with invalid data
        input_data = Mock(spec=StepInput)
        input_data.input = {}  # Missing required fields
        session_state = {}
        input_data.session_state = session_state

        # Call selector - should fallback to emergency fallback
        result = agno_router._original_complexity_selector(input_data)

        assert len(result) == 1
        # With simplified retry mechanism, should still return a valid step
        # (either from retry success or final fallback to single_agent_step)
        assert result[0] in [
            agno_router.single_agent_step,
            agno_router.hybrid_team_step,
            agno_router.full_team_step
        ]

    def test_determine_complexity_level(self, agno_router):
        """Test complexity level determination."""
        assert agno_router._determine_complexity_level(2) == ComplexityLevel.SIMPLE
        assert agno_router._determine_complexity_level(10) == ComplexityLevel.MODERATE
        assert agno_router._determine_complexity_level(20) == ComplexityLevel.COMPLEX
        assert (
            agno_router._determine_complexity_level(30)
            == ComplexityLevel.HIGHLY_COMPLEX
        )

    @pytest.mark.asyncio
    async def test_process_thought_workflow_mock(
        self, agno_router, sample_thought_data
    ):
        """Test workflow processing with mocked workflow execution."""
        # Mock workflow execution
        mock_result = Mock()
        mock_result.content = "Test response from workflow"
        agno_router.workflow.arun = AsyncMock(return_value=mock_result)

        # Process thought
        result = await agno_router.process_thought_workflow(
            sample_thought_data, "Test context prompt"
        )

        # Verify result
        assert isinstance(result, WorkflowResult)
        assert result.content == "Test response from workflow"
        assert result.processing_time > 0
        assert result.step_name == "workflow_execution"

    @pytest.mark.asyncio
    async def test_process_thought_workflow_error(
        self, agno_router, sample_thought_data
    ):
        """Test workflow processing error handling."""
        # Mock workflow execution to raise error
        agno_router.workflow.arun = AsyncMock(side_effect=Exception("Workflow error"))

        # Process thought
        result = await agno_router.process_thought_workflow(
            sample_thought_data, "Test context prompt"
        )

        # Should return error result
        assert isinstance(result, WorkflowResult)
        assert "Error processing thought" in result.content
        assert result.strategy_used == "error_fallback"
        assert result.step_name == "error_handling"


class TestWorkflowIntegration:
    """Test integration with ThoughtProcessor."""

    @pytest.mark.asyncio
    async def test_thought_processor_workflow_integration(self):
        """Test ThoughtProcessor integration with workflow."""
        from src.mcp_server_mas_sequential_thinking.server_core import ThoughtProcessor
        from src.mcp_server_mas_sequential_thinking.session import SessionMemory

        # Create mock session
        mock_team = Mock()
        mock_team.name = "TestTeam"

        with patch(
            "src.mcp_server_mas_sequential_thinking.unified_team.create_team_by_type"
        ) as mock_create_team:
            mock_create_team.return_value = mock_team
            session = SessionMemory(mock_team)

        # Create processor (always uses Agno workflow)
        processor = ThoughtProcessor(session)

        # Verify workflow router is initialized
        assert processor._agno_router is not None
