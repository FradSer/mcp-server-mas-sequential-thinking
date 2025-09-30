"""Integration tests for thought processing workflow.

Tests the complete thought processing pipeline including validation,
routing, and response generation.
"""

import pytest
from unittest.mock import patch, MagicMock

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.core.session import SessionMemory
from mcp_server_mas_sequential_thinking.services.thought_processor_refactored import ThoughtProcessor
from tests.fixtures.test_data import VALID_THOUGHTS, SAMPLE_API_KEYS


class TestThoughtProcessingIntegration:
    """Integration tests for the complete thought processing workflow."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.session = SessionMemory()

    @pytest.mark.asyncio
    async def test_complete_thought_processing_workflow(self):
        """Test the complete workflow from thought input to response."""
        # Mock the workflow router to avoid actual AI calls
        with patch('mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter') as mock_router:
            # Set up mock response
            mock_result = MagicMock()
            mock_result.content = "This is a comprehensive analysis of the problem."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 45.0
            mock_result.step_name = "synthesis"
            mock_result.processing_time = 2.5

            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.return_value = mock_result

            # Create thought processor
            processor = ThoughtProcessor(self.session)

            # Create valid thought data
            sample_thought = VALID_THOUGHTS[0]
            thought_data = ThoughtData(
                thought=sample_thought.thought,
                thoughtNumber=sample_thought.thoughtNumber,
                totalThoughts=sample_thought.totalThoughts,
                nextThoughtNeeded=sample_thought.nextThoughtNeeded,
                isRevision=sample_thought.isRevision,
                branchFromThought=sample_thought.branchFromThought,
                branchId=sample_thought.branchId,
                needsMoreThoughts=sample_thought.needsMoreThoughts,
            )

            # Process the thought
            result = await processor.process_thought(thought_data)

            # Verify response
            assert isinstance(result, str)
            assert len(result) > 0
            assert "comprehensive analysis" in result.lower()

            # Verify workflow was called
            mock_router_instance.process_thought_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_memory_integration(self):
        """Test that thoughts are properly stored in session memory."""
        with patch('mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter') as mock_router:
            # Set up mock response
            mock_result = MagicMock()
            mock_result.content = "Analysis complete."
            mock_result.strategy_used = "single_agent"
            mock_result.complexity_score = 25.0
            mock_result.step_name = "analysis"
            mock_result.processing_time = 1.0

            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.return_value = mock_result

            processor = ThoughtProcessor(self.session)

            # Process multiple thoughts
            for sample_thought in VALID_THOUGHTS:
                thought_data = ThoughtData(
                    thought=sample_thought.thought,
                    thoughtNumber=sample_thought.thoughtNumber,
                    totalThoughts=sample_thought.totalThoughts,
                    nextThoughtNeeded=sample_thought.nextThoughtNeeded,
                    isRevision=sample_thought.isRevision,
                    branchFromThought=sample_thought.branchFromThought,
                    branchId=sample_thought.branchId,
                    needsMoreThoughts=sample_thought.needsMoreThoughts,
                )

                await processor.process_thought(thought_data)

            # Verify thoughts are stored in session
            assert len(self.session._thoughts) == len(VALID_THOUGHTS)

            # Test thought retrieval
            for i, thought in enumerate(VALID_THOUGHTS, 1):
                stored_content = await self.session.find_thought_content(i)
                assert stored_content == thought.thought

    @pytest.mark.asyncio
    async def test_context_building_integration(self):
        """Test that context is properly built from session history."""
        with patch('mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter') as mock_router:
            mock_result = MagicMock()
            mock_result.content = "Context-aware response."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 35.0
            mock_result.step_name = "synthesis"
            mock_result.processing_time = 2.0

            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.return_value = mock_result

            processor = ThoughtProcessor(self.session)

            # Add some context to the session first
            context_thought = ThoughtData(
                thought="I need to understand the fundamentals first.",
                thoughtNumber=1,
                totalThoughts=2,
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            await processor.process_thought(context_thought)

            # Now process a follow-up thought
            followup_thought = ThoughtData(
                thought="Building on the previous analysis, what are the next steps?",
                thoughtNumber=2,
                totalThoughts=2,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            result = await processor.process_thought(followup_thought)

            # Verify that context was passed to the workflow
            call_args = mock_router_instance.process_thought_workflow.call_args
            thought_data_arg = call_args[0][0]
            input_prompt_arg = call_args[0][1]

            assert thought_data_arg.thoughtNumber == 2
            assert "fundamentals" in input_prompt_arg  # Context should include previous thought

    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling throughout the processing pipeline."""
        with patch('mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter') as mock_router:
            # Set up router to raise an exception
            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.side_effect = Exception("AI service unavailable")

            processor = ThoughtProcessor(self.session)

            thought_data = ThoughtData(
                thought="This will trigger an error.",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=False,
            )

            # Should raise ThoughtProcessingError
            from mcp_server_mas_sequential_thinking.core import ThoughtProcessingError
            with pytest.raises(ThoughtProcessingError):
                await processor.process_thought(thought_data)

    @pytest.mark.asyncio
    async def test_branching_workflow_integration(self):
        """Test branching workflow integration."""
        with patch('mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter') as mock_router:
            mock_result = MagicMock()
            mock_result.content = "Branch analysis complete."
            mock_result.strategy_used = "multi_agent"
            mock_result.complexity_score = 50.0
            mock_result.step_name = "branch_synthesis"
            mock_result.processing_time = 3.0

            mock_router_instance = mock_router.return_value
            mock_router_instance.process_thought_workflow.return_value = mock_result

            processor = ThoughtProcessor(self.session)

            # Create main thought
            main_thought = ThoughtData(
                thought="What are the different approaches to this problem?",
                thoughtNumber=1,
                totalThoughts=1,
                nextThoughtNeeded=True,
                isRevision=False,
                branchFromThought=None,
                branchId=None,
                needsMoreThoughts=True,
            )

            await processor.process_thought(main_thought)

            # Create branch thought
            branch_thought = ThoughtData(
                thought="Approach A: Use machine learning algorithms.",
                thoughtNumber=2,
                totalThoughts=2,
                nextThoughtNeeded=False,
                isRevision=False,
                branchFromThought=1,
                branchId="approach_a",
                needsMoreThoughts=False,
            )

            result = await processor.process_thought(branch_thought)

            # Verify branching context
            assert len(self.session._thoughts) == 2
            assert result is not None
            assert "branch analysis" in result.lower()


class TestConfigurationIntegration:
    """Integration tests for configuration and environment setup."""

    @patch.dict('os.environ', {'LLM_PROVIDER': 'deepseek', 'DEEPSEEK_API_KEY': SAMPLE_API_KEYS['valid']['deepseek']})
    def test_valid_environment_configuration(self):
        """Test that valid environment configuration works end-to-end."""
        from mcp_server_mas_sequential_thinking.config import validate_configuration_comprehensive

        result = validate_configuration_comprehensive()
        # Should not have any validation errors for required keys
        assert "DEEPSEEK_API_KEY" not in result or "Required but not set" not in result.get("DEEPSEEK_API_KEY", "")

    @patch.dict('os.environ', {}, clear=True)
    def test_missing_configuration_detection(self):
        """Test that missing configuration is properly detected."""
        from mcp_server_mas_sequential_thinking.config import validate_configuration_comprehensive

        result = validate_configuration_comprehensive("deepseek")
        assert "DEEPSEEK_API_KEY" in result
        assert "Required but not set" in result["DEEPSEEK_API_KEY"]