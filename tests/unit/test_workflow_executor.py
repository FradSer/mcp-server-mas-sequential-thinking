"""Unit tests for services/workflow_executor.py WorkflowExecutor."""

from unittest.mock import MagicMock, patch

from mcp_server_mas_sequential_thinking.core.session import SessionMemory
from mcp_server_mas_sequential_thinking.services.workflow_executor import WorkflowExecutor


def make_workflow_result(
    content="result text",
    strategy_used="full_sequence",
    complexity_score=7.5,
    step_name="synthesis",
    processing_time=1.2,
) -> MagicMock:
    result = MagicMock()
    result.content = content
    result.strategy_used = strategy_used
    result.complexity_score = complexity_score
    result.step_name = step_name
    result.processing_time = processing_time
    return result


def make_executor() -> WorkflowExecutor:
    session = SessionMemory()
    mock_router = MagicMock()
    with patch(
        "mcp_server_mas_sequential_thinking.routing.MultiThinkingWorkflowRouter",
        return_value=mock_router,
    ):
        executor = WorkflowExecutor(session)
    executor._agno_router = mock_router
    return executor


class TestWorkflowExecutorCalculations:
    """Tests for WorkflowExecutor calculation methods."""

    def test_calculate_efficiency_score_fast(self):
        executor = make_executor()
        score = executor._calculate_efficiency_score(0.1)
        assert score > 0.0

    def test_calculate_efficiency_score_slow(self):
        executor = make_executor()
        score = executor._calculate_efficiency_score(100.0)
        assert 0.0 <= score <= 1.0

    def test_calculate_execution_consistency_success(self):
        executor = make_executor()
        score = executor._calculate_execution_consistency(True)
        assert score > 0.0

    def test_calculate_execution_consistency_failure(self):
        executor = make_executor()
        score = executor._calculate_execution_consistency(False)
        assert score >= 0.0

    def test_log_metrics_block(self):
        executor = make_executor()
        executor._log_metrics_block("Test Title", {"key1": "val", "key2": 3.14})

    def test_log_separator(self):
        executor = make_executor()
        executor._log_separator()

    def test_log_workflow_completion(self):
        from mcp_server_mas_sequential_thinking.core.models import ThoughtData
        executor = make_executor()
        thought = ThoughtData(
            thought="test",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=False,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        workflow_result = make_workflow_result()
        executor.log_workflow_completion(thought, workflow_result, 2.5, "final response")
