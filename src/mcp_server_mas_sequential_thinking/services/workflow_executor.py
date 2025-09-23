"""Workflow execution service for multi-thinking processing.

This service handles the execution of multi-thinking workflows,
managing workflow routing and coordination of processing strategies.
"""

import time
from typing import TYPE_CHECKING, Any

from mcp_server_mas_sequential_thinking.core import SessionMemory, ThoughtData
from mcp_server_mas_sequential_thinking.utils import setup_logging

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.routing import MultiThinkingWorkflowResult

logger = setup_logging()


class WorkflowExecutor:
    """Service responsible for executing multi-thinking workflows."""

    def __init__(self, session: SessionMemory) -> None:
        """Initialize the workflow executor with session memory.

        Args:
            session: The session memory instance for accessing team and context
        """
        self._session = session
        self._agno_router: Any = None
        self._initialize_multi_thinking_workflow()

    def _initialize_multi_thinking_workflow(self) -> None:
        """Initialize multi-thinking workflow router.

        Uses dynamic import to avoid circular dependency issues.
        """
        logger.info("Initializing Multi-Thinking Workflow Router")
        # Dynamic import to avoid circular dependency
        from mcp_server_mas_sequential_thinking.routing import (
            MultiThinkingWorkflowRouter,
        )

        self._agno_router = MultiThinkingWorkflowRouter()
        logger.info("✅ Multi-Thinking Workflow Router ready")

    async def execute_workflow(
        self, thought_data: ThoughtData, input_prompt: str, start_time: float
    ) -> tuple[str, "MultiThinkingWorkflowResult", float]:
        """Execute multi-thinking workflow for the given thought.

        Args:
            thought_data: The thought data to process
            input_prompt: The context-aware input prompt
            start_time: Processing start time for metrics

        Returns:
            Tuple of (final_response, workflow_result, total_time)
        """
        # Execute multi-thinking workflow
        workflow_result: MultiThinkingWorkflowResult = (
            await self._agno_router.process_thought_workflow(thought_data, input_prompt)
        )

        total_time = time.time() - start_time

        return workflow_result.content, workflow_result, total_time

    def log_workflow_completion(
        self,
        thought_data: ThoughtData,
        workflow_result: "MultiThinkingWorkflowResult",
        total_time: float,
        final_response: str,
    ) -> None:
        """Log workflow completion with multi-thinking specific metrics.

        Args:
            thought_data: The processed thought data
            workflow_result: The workflow execution result
            total_time: Total processing time
            final_response: The final formatted response
        """
        # Basic completion info
        completion_metrics = {
            f"Thought #{thought_data.thoughtNumber}": "processed successfully",
            "Strategy": workflow_result.strategy_used,
            "Complexity Score": f"{workflow_result.complexity_score:.1f}/100",
            "Step": workflow_result.step_name,
            "Processing time": f"{workflow_result.processing_time:.3f}s",
            "Total time": f"{total_time:.3f}s",
            "Response length": f"{len(final_response)} chars",
        }

        self._log_metrics_block("🧠 MULTI-THINKING WORKFLOW COMPLETION:", completion_metrics)
        self._log_separator()

        # Performance metrics
        execution_consistency = self._calculate_execution_consistency(
            workflow_result.strategy_used != "error_fallback"
        )
        efficiency_score = self._calculate_efficiency_score(
            workflow_result.processing_time
        )

        performance_metrics = {
            "Execution Consistency": execution_consistency,
            "Efficiency Score": efficiency_score,
            "Response Length": f"{len(final_response)} chars",
            "Strategy Executed": workflow_result.strategy_used,
        }
        self._log_metrics_block("📊 WORKFLOW PERFORMANCE METRICS:", performance_metrics)

    def _log_metrics_block(self, title: str, metrics: dict[str, Any]) -> None:
        """Log a formatted metrics block.

        Args:
            title: The title for the metrics block
            metrics: Dictionary of metrics to log
        """
        logger.info(f"{title}")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            elif isinstance(value, (int, str)):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")

    def _log_separator(self, length: int = 60) -> None:
        """Log a separator line.

        Args:
            length: Length of the separator line
        """
        # Try to use performance metrics constant if available
        try:
            from mcp_server_mas_sequential_thinking.config import PerformanceMetrics

            length = PerformanceMetrics.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * length}")

    def _calculate_efficiency_score(self, processing_time: float) -> float:
        """Calculate efficiency score using standard metrics.

        Args:
            processing_time: The processing time in seconds

        Returns:
            Efficiency score between 0 and 1
        """
        # Use constants if available, otherwise use fallback values
        try:
            from mcp_server_mas_sequential_thinking.config import PerformanceMetrics

            perfect_score = PerformanceMetrics.PERFECT_EFFICIENCY_SCORE
            threshold = PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD
            minimum_score = PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE
        except ImportError:
            perfect_score = 1.0
            threshold = 5.0
            minimum_score = 0.1

        return (
            perfect_score
            if processing_time < threshold
            else max(minimum_score, threshold / processing_time)
        )

    def _calculate_execution_consistency(self, success_indicator: bool) -> float:
        """Calculate execution consistency using standard metrics.

        Args:
            success_indicator: Whether execution was successful

        Returns:
            Execution consistency score
        """
        try:
            from mcp_server_mas_sequential_thinking.config import PerformanceMetrics

            perfect_consistency = PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY
            default_consistency = PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        except ImportError:
            perfect_consistency = 1.0
            default_consistency = 0.5

        return perfect_consistency if success_indicator else default_consistency
