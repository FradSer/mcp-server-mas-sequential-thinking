"""Refactored ThoughtProcessor using dependency injection and single responsibility services.

This module provides a clean, maintainable implementation of thought processing
that delegates responsibilities to specialized services following clean architecture.
"""

import time

from mcp_server_mas_sequential_thinking.config import ProcessingDefaults
from mcp_server_mas_sequential_thinking.core import (
    ProcessingMetadata,
    SessionMemory,
    ThoughtData,
    ThoughtProcessingError,
)
from mcp_server_mas_sequential_thinking.infrastructure import (
    MetricsLogger,
    PerformanceTracker,
)
from mcp_server_mas_sequential_thinking.utils import setup_logging

from .context_builder import ContextBuilder
from .processing_orchestrator import ProcessingOrchestrator
from .response_formatter import ResponseFormatter
from .response_processor import ResponseProcessor
from .retry_handler import TeamProcessingRetryHandler
from .workflow_executor import WorkflowExecutor

logger = setup_logging()


class ThoughtProcessor:
    """Refactored thought processor using dependency injection and clean architecture.

    This class orchestrates thought processing by delegating specific responsibilities
    to specialized services, maintaining a clean separation of concerns.
    """

    __slots__ = (
        "_context_builder",
        "_metrics_logger",
        "_performance_tracker",
        "_processing_orchestrator",
        "_response_formatter",
        "_session",
        "_workflow_executor",
    )

    def __init__(self, session: SessionMemory) -> None:
        """Initialize the thought processor with dependency injection.

        Args:
            session: The session memory instance for accessing team and context
        """
        self._session = session

        # Initialize core services with dependency injection
        self._context_builder = ContextBuilder(session)
        self._workflow_executor = WorkflowExecutor(session)
        self._response_formatter = ResponseFormatter()

        # Initialize supporting services
        response_processor = ResponseProcessor()
        retry_handler = TeamProcessingRetryHandler()
        self._processing_orchestrator = ProcessingOrchestrator(
            session, response_processor, retry_handler
        )

        # Initialize monitoring services
        self._metrics_logger = MetricsLogger()
        self._performance_tracker = PerformanceTracker()

        logger.info("ThoughtProcessor initialized with specialized services")

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the appropriate workflow with comprehensive error handling.

        This is the main public API method that maintains backward compatibility
        while using the new service-based architecture internally.

        Args:
            thought_data: The thought data to process

        Returns:
            Processed thought response

        Raises:
            ThoughtProcessingError: If processing fails
        """
        try:
            return await self._process_thought_internal(thought_data)
        except Exception as e:
            error_msg = f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thoughtNumber}: {e}"
            logger.error(error_msg, exc_info=True)
            metadata: ProcessingMetadata = {
                "error_count": ProcessingDefaults.ERROR_COUNT_INITIAL,
                "retry_count": ProcessingDefaults.RETRY_COUNT_INITIAL,
                "processing_time": ProcessingDefaults.PROCESSING_TIME_INITIAL,
            }
            raise ThoughtProcessingError(error_msg, metadata) from e

    async def _process_thought_internal(self, thought_data: ThoughtData) -> str:
        """Internal thought processing logic using specialized services.

        Args:
            thought_data: The thought data to process

        Returns:
            Processed thought response
        """
        start_time = time.time()

        # Log thought data and add to session (now async for thread safety)
        self._log_thought_data(thought_data)
        await self._session.add_thought(thought_data)

        # Build context using specialized service (now async for thread safety)
        input_prompt = await self._context_builder.build_context_prompt(thought_data)
        await self._context_builder.log_context_building(thought_data, input_prompt)

        # Execute Multi-Thinking workflow using specialized service
        (
            content,
            workflow_result,
            total_time,
        ) = await self._workflow_executor.execute_workflow(
            thought_data, input_prompt, start_time
        )

        # Format response using specialized service
        final_response = self._response_formatter.format_response(content, thought_data)

        # Log workflow completion
        self._workflow_executor.log_workflow_completion(
            thought_data, workflow_result, total_time, final_response
        )

        return final_response

    def _log_thought_data(self, thought_data: ThoughtData) -> None:
        """Log comprehensive thought data information using centralized logger.

        Args:
            thought_data: The thought data to log
        """
        basic_info = {
            f"Thought #{thought_data.thoughtNumber}": f"{thought_data.thoughtNumber}/{thought_data.totalThoughts}",
            "Type": thought_data.thought_type.value,
            "Content": thought_data.thought,
            "Next needed": thought_data.nextThoughtNeeded,
            "Needs more": thought_data.needsMoreThoughts,
        }

        # Add conditional fields
        if thought_data.isRevision:
            basic_info["Is revision"] = (
                f"True (revises thought #{thought_data.branchFromThought})"
            )
        if thought_data.branchFromThought:
            basic_info["Branch from"] = (
                f"#{thought_data.branchFromThought} (ID: {thought_data.branchId})"
            )

        basic_info["Raw data"] = thought_data.format_for_log()

        self._metrics_logger.log_metrics_block("🧩 THOUGHT DATA:", basic_info)

        # Use field length limits constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import FieldLengthLimits

            separator_length = FieldLengthLimits.SEPARATOR_LENGTH
        except ImportError:
            pass

        self._metrics_logger.log_separator(separator_length)

    # Legacy methods for backward compatibility - these delegate to orchestrator
    async def _execute_single_agent_processing(
        self, input_prompt: str, routing_decision=None
    ) -> str:
        """Legacy method - delegates to orchestrator for backward compatibility."""
        return await self._processing_orchestrator.execute_single_agent_processing(
            input_prompt, simplified=True
        )

    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Legacy method - delegates to orchestrator for backward compatibility."""
        return await self._processing_orchestrator.execute_team_processing(input_prompt)

    def _extract_response_content(self, response) -> str:
        """Legacy method - delegates to formatter for backward compatibility."""
        return self._response_formatter.extract_response_content(response)

    def _build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Legacy method - delegates to context builder for backward compatibility."""
        return self._context_builder.build_context_prompt(thought_data)

    def _format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Legacy method - delegates to formatter for backward compatibility."""
        return self._response_formatter.format_response(content, thought_data)
