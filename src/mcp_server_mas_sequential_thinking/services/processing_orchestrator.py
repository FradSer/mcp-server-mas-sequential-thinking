"""Processing orchestration service for thought coordination.

This service handles the orchestration of different processing strategies,
coordinating between single-agent and multi-agent approaches, and managing
execution flows based on complexity analysis.
"""

import time
from typing import TYPE_CHECKING

from agno.agent import Agent

from mcp_server_mas_sequential_thinking.config import get_model_config
from mcp_server_mas_sequential_thinking.core import (
    SessionMemory,
    ThoughtProcessingError,
)
from mcp_server_mas_sequential_thinking.utils import setup_logging

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.routing import ComplexityLevel
    from mcp_server_mas_sequential_thinking.services.response_processor import (
        ResponseProcessor,
    )
    from mcp_server_mas_sequential_thinking.services.retry_handler import (
        TeamProcessingRetryHandler,
    )

logger = setup_logging()


class ProcessingOrchestrator:
    """Service responsible for orchestrating different processing strategies."""

    def __init__(
        self,
        session: SessionMemory,
        response_processor: "ResponseProcessor",
        retry_handler: "TeamProcessingRetryHandler",
    ) -> None:
        """Initialize the processing orchestrator.

        Args:
            session: The session memory instance
            response_processor: Response processing service
            retry_handler: Retry handling service
        """
        self._session = session
        self._response_processor = response_processor
        self._retry_handler = retry_handler

        # Initialize performance tracking
        from mcp_server_mas_sequential_thinking.infrastructure import (
            MetricsLogger,
            PerformanceTracker,
        )

        self._metrics_logger = MetricsLogger()
        self._performance_tracker = PerformanceTracker()

    async def execute_single_agent_processing(
        self, input_prompt: str, simplified: bool = False
    ) -> str:
        """Execute single-agent processing for simple thoughts.

        Args:
            input_prompt: The input prompt to process
            simplified: Whether to use simplified prompt format

        Returns:
            Processed response content
        """
        try:
            # Create a lightweight agent for single processing
            simple_agent = self._create_simple_agent(
                processing_type="simple thought" if simplified else "thought",
                use_markdown=not simplified,
            )

            # Optionally simplify the prompt
            prompt_to_use = (
                self._create_simplified_prompt(input_prompt)
                if simplified
                else input_prompt
            )

            # Log single agent call details
            self._log_single_agent_call(simple_agent, prompt_to_use)

            start_time = time.time()
            response = await simple_agent.arun(prompt_to_use)
            processing_time = time.time() - start_time

            # Extract content from Agno RunOutput
            response_content = self._extract_response_content(response)

            # Log single agent response details
            self._log_single_agent_response(response_content, processing_time)

            logger.info("Single-agent processing completed successfully")
            return response_content

        except Exception as e:
            logger.warning(f"Single-agent processing failed, falling back to team: {e}")
            # Fallback to team processing
            return await self.execute_team_processing(input_prompt)

    async def execute_team_processing(self, input_prompt: str) -> str:
        """Execute team processing without timeout restrictions.

        Args:
            input_prompt: The input prompt to process

        Returns:
            Processed response content
        """
        try:
            response = await self._session.team.arun(input_prompt)
            return self._extract_response_content(response)
        except Exception as e:
            raise ThoughtProcessingError(f"Team coordination failed: {e}") from e

    async def execute_team_processing_with_retries(
        self, input_prompt: str, complexity_level: "ComplexityLevel"
    ) -> str:
        """Execute team processing using centralized retry handler.

        Args:
            input_prompt: The input prompt to process
            complexity_level: Complexity level for retry strategy

        Returns:
            Processed response content
        """
        team_info = self._get_team_info()
        self._metrics_logger.log_team_details(team_info)
        self._metrics_logger.log_input_details(input_prompt)

        async def team_operation():
            start_time = time.time()
            response = await self._session.team.arun(input_prompt)
            processing_time = time.time() - start_time

            processed_response = self._response_processor.process_response(
                response, processing_time, "MULTI-AGENT TEAM"
            )

            self._performance_tracker.record_processing(processing_time, True)
            return processed_response.content

        return await self._retry_handler.execute_team_processing(
            team_operation, team_info, complexity_level.value
        )

    def _create_simple_agent(
        self, processing_type: str = "thought", use_markdown: bool = False
    ) -> Agent:
        """Create a simple agent for single-thought processing.

        Args:
            processing_type: Type of processing for instructions
            use_markdown: Whether to enable markdown formatting

        Returns:
            Configured Agent instance
        """
        model_config = get_model_config()
        single_model = model_config.create_team_model()

        return Agent(
            name="SimpleProcessor",
            role="Simple Thought Processor",
            description=f"Processes {processing_type}s efficiently without multi-agent overhead",
            model=single_model,
            instructions=[
                f"You are processing a {processing_type} efficiently.",
                "Provide a focused, clear response.",
                "Include guidance for the next step.",
                "Be concise but helpful.",
            ],
            markdown=use_markdown,
        )

    def _create_simplified_prompt(self, input_prompt: str) -> str:
        """Create a simplified prompt for single-agent processing.

        Args:
            input_prompt: The original input prompt

        Returns:
            Simplified prompt
        """
        return f"""Process this thought efficiently:

{input_prompt}

Provide a focused response with clear guidance for the next step."""

    def _extract_response_content(self, response) -> str:
        """Extract clean content from Agno RunOutput objects.

        Args:
            response: The response object from agent processing

        Returns:
            Extracted text content
        """
        from mcp_server_mas_sequential_thinking.services.response_formatter import (
            ResponseExtractor,
        )

        return ResponseExtractor.extract_content(response)

    def _get_team_info(self) -> dict:
        """Extract team information for logging and retry handling.

        Returns:
            Dictionary containing team information
        """
        team = self._session.team
        return {
            "name": team.name,
            "member_count": len(team.members),
            "leader_class": team.model.__class__.__name__,
            "leader_model": getattr(team.model, "id", "unknown"),
            "member_names": ", ".join([m.name for m in team.members]),
        }

    def _log_single_agent_call(self, agent: Agent, prompt: str) -> None:
        """Log single agent call details.

        Args:
            agent: The agent being used
            prompt: The prompt being processed
        """
        logger.info("🤖 SINGLE-AGENT CALL:")
        logger.info(f"  Agent: {agent.name} ({agent.role})")
        logger.info(
            f"  Model: {getattr(agent.model, 'id', 'unknown')} ({agent.model.__class__.__name__})"
        )
        self._log_input_details(prompt)

    def _log_single_agent_response(self, content: str, processing_time: float) -> None:
        """Log single agent response details.

        Args:
            content: The response content
            processing_time: Processing time in seconds
        """
        logger.info("✅ SINGLE-AGENT RESPONSE:")
        self._log_output_details(content, processing_time)

    def _log_input_details(
        self, input_prompt: str, context_description: str = "input"
    ) -> None:
        """Log input details with consistent formatting.

        Args:
            input_prompt: The input prompt to log
            context_description: Description of the context
        """
        logger.info(f"  Input length: {len(input_prompt)} chars")
        logger.info(f"  Full {context_description}:\\n{input_prompt}")

        # Use performance metrics constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import PerformanceMetrics

            separator_length = PerformanceMetrics.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * separator_length}")

    def _log_output_details(
        self,
        response_content: str,
        processing_time: float,
        context_description: str = "response",
    ) -> None:
        """Log output details with consistent formatting.

        Args:
            response_content: The response content to log
            processing_time: Processing time in seconds
            context_description: Description of the context
        """
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Output length: {len(response_content)} chars")
        logger.info(f"  Full {context_description}:\\n{response_content}")

        # Use performance metrics constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import PerformanceMetrics

            separator_length = PerformanceMetrics.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * separator_length}")
