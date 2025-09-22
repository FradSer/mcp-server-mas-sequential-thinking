"""Refactored server core with separated concerns and reduced complexity."""

import os
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from agno.agent import Agent
from agno.team.team import Team
from pydantic import ValidationError

from .adaptive_routing import ComplexityLevel, ProcessingStrategy
from .agno_workflow_router import SixHatsWorkflowRouter, SixHatsWorkflowResult
from .constants import (
    DefaultTimeouts,
    DefaultValues,
    FieldLengthLimits,
    PerformanceMetrics,
    ProcessingDefaults,
)
from .metrics_logger import MetricsLogger, PerformanceTracker
from .models import ThoughtData
from .modernized_config import check_required_api_keys, get_model_config
from .response_processor import ResponseExtractor, ResponseProcessor
from .retry_handler import TeamProcessingRetryHandler
from .session import SessionMemory
from .types import (
    ConfigurationError,
    CoordinationPlan,
    ExecutionMode,
    ProcessingMetadata,
    TeamCreationError,
    ThoughtProcessingError,
)
from .unified_team import create_team_by_type
from .utils import setup_logging

logger = setup_logging()


class LoggingMixin:
    """Mixin class providing common logging utilities with reduced duplication."""

    @staticmethod
    def _log_section_header(
        title: str, separator_length: int = PerformanceMetrics.SEPARATOR_LENGTH
    ) -> None:
        """Log a formatted section header."""
        logger.info(f"{title}")

    @staticmethod
    def _log_metrics_block(title: str, metrics: dict[str, any]) -> None:
        """Log a formatted metrics block."""
        logger.info(f"{title}")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.2f}")
            elif isinstance(value, (int, str)):
                logger.info(f"  {key}: {value}")
            else:
                logger.info(f"  {key}: {value}")

    @staticmethod
    def _log_separator(length: int = PerformanceMetrics.SEPARATOR_LENGTH) -> None:
        """Log a separator line."""
        logger.info(f"  {'=' * length}")

    @staticmethod
    def _calculate_efficiency_score(processing_time: float) -> float:
        """Calculate efficiency score using standard metrics."""
        return (
            PerformanceMetrics.PERFECT_EFFICIENCY_SCORE
            if processing_time < PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD
            else max(
                PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
                PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / processing_time,
            )
        )

    @staticmethod
    def _calculate_execution_consistency(success_indicator: bool) -> float:
        """Calculate execution consistency using standard metrics."""
        return (
            PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY
            if success_indicator
            else PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        )


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration with clear defaults."""

    provider: str
    team_mode: str = DefaultValues.DEFAULT_TEAM_MODE
    log_level: str = DefaultValues.DEFAULT_LOG_LEVEL
    max_retries: int = DefaultValues.DEFAULT_MAX_RETRIES
    timeout: float = DefaultTimeouts.PROCESSING_TIMEOUT

    @classmethod
    def from_environment(cls) -> "ServerConfig":
        """Create config from environment with sensible defaults."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", DefaultValues.DEFAULT_LLM_PROVIDER),
            team_mode=os.environ.get(
                "TEAM_MODE", DefaultValues.DEFAULT_TEAM_MODE
            ).lower(),
            log_level=os.environ.get("LOG_LEVEL", DefaultValues.DEFAULT_LOG_LEVEL),
            max_retries=int(
                os.environ.get("MAX_RETRIES", str(DefaultValues.DEFAULT_MAX_RETRIES))
            ),
            timeout=float(
                os.environ.get("TIMEOUT", str(DefaultValues.DEFAULT_TIMEOUT))
            ),
        )


class ServerInitializer(ABC):
    """Abstract initializer for server components."""

    @abstractmethod
    async def initialize(self, config: ServerConfig) -> None:
        """Initialize server component."""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up server component."""


class EnvironmentInitializer(ServerInitializer):
    """Handles environment validation and setup."""

    async def initialize(self, config: ServerConfig) -> None:
        """Validate environment requirements with enhanced error handling."""
        logger.info(f"Initializing environment with {config.provider} provider")

        try:
            # Check required API keys
            missing_keys = check_required_api_keys()
            if missing_keys:
                logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
                # Note: Don't fail here as some providers might not require keys

            # Ensure log directory exists
            log_dir = Path.home() / ".sequential_thinking" / "logs"
            if not log_dir.exists():
                logger.info(f"Creating log directory: {log_dir}")
                try:
                    log_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise ConfigurationError(
                        f"Failed to create log directory {log_dir}: {e}"
                    ) from e

        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(
                    f"Environment initialization failed: {e}"
                ) from e
            raise

    async def cleanup(self) -> None:
        """No cleanup needed for environment."""


class TeamInitializer(ServerInitializer):
    """Handles team creation and configuration."""

    def __init__(self):
        self._team: Team | None = None

    async def initialize(self, config: ServerConfig) -> None:
        """Initialize team based on configuration with enhanced error handling."""
        logger.info(f"Creating {config.team_mode} team")

        try:
            # Create Six Hats team using unified factory
            model_config = get_model_config()

            # Map legacy team modes to Six Hats equivalents
            six_hats_mode_mapping = {
                "standard": "philosophical",
                "enhanced": "full",
                "hybrid": "creative",
                "enhanced_specialized": "decision",
            }

            # Use Six Hats mapping
            six_hats_mode = six_hats_mode_mapping.get(config.team_mode, config.team_mode)
            logger.info(f"Creating Six Hats team: {six_hats_mode} (from {config.team_mode})")

            self._team = create_team_by_type(six_hats_mode, model_config)

            if not self._team:
                raise TeamCreationError(
                    f"Failed to create Six Hats team of type '{six_hats_mode}'"
                )

            logger.info(f"Six Hats team initialized successfully: {self._team.name}")

        except Exception as e:
            if not isinstance(e, TeamCreationError):
                raise TeamCreationError(
                    f"Six Hats team initialization failed for type '{config.team_mode}': {e}"
                ) from e
            raise

    async def cleanup(self) -> None:
        """Clean up team resources."""
        self._team = None

    @property
    def team(self) -> Team:
        """Get initialized team."""
        if self._team is None:
            raise RuntimeError("Team not initialized")
        return self._team


class ServerState:
    """Manages server state with proper lifecycle and separation of concerns."""

    def __init__(self):
        self._config: ServerConfig | None = None
        self._session: SessionMemory | None = None
        self._initializers = [
            EnvironmentInitializer(),
            TeamInitializer(),
        ]
        self._team_initializer = self._initializers[
            ProcessingDefaults.TEAM_INITIALIZER_INDEX
        ]

    async def initialize(self, config: ServerConfig) -> None:
        """Initialize all server components."""
        self._config = config

        # Initialize all components in order
        for initializer in self._initializers:
            await initializer.initialize(config)

        # Create session with initialized team
        self._session = SessionMemory(team=self._team_initializer.team)

        logger.info("Server state initialized successfully")

    async def cleanup(self) -> None:
        """Clean up all server components."""
        # Clean up in reverse order
        for initializer in reversed(self._initializers):
            await initializer.cleanup()

        self._config = None
        self._session = None

        logger.info("Server state cleaned up")

    @property
    def config(self) -> ServerConfig:
        """Get current configuration."""
        if self._config is None:
            raise RuntimeError("Server not initialized - config unavailable")
        return self._config

    @property
    def session(self) -> SessionMemory:
        """Get current session."""
        if self._session is None:
            raise RuntimeError("Server not initialized - session unavailable")
        return self._session


# Remove redundant exception definition as it's now in types.py


class ThoughtProcessor(LoggingMixin):
    """Handles thought processing with optimized performance and error handling."""

    __slots__ = (
        "_agno_router",
        "_metrics_logger",
        "_performance_tracker",
        "_response_processor",
        "_retry_handler",
        "_session",
    )

    def __init__(self, session: SessionMemory) -> None:
        self._session = session

        # Initialize utilities
        self._retry_handler = TeamProcessingRetryHandler()
        self._response_processor = ResponseProcessor()
        self._metrics_logger = MetricsLogger()
        self._performance_tracker = PerformanceTracker()

        # Initialize Six Hats workflow (only option)
        self._initialize_six_hats_workflow()

    def _initialize_six_hats_workflow(self) -> None:
        """Initialize Six Thinking Hats workflow router."""
        logger.info("Initializing Six Hats Workflow Router")
        self._agno_router = SixHatsWorkflowRouter()
        logger.info("✅ Six Hats Workflow Router ready")

    def _extract_response_content(self, response) -> str:
        """Extract clean content from Agno RunOutput objects."""
        return ResponseExtractor.extract_content(response)

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        try:
            result = await self._process_thought_internal(thought_data)
            return result
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
        """Internal thought processing logic with structured logging."""
        start_time = time.time()

        self._log_thought_data(thought_data)
        self._session.add_thought(thought_data)

        # Always use Six Hats workflow processing
        return await self._process_with_six_hats_workflow(thought_data, start_time)

    async def _process_with_six_hats_workflow(
        self, thought_data: ThoughtData, start_time: float
    ) -> str:
        """Process thought using Six Thinking Hats workflow."""
        input_prompt = self._build_context_prompt(thought_data)
        self._log_context_building(thought_data, input_prompt)

        # Execute Six Hats workflow
        workflow_result: SixHatsWorkflowResult = (
            await self._agno_router.process_thought_workflow(thought_data, input_prompt)
        )

        final_response = self._format_response(workflow_result.content, thought_data)
        total_time = time.time() - start_time

        # Log workflow completion with Six Hats metrics
        self._log_workflow_completion(
            thought_data, workflow_result, total_time, final_response
        )

        return final_response

    def _log_input_details(
        self, input_prompt: str, context_description: str = "input"
    ) -> None:
        """Log input details with consistent formatting."""
        logger.info(f"  Input length: {len(input_prompt)} chars")
        logger.info(f"  Full {context_description}:\\n{input_prompt}")
        logger.info(f"  {'=' * PerformanceMetrics.SEPARATOR_LENGTH}")

    def _log_output_details(
        self,
        response_content: str,
        processing_time: float,
        context_description: str = "response",
    ) -> None:
        """Log output details with consistent formatting."""
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Output length: {len(response_content)} chars")
        logger.info(f"  Full {context_description}:\\n{response_content}")
        logger.info(f"  {'=' * PerformanceMetrics.SEPARATOR_LENGTH}")

    def _create_simple_agent(
        self, processing_type: str = "thought", use_markdown: bool = False
    ) -> Agent:
        """Create a simple agent for single-thought processing."""
        from agno.agent import Agent

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

    async def _create_coordination_plan(
        self, thought_data: ThoughtData
    ) -> tuple[CoordinationPlan, float]:
        """Create coordination plan from routing decision with timing."""
        coordination_start = time.time()
        routing_decision = self._router.route_thought(thought_data)
        coordination_plan = CoordinationPlan.from_routing_decision(
            routing_decision, thought_data
        )
        coordination_time = time.time() - coordination_start

        self._log_coordination_plan(coordination_plan, coordination_time)
        return coordination_plan, coordination_time

    def _log_coordination_plan(
        self, coordination_plan: CoordinationPlan, coordination_time: float
    ) -> None:
        """Log detailed coordination plan analysis."""
        logger.info("🎯 COORDINATION PLAN:")
        logger.info(f"  Strategy: {coordination_plan.strategy}")
        logger.info(
            f"  Complexity: {coordination_plan.complexity_level} (score: {coordination_plan.complexity_score:.1f}/100)"
        )
        logger.info(f"  Execution mode: {coordination_plan.execution_mode.value}")
        logger.info(f"  Specialists: {coordination_plan.specialist_roles}")
        logger.info(f"  Team size: {coordination_plan.team_size}")
        logger.info(f"  Coordination: {coordination_plan.coordination_strategy}")
        logger.info("  Processing mode: Unlimited time allowed")
        logger.info(f"  Coordination time: {coordination_time:.3f}s")
        logger.info(f"  Confidence: {coordination_plan.confidence:.2f}")
        logger.debug(f"  Reasoning: {coordination_plan.reasoning}")

    async def _execute_with_timing(
        self, input_prompt: str, coordination_plan: CoordinationPlan
    ) -> tuple[str, float]:
        """Execute coordination plan with timing and logging."""
        processing_start = time.time()

        logger.info("📋 EXECUTING COORDINATION PLAN:")
        logger.info(f"  Mode: {coordination_plan.execution_mode.value}")
        logger.info(f"  Task breakdown: {coordination_plan.task_breakdown}")
        logger.info(
            f"  Expected interactions: {coordination_plan.expected_interactions}"
        )

        response = await self._execute_coordination_plan(
            input_prompt, coordination_plan
        )
        processing_time = time.time() - processing_start

        return response, processing_time

    def _log_workflow_completion(
        self,
        thought_data: ThoughtData,
        workflow_result: SixHatsWorkflowResult,
        total_time: float,
        final_response: str,
    ) -> None:
        """Log workflow completion with Six Hats specific metrics."""
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
        self._log_metrics_block("🎩 SIX HATS WORKFLOW COMPLETION:", completion_metrics)
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

    def _log_completion_summary(
        self,
        thought_data: ThoughtData,
        coordination_plan: CoordinationPlan,
        processing_time: float,
        total_time: float,
        final_response: str,
    ) -> None:
        """Log performance metrics and completion summary using centralized logger."""
        self._metrics_logger.log_completion_summary(
            thought_data=thought_data,
            strategy=coordination_plan.execution_mode.value,
            specialists_count=len(coordination_plan.specialist_roles),
            processing_time=processing_time,
            total_time=total_time,
            confidence=coordination_plan.confidence,
            final_response=final_response,
        )

        # Record performance metrics for tracking
        self._performance_tracker.record_processing(processing_time, True)

    def _log_thought_data(self, thought_data: ThoughtData) -> None:
        """Log comprehensive thought data information."""
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

        self._log_metrics_block("🧩 THOUGHT DATA:", basic_info)
        self._log_separator(FieldLengthLimits.SEPARATOR_LENGTH)

    async def _analyze_routing(self, thought_data: ThoughtData) -> tuple[object, float]:
        """Analyze routing decision and return decision with timing."""
        routing_start = time.time()
        routing_decision = self._router.route_thought(thought_data)
        routing_time = time.time() - routing_start

        self._log_routing_analysis(routing_decision, routing_time)
        return routing_decision, routing_time

    def _log_routing_analysis(self, routing_decision, routing_time: float) -> None:
        """Log detailed routing analysis information."""
        logger.info("🧠 ROUTING ANALYSIS:")
        logger.info(f"  Strategy: {routing_decision.strategy.value}")
        logger.info(
            f"  Complexity: {routing_decision.complexity_level.value} (score: {routing_decision.complexity_score:.1f}/100)"
        )
        logger.info(
            f"  Estimated tokens: {routing_decision.estimated_token_usage[0]}-{routing_decision.estimated_token_usage[1]}"
        )
        logger.info(f"  Estimated cost: ${routing_decision.estimated_cost:.6f}")
        logger.info(f"  Routing time: {routing_time:.3f}s")

        if routing_decision.specialist_recommendations:
            logger.info(
                f"  Recommended specialists: {', '.join(routing_decision.specialist_recommendations)}"
            )
        logger.debug(f"  Reasoning: {routing_decision.reasoning}")

    def _log_context_building(
        self, thought_data: ThoughtData, input_prompt: str
    ) -> None:
        """Log context building details."""
        logger.info("📝 CONTEXT BUILDING:")

        if thought_data.isRevision and thought_data.branchFromThought:
            logger.info(f"  Type: Revision of thought #{thought_data.branchFromThought}")
            try:
                original = self._session.find_thought_content(
                    thought_data.branchFromThought
                )
                logger.info(f"  Original thought: {original}")
            except:
                logger.info("  Original thought: [not found]")
        elif thought_data.branchFromThought and thought_data.branchId:
            logger.info(
                f"  Type: Branch '{thought_data.branchId}' from thought #{thought_data.branchFromThought}"
            )
            try:
                origin = self._session.find_thought_content(thought_data.branchFromThought)
                logger.info(f"  Branch origin: {origin}")
            except:
                logger.info("  Branch origin: [not found]")
        else:
            logger.info(f"  Type: Sequential thought #{thought_data.thoughtNumber}")

        logger.info(f"  Session thoughts: {len(self._session.thought_history)} total")
        logger.info(f"  Input thought: {thought_data.thought}")
        logger.info(f"  Built prompt length: {len(input_prompt)} chars")
        logger.info(f"  Built prompt:\n{input_prompt}")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    async def _execute_processing_strategy(
        self, input_prompt: str, routing_decision
    ) -> tuple[str, str, float]:
        """Execute processing strategy and return response, strategy used, and processing time."""
        processing_start = time.time()

        if routing_decision.strategy == ProcessingStrategy.SINGLE_AGENT:
            response = await self._execute_single_agent_processing(
                input_prompt, routing_decision
            )
            strategy_used = "single_agent"
        else:
            response = await self._execute_team_processing_with_retries(
                input_prompt, routing_decision.complexity_level
            )
            strategy_used = "multi_agent"

        processing_time = time.time() - processing_start
        return response, strategy_used, processing_time

    def _log_performance_metrics(
        self,
        thought_data: ThoughtData,
        strategy_used: str,
        processing_time: float,
        total_time: float,
    ) -> None:
        """Log performance metrics for the processing."""
        logger.info(
            f"Thought #{thought_data.thoughtNumber} completed: "
            f"strategy={strategy_used}, "
            f"processing_time={processing_time:.3f}s, "
            f"total_time={total_time:.3f}s"
        )

    def _log_processing_completion(
        self,
        thought_data: ThoughtData,
        strategy_used: str,
        processing_time: float,
        total_time: float,
        final_response: str,
    ) -> None:
        """Log final processing completion summary."""
        logger.info("🎯 PROCESSING COMPLETE:")
        logger.info(f"  Thought #{thought_data.thoughtNumber} processed successfully")
        logger.info(f"  Strategy used: {strategy_used}")
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Response length: {len(final_response)} chars")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    async def _execute_coordination_plan(
        self, input_prompt: str, plan: CoordinationPlan
    ) -> str:
        """Execute thought processing based on coordination plan (unified approach)."""
        logger.info(
            f"🎯 Executing {plan.execution_mode.value} with {plan.specialist_roles}"
        )

        try:
            if plan.execution_mode == ExecutionMode.SINGLE_AGENT:
                return await self._execute_single_agent_simple(input_prompt)

            if plan.execution_mode == ExecutionMode.SELECTIVE_TEAM:
                return await self._execute_selective_team(input_prompt, plan)

            if plan.execution_mode == ExecutionMode.FULL_TEAM:
                return await self._execute_full_team_unlimited(input_prompt, plan)

            raise ValueError(f"Unknown execution mode: {plan.execution_mode}")

        except Exception as e:
            logger.error(f"Coordination plan execution failed: {e}")
            # Fallback to single agent for reliability
            logger.info("Falling back to single-agent processing")
            return await self._execute_single_agent_simple(input_prompt)

    async def _execute_single_agent_simple(self, input_prompt: str) -> str:
        """Execute simple single-agent processing."""
        simple_agent = self._create_simple_agent(
            processing_type="thought", use_markdown=False
        )

        logger.info("🤖 SINGLE-AGENT CALL:")
        logger.info(f"  Agent: {simple_agent.name} ({simple_agent.role})")
        logger.info(
            f"  Model: {getattr(simple_agent.model, 'id', 'unknown')} ({simple_agent.model.__class__.__name__})"
        )
        self._log_input_details(input_prompt)

        start_time = time.time()
        response = await simple_agent.arun(input_prompt)
        processing_time = time.time() - start_time

        # HOTFIX: Properly extract content from Agno RunOutput
        response_content = self._extract_response_content(response)

        logger.info("✅ SINGLE-AGENT RESPONSE:")
        self._log_output_details(response_content, processing_time)

        return response_content

    async def _execute_selective_team(
        self, input_prompt: str, plan: CoordinationPlan
    ) -> str:
        """Execute selective team processing (hybrid approach)."""
        # For now, delegate to full team but log as selective
        logger.info("🏢 SELECTIVE TEAM CALL:")
        logger.info(f"  Required specialists: {plan.specialist_roles}")
        logger.info(f"  Coordination strategy: {plan.coordination_strategy}")

        # TODO: Implement actual selective team creation
        # For now, use existing team without timeout
        return await self._execute_full_team_unlimited(input_prompt, plan)

    async def _execute_full_team_unlimited(
        self, input_prompt: str, plan: CoordinationPlan
    ) -> str:
        """Execute full team processing without timeout restrictions."""
        return await self._execute_team_processing_with_retries(
            input_prompt, plan.complexity_level
        )

    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Execute team processing without timeout restrictions."""
        try:
            # REMOVED: All timeout restrictions for unlimited processing time
            response = await self._session.team.arun(input_prompt)
            # HOTFIX: Properly extract content from Agno RunOutput
            return self._extract_response_content(response)
        except Exception as e:
            raise ThoughtProcessingError(f"Team coordination failed: {e}") from e

    async def _execute_team_processing_with_retries(
        self, input_prompt: str, complexity_level: ComplexityLevel
    ) -> str:
        """Execute team processing using centralized retry handler."""
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

    def _get_team_info(self) -> dict:
        """Extract team information for logging and retry handling."""
        team = self._session.team
        return {
            "name": team.name,
            "member_count": len(team.members),
            "leader_class": team.model.__class__.__name__,
            "leader_model": getattr(team.model, "id", "unknown"),
            "member_names": ", ".join([m.name for m in team.members]),
        }

    async def _execute_single_agent_processing(
        self, input_prompt: str, routing_decision
    ) -> str:
        """Execute single-agent processing for simple thoughts without timeout restrictions."""
        try:
            # Create a lightweight agent for single processing
            simple_agent = self._create_simple_agent(
                processing_type="simple thought", use_markdown=True
            )

            # Create a simple response without multi-agent overhead
            simplified_prompt = f"""Process this thought efficiently:

{input_prompt}

Provide a focused response with clear guidance for the next step."""

            # ENHANCED LOGGING: Log single agent call details
            logger.info("🤖 SINGLE-AGENT CALL:")
            logger.info(f"  Agent: {simple_agent.name} ({simple_agent.role})")
            logger.info(
                f"  Model: {getattr(simple_agent.model, 'id', 'unknown')} ({simple_agent.model.__class__.__name__})"
            )
            self._log_input_details(simplified_prompt)

            start_time = time.time()
            # REMOVED: All timeout restrictions for unlimited processing time
            response = await simple_agent.arun(simplified_prompt)
            processing_time = time.time() - start_time

            # HOTFIX: Properly extract content from Agno RunOutput
            response_content = self._extract_response_content(response)

            # ENHANCED LOGGING: Log single agent response details
            logger.info("✅ SINGLE-AGENT RESPONSE:")
            self._log_output_details(response_content, processing_time)

            logger.info(
                f"Single-agent processing completed (saved ~{routing_decision.estimated_cost:.4f}$ vs multi-agent)"
            )
            return response_content

        except Exception as e:
            logger.warning(f"Single-agent processing failed, falling back to team: {e}")
            # HOTFIX: Use unlimited processing for fallback
            return await self._execute_team_processing(input_prompt)

    def _build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Build context-aware input prompt with optimized string construction."""
        # Pre-calculate base components for efficiency
        base = f"Process Thought #{thought_data.thoughtNumber}:\n"
        content = f'\nThought Content: "{thought_data.thought}"'

        # Add context using pattern matching with optimized string building
        match thought_data:
            case ThoughtData(isRevision=True, branchFromThought=revision_num) if (
                revision_num
            ):
                original = self._session.find_thought_content(revision_num)
                context = f'**REVISION of Thought #{revision_num}** (Original: "{original}")\n'
                return f"{base}{context}{content}"

            case ThoughtData(branchFromThought=branch_from, branchId=branch_id) if (
                branch_from and branch_id
            ):
                origin = self._session.find_thought_content(branch_from)
                context = f'**BRANCH (ID: {branch_id}) from Thought #{branch_from}** (Origin: "{origin}")\n'
                return f"{base}{context}{content}"

            case _:
                return f"{base}{content}"

    def _format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Format response for MCP - clean content without guidance."""
        # MCP servers should return clean content, let AI decide next steps
        final_response = content

        # ENHANCED LOGGING: Response formatting details
        logger.info("📤 RESPONSE FORMATTING:")
        logger.info(f"  Original content length: {len(content)} chars")
        logger.info(f"  Next needed: {thought_data.nextThoughtNeeded}")
        logger.info(f"  Final response length: {len(final_response)} chars")
        logger.info(f"  Final response:\n{final_response}")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

        return final_response


@asynccontextmanager
async def create_server_lifespan() -> AsyncIterator[ServerState]:
    """Create server lifespan context manager with proper resource management."""
    config = ServerConfig.from_environment()
    server_state = ServerState()

    try:
        await server_state.initialize(config)
        logger.info("Server started successfully")
        yield server_state

    except Exception as e:
        logger.error(f"Server initialization failed: {e}", exc_info=True)
        raise ServerInitializationError(f"Failed to initialize server: {e}") from e

    finally:
        await server_state.cleanup()
        logger.info("Server shutdown complete")


class ServerInitializationError(Exception):
    """Custom exception for server initialization failures."""



def create_validated_thought_data(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool,
    branchFromThought: int | None,
    branchId: str | None,
    needsMoreThoughts: bool,
) -> ThoughtData:
    """Create and validate thought data with enhanced error reporting."""
    try:
        return ThoughtData(
            thought=thought.strip(),
            thoughtNumber=thoughtNumber,
            totalThoughts=totalThoughts,
            nextThoughtNeeded=nextThoughtNeeded,
            isRevision=isRevision,
            branchFromThought=branchFromThought,
            branchId=branchId.strip() if branchId else None,
            needsMoreThoughts=needsMoreThoughts,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid thought data: {e}") from e


# Global server state with workflow support
_server_state: ServerState | None = None
_thought_processor: ThoughtProcessor | None = None


async def get_thought_processor() -> ThoughtProcessor:
    """Get or create the global thought processor with workflow support."""
    global _thought_processor, _server_state

    if _thought_processor is None:
        if _server_state is None:
            raise RuntimeError("Server not properly initialized - _server_state is None. Ensure app_lifespan is running.")

        logger.info("Initializing ThoughtProcessor with Six Hats workflow")
        _thought_processor = ThoughtProcessor(_server_state.session)

    return _thought_processor
