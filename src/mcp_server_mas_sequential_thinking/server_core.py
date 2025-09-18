"""Refactored server core with separated concerns and reduced complexity."""

import asyncio
import logging
import os
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from agno.team.team import Team
from agno.agent import Agent
from agno.tools.reasoning import ReasoningTools
from pydantic import ValidationError

from .modernized_config import check_required_api_keys, get_model_config
from .models import ThoughtData
from .session import SessionMemory
from .unified_team import create_team_by_type
from .utils import setup_logging
from .constants import (
    DefaultValues,
    DefaultTimeouts,
    ProcessingDefaults,
    FieldLengthLimits,
    PerformanceMetrics
)
from .adaptive_routing import AdaptiveRouter, ComplexityLevel, ProcessingStrategy
from .agno_workflow_router import AgnoWorkflowRouter, WorkflowResult
from .types import (
    ProcessingMetadata,
    ThoughtProcessingError,
    ConfigurationError,
    TeamCreationError,
    ConfigDict,
    CoordinationPlan,
    ExecutionMode,
)

logger = setup_logging()


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
        import os

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
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up server component."""
        pass


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
        pass


class TeamInitializer(ServerInitializer):
    """Handles team creation and configuration."""

    def __init__(self):
        self._team: Optional[Team] = None

    async def initialize(self, config: ServerConfig) -> None:
        """Initialize team based on configuration with enhanced error handling."""
        logger.info(f"Creating {config.team_mode} team")

        try:
            # Create team using unified factory
            model_config = get_model_config()
            self._team = create_team_by_type(config.team_mode, model_config)

            if not self._team:
                raise TeamCreationError(
                    f"Failed to create team of type '{config.team_mode}'"
                )

            logger.info(f"Team initialized successfully: {self._team.name}")

        except Exception as e:
            if not isinstance(e, TeamCreationError):
                raise TeamCreationError(
                    f"Team initialization failed for type '{config.team_mode}': {e}"
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
        self._config: Optional[ServerConfig] = None
        self._session: Optional[SessionMemory] = None
        self._initializers = [
            EnvironmentInitializer(),
            TeamInitializer(),
        ]
        self._team_initializer = self._initializers[ProcessingDefaults.TEAM_INITIALIZER_INDEX]

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


class ThoughtProcessor:
    """Handles thought processing with optimized performance and error handling."""

    __slots__ = ("_session", "_router", "_agno_router", "_use_workflow")  # Memory optimization

    def __init__(self, session: SessionMemory, use_agno_workflow: bool = False) -> None:
        self._session = session
        self._use_workflow = use_agno_workflow

        if use_agno_workflow:
            # AGNO WORKFLOW: Standard Agno Router + Workflow pattern
            logger.info("Initializing Agno-compliant Workflow Router (Workflow + Router pattern)")
            self._agno_router = AgnoWorkflowRouter()
            self._router = None  # Legacy router disabled
            logger.info("✅ Agno Workflow Router ready - standard routing activated")
        else:
            # LEGACY ADAPTIVE ROUTING: Custom routing system (deprecated)
            logger.info("Initializing Legacy Adaptive Router (complexity analysis + strategy selection)")
            self._router = AdaptiveRouter()
            self._agno_router = None
            logger.warning("⚠️  Using legacy AdaptiveRouter - consider migrating to Agno workflow")
            logger.info("✅ Legacy Adaptive Router ready - custom routing activated")


    def _extract_response_content(self, response) -> str:
        """Extract clean content from Agno RunOutput objects."""
        if hasattr(response, 'content') and response.content:
            return str(response.content)
        elif hasattr(response, 'messages') and response.messages:
            # Extract from last assistant message
            for msg in reversed(response.messages):
                if hasattr(msg, 'role') and msg.role == 'assistant' and hasattr(msg, 'content'):
                    return str(msg.content)
            return str(response)
        else:
            return str(response)

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        try:
            result = await self._process_thought_internal(thought_data)
            return result
        except Exception as e:
            error_msg = f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thought_number}: {e}"
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

        if self._use_workflow:
            # Use Agno-compliant workflow processing
            return await self._process_with_agno_workflow(thought_data, start_time)
        else:
            # Use legacy adaptive routing (deprecated)
            return await self._process_with_legacy_router(thought_data, start_time)

    async def _process_with_agno_workflow(self, thought_data: ThoughtData, start_time: float) -> str:
        """Process thought using Agno-compliant workflow."""
        input_prompt = self._build_context_prompt(thought_data)
        self._log_context_building(thought_data, input_prompt)

        # Execute Agno workflow
        workflow_result: WorkflowResult = await self._agno_router.process_thought_workflow(
            thought_data, input_prompt
        )

        final_response = self._format_response(workflow_result.content, thought_data)
        total_time = time.time() - start_time

        # Log workflow completion with Agno-specific metrics
        self._log_workflow_completion(thought_data, workflow_result, total_time, final_response)

        return final_response

    async def _process_with_legacy_router(self, thought_data: ThoughtData, start_time: float) -> str:
        """Process thought using legacy adaptive router (deprecated)."""
        coordination_plan, coordination_time = await self._create_coordination_plan(thought_data)
        input_prompt = self._build_context_prompt(thought_data)
        self._log_context_building(thought_data, input_prompt)

        response, processing_time = await self._execute_with_timing(input_prompt, coordination_plan)
        final_response = self._format_response(response, thought_data)

        total_time = time.time() - start_time
        self._log_completion_summary(thought_data, coordination_plan, processing_time, total_time, final_response)

        return final_response

    def _log_input_details(self, input_prompt: str, context_description: str = "input") -> None:
        """Log input details with consistent formatting."""
        logger.info(f"  Input length: {len(input_prompt)} chars")
        logger.info(f"  Full {context_description}:\\n{input_prompt}")
        logger.info(f"  {'='*PerformanceMetrics.SEPARATOR_LENGTH}")

    def _log_output_details(self, response_content: str, processing_time: float, context_description: str = "response") -> None:
        """Log output details with consistent formatting."""
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Output length: {len(response_content)} chars")
        logger.info(f"  Full {context_description}:\\n{response_content}")
        logger.info(f"  {'='*PerformanceMetrics.SEPARATOR_LENGTH}")

    def _create_simple_agent(self, processing_type: str = "thought", use_markdown: bool = False) -> Agent:
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
                "Be concise but helpful."
            ],
            markdown=use_markdown
        )

    async def _create_coordination_plan(self, thought_data: ThoughtData) -> tuple[CoordinationPlan, float]:
        """Create coordination plan from routing decision with timing."""
        coordination_start = time.time()
        routing_decision = self._router.route_thought(thought_data)
        coordination_plan = CoordinationPlan.from_routing_decision(routing_decision, thought_data)
        coordination_time = time.time() - coordination_start

        self._log_coordination_plan(coordination_plan, coordination_time)
        return coordination_plan, coordination_time

    def _log_coordination_plan(self, coordination_plan: CoordinationPlan, coordination_time: float) -> None:
        """Log detailed coordination plan analysis."""
        logger.info(f"🎯 COORDINATION PLAN:")
        logger.info(f"  Strategy: {coordination_plan.strategy}")
        logger.info(f"  Complexity: {coordination_plan.complexity_level} (score: {coordination_plan.complexity_score:.1f}/100)")
        logger.info(f"  Execution mode: {coordination_plan.execution_mode.value}")
        logger.info(f"  Specialists: {coordination_plan.specialist_roles}")
        logger.info(f"  Team size: {coordination_plan.team_size}")
        logger.info(f"  Coordination: {coordination_plan.coordination_strategy}")
        logger.info(f"  Processing mode: Unlimited time allowed")
        logger.info(f"  Coordination time: {coordination_time:.3f}s")
        logger.info(f"  Confidence: {coordination_plan.confidence:.2f}")
        logger.debug(f"  Reasoning: {coordination_plan.reasoning}")

    async def _execute_with_timing(self, input_prompt: str, coordination_plan: CoordinationPlan) -> tuple[str, float]:
        """Execute coordination plan with timing and logging."""
        processing_start = time.time()

        logger.info(f"📋 EXECUTING COORDINATION PLAN:")
        logger.info(f"  Mode: {coordination_plan.execution_mode.value}")
        logger.info(f"  Task breakdown: {coordination_plan.task_breakdown}")
        logger.info(f"  Expected interactions: {coordination_plan.expected_interactions}")

        response = await self._execute_coordination_plan(input_prompt, coordination_plan)
        processing_time = time.time() - processing_start

        return response, processing_time

    def _log_workflow_completion(self, thought_data: ThoughtData, workflow_result: WorkflowResult,
                                total_time: float, final_response: str) -> None:
        """Log workflow completion with Agno-specific metrics."""
        logger.info(f"🎯 AGNO WORKFLOW COMPLETION:")
        logger.info(f"  Thought #{thought_data.thought_number} processed successfully")
        logger.info(f"  Strategy: {workflow_result.strategy_used}")
        logger.info(f"  Complexity Score: {workflow_result.complexity_score:.1f}/100")
        logger.info(f"  Step: {workflow_result.step_name}")
        logger.info(f"  Processing time: {workflow_result.processing_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Response length: {len(final_response)} chars")
        logger.info(f"  {'=' * PerformanceMetrics.SEPARATOR_LENGTH}")

        # Performance metrics calculation
        execution_consistency = PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY if workflow_result.strategy_used != "error_fallback" else PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        efficiency_score = PerformanceMetrics.PERFECT_EFFICIENCY_SCORE if workflow_result.processing_time < PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD else max(PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE, PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / workflow_result.processing_time)

        logger.info(f"📊 WORKFLOW PERFORMANCE METRICS:")
        logger.info(f"  Execution Consistency: {execution_consistency:.2f}")
        logger.info(f"  Efficiency Score: {efficiency_score:.2f}")
        logger.info(f"  Response Length: {len(final_response)} chars")
        logger.info(f"  Strategy Executed: {workflow_result.strategy_used}")

    def _log_completion_summary(self, thought_data: ThoughtData, coordination_plan: CoordinationPlan,
                               processing_time: float, total_time: float, final_response: str) -> None:
        """Log performance metrics and completion summary."""
        # Log completion status
        logger.info(
            f"Thought #{thought_data.thought_number} completed: "
            f"strategy={coordination_plan.execution_mode.value}, "
            f"specialists={len(coordination_plan.specialist_roles)}, "
            f"processing_time={processing_time:.3f}s, "
            f"total_time={total_time:.3f}s, "
            f"confidence={coordination_plan.confidence:.2f}"
        )

        # Calculate and log performance metrics
        execution_consistency = PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY if coordination_plan.execution_mode.value else PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        efficiency_score = PerformanceMetrics.PERFECT_EFFICIENCY_SCORE if processing_time < PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD else max(PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE, PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / processing_time)

        logger.info(f"📊 PERFORMANCE METRICS:")
        logger.info(f"  Execution Consistency: {execution_consistency:.2f}")
        logger.info(f"  Efficiency Score: {efficiency_score:.2f}")
        logger.info(f"  Response Length: {len(final_response)} chars")
        logger.info(f"  Strategy Executed: {coordination_plan.execution_mode.value}")

        # Final processing summary
        logger.info(f"🎯 PROCESSING COMPLETE:")
        logger.info(f"  Thought #{thought_data.thought_number} processed successfully")
        logger.info(f"  Strategy used: {coordination_plan.execution_mode.value}")
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Response length: {len(final_response)} chars")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    def _log_thought_data(self, thought_data: ThoughtData) -> None:
        """Log comprehensive thought data information."""
        logger.info(f"🧩 THOUGHT DATA:")
        logger.info(f"  Thought #{thought_data.thought_number}/{thought_data.total_thoughts}")
        logger.info(f"  Type: {thought_data.thought_type.value}")
        logger.info(f"  Content: {thought_data.thought}")
        logger.info(f"  Next needed: {thought_data.next_needed}")
        logger.info(f"  Needs more: {thought_data.needs_more}")

        if thought_data.is_revision:
            logger.info(f"  Is revision: True (revises thought #{thought_data.revises_thought})")
        if thought_data.branch_from:
            logger.info(f"  Branch from: #{thought_data.branch_from} (ID: {thought_data.branch_id})")

        logger.info(f"  Raw data: {thought_data.format_for_log()}")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    async def _analyze_routing(self, thought_data: ThoughtData) -> tuple[object, float]:
        """Analyze routing decision and return decision with timing."""
        routing_start = time.time()
        routing_decision = self._router.route_thought(thought_data)
        routing_time = time.time() - routing_start

        self._log_routing_analysis(routing_decision, routing_time)
        return routing_decision, routing_time

    def _log_routing_analysis(self, routing_decision, routing_time: float) -> None:
        """Log detailed routing analysis information."""
        logger.info(f"🧠 ROUTING ANALYSIS:")
        logger.info(f"  Strategy: {routing_decision.strategy.value}")
        logger.info(f"  Complexity: {routing_decision.complexity_level.value} (score: {routing_decision.complexity_score:.1f}/100)")
        logger.info(f"  Estimated tokens: {routing_decision.estimated_token_usage[0]}-{routing_decision.estimated_token_usage[1]}")
        logger.info(f"  Estimated cost: ${routing_decision.estimated_cost:.6f}")
        logger.info(f"  Routing time: {routing_time:.3f}s")

        if routing_decision.specialist_recommendations:
            logger.info(f"  Recommended specialists: {', '.join(routing_decision.specialist_recommendations)}")
        logger.debug(f"  Reasoning: {routing_decision.reasoning}")

    def _log_context_building(self, thought_data: ThoughtData, input_prompt: str) -> None:
        """Log context building details."""
        logger.info(f"📝 CONTEXT BUILDING:")

        if thought_data.is_revision and thought_data.revises_thought:
            logger.info(f"  Type: Revision of thought #{thought_data.revises_thought}")
            try:
                original = self._session.find_thought_content(thought_data.revises_thought)
                logger.info(f"  Original thought: {original}")
            except:
                logger.info(f"  Original thought: [not found]")
        elif thought_data.branch_from and thought_data.branch_id:
            logger.info(f"  Type: Branch '{thought_data.branch_id}' from thought #{thought_data.branch_from}")
            try:
                origin = self._session.find_thought_content(thought_data.branch_from)
                logger.info(f"  Branch origin: {origin}")
            except:
                logger.info(f"  Branch origin: [not found]")
        else:
            logger.info(f"  Type: Sequential thought #{thought_data.thought_number}")

        logger.info(f"  Session thoughts: {len(self._session.thought_history)} total")
        logger.info(f"  Input thought: {thought_data.thought}")
        logger.info(f"  Built prompt length: {len(input_prompt)} chars")
        logger.info(f"  Built prompt:\n{input_prompt}")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    async def _execute_processing_strategy(self, input_prompt: str, routing_decision) -> tuple[str, str, float]:
        """Execute processing strategy and return response, strategy used, and processing time."""
        processing_start = time.time()

        if routing_decision.strategy == ProcessingStrategy.SINGLE_AGENT:
            response = await self._execute_single_agent_processing(input_prompt, routing_decision)
            strategy_used = "single_agent"
        else:
            response = await self._execute_team_processing_with_retries(
                input_prompt, routing_decision.complexity_level
            )
            strategy_used = "multi_agent"

        processing_time = time.time() - processing_start
        return response, strategy_used, processing_time

    def _log_performance_metrics(self, thought_data: ThoughtData, strategy_used: str, processing_time: float, total_time: float) -> None:
        """Log performance metrics for the processing."""
        logger.info(
            f"Thought #{thought_data.thought_number} completed: "
            f"strategy={strategy_used}, "
            f"processing_time={processing_time:.3f}s, "
            f"total_time={total_time:.3f}s"
        )

    def _log_processing_completion(self, thought_data: ThoughtData, strategy_used: str, processing_time: float, total_time: float, final_response: str) -> None:
        """Log final processing completion summary."""
        logger.info(f"🎯 PROCESSING COMPLETE:")
        logger.info(f"  Thought #{thought_data.thought_number} processed successfully")
        logger.info(f"  Strategy used: {strategy_used}")
        logger.info(f"  Processing time: {processing_time:.3f}s")
        logger.info(f"  Total time: {total_time:.3f}s")
        logger.info(f"  Response length: {len(final_response)} chars")
        logger.info(f"  {'=' * FieldLengthLimits.SEPARATOR_LENGTH}")

    async def _execute_coordination_plan(self, input_prompt: str, plan: CoordinationPlan) -> str:
        """Execute thought processing based on coordination plan (unified approach)."""

        logger.info(f"🎯 Executing {plan.execution_mode.value} with {plan.specialist_roles}")

        try:
            if plan.execution_mode == ExecutionMode.SINGLE_AGENT:
                return await self._execute_single_agent_simple(input_prompt)

            elif plan.execution_mode == ExecutionMode.SELECTIVE_TEAM:
                return await self._execute_selective_team(input_prompt, plan)

            elif plan.execution_mode == ExecutionMode.FULL_TEAM:
                return await self._execute_full_team_unlimited(input_prompt, plan)

            else:
                raise ValueError(f"Unknown execution mode: {plan.execution_mode}")

        except Exception as e:
            logger.error(f"Coordination plan execution failed: {e}")
            # Fallback to single agent for reliability
            logger.info("Falling back to single-agent processing")
            return await self._execute_single_agent_simple(input_prompt)

    async def _execute_single_agent_simple(self, input_prompt: str) -> str:
        """Execute simple single-agent processing."""
        simple_agent = self._create_simple_agent(processing_type="thought", use_markdown=False)

        logger.info(f"🤖 SINGLE-AGENT CALL:")
        logger.info(f"  Agent: {simple_agent.name} ({simple_agent.role})")
        logger.info(f"  Model: {getattr(simple_agent.model, 'id', 'unknown')} ({simple_agent.model.__class__.__name__})")
        self._log_input_details(input_prompt)

        start_time = time.time()
        response = await simple_agent.arun(input_prompt)
        processing_time = time.time() - start_time

        # HOTFIX: Properly extract content from Agno RunOutput
        response_content = self._extract_response_content(response)

        logger.info(f"✅ SINGLE-AGENT RESPONSE:")
        self._log_output_details(response_content, processing_time)

        return response_content

    async def _execute_selective_team(self, input_prompt: str, plan: CoordinationPlan) -> str:
        """Execute selective team processing (hybrid approach)."""
        # For now, delegate to full team but log as selective
        logger.info(f"🏢 SELECTIVE TEAM CALL:")
        logger.info(f"  Required specialists: {plan.specialist_roles}")
        logger.info(f"  Coordination strategy: {plan.coordination_strategy}")

        # TODO: Implement actual selective team creation
        # For now, use existing team without timeout
        return await self._execute_full_team_unlimited(input_prompt, plan)

    async def _execute_full_team_unlimited(self, input_prompt: str, plan: CoordinationPlan) -> str:
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
        """Execute team processing without timeout restrictions (for coordination plan)."""
        max_retries = DefaultTimeouts.MAX_RETRY_ATTEMPTS
        last_exception = None

        for retry_count in range(max_retries + 1):
            try:
                logger.info(
                    f"Processing attempt {retry_count + 1}/{max_retries + 1}: "
                    f"complexity={complexity_level.value}"
                )

                # ENHANCED LOGGING: Log multi-agent team call details
                team = self._session.team
                logger.info(f"🏢 MULTI-AGENT TEAM CALL:")
                logger.info(f"  Team: {team.name} ({len(team.members)} agents)")
                logger.info(f"  Leader: {team.model.__class__.__name__} (model: {getattr(team.model, 'id', 'unknown')})")
                logger.info(f"  Members: {', '.join([m.name for m in team.members])}")
                self._log_input_details(input_prompt)

                start_time = time.time()
                # REMOVED: All timeout restrictions for unlimited processing time
                response = await self._session.team.arun(input_prompt)
                processing_time = time.time() - start_time

                # HOTFIX: Properly extract content from Agno RunOutput
                response_content = self._extract_response_content(response)

                logger.info(f"✅ MULTI-AGENT RESPONSE:")
                self._log_output_details(response_content, processing_time)

                return response_content

            except Exception as e:
                last_exception = e
                logger.error(f"Processing error on attempt {retry_count + 1}: {e}")

                if retry_count < max_retries:
                    logger.info(f"Retrying... ({retry_count + 1}/{max_retries})")
                    await asyncio.sleep(PerformanceMetrics.RETRY_SLEEP_DURATION)  # Brief pause before retry
                else:
                    logger.error(f"All retry attempts exhausted")
                    raise ThoughtProcessingError(f"Team processing failed after {max_retries + 1} attempts: {e}") from e

        # This should never be reached, but just in case
        raise ThoughtProcessingError("Unexpected error in retry logic") from last_exception

    async def _execute_single_agent_processing(self, input_prompt: str, routing_decision) -> str:
        """Execute single-agent processing for simple thoughts without timeout restrictions."""
        try:
            # Create a lightweight agent for single processing
            simple_agent = self._create_simple_agent(processing_type="simple thought", use_markdown=True)

            # Create a simple response without multi-agent overhead
            simplified_prompt = f"""Process this thought efficiently:

{input_prompt}

Provide a focused response with clear guidance for the next step."""

            # ENHANCED LOGGING: Log single agent call details
            logger.info(f"🤖 SINGLE-AGENT CALL:")
            logger.info(f"  Agent: {simple_agent.name} ({simple_agent.role})")
            logger.info(f"  Model: {getattr(simple_agent.model, 'id', 'unknown')} ({simple_agent.model.__class__.__name__})")
            self._log_input_details(simplified_prompt)

            start_time = time.time()
            # REMOVED: All timeout restrictions for unlimited processing time
            response = await simple_agent.arun(simplified_prompt)
            processing_time = time.time() - start_time

            # HOTFIX: Properly extract content from Agno RunOutput
            response_content = self._extract_response_content(response)

            # ENHANCED LOGGING: Log single agent response details
            logger.info(f"✅ SINGLE-AGENT RESPONSE:")
            self._log_output_details(response_content, processing_time)

            logger.info(f"Single-agent processing completed (saved ~{routing_decision.estimated_cost:.4f}$ vs multi-agent)")
            return response_content

        except Exception as e:
            logger.warning(f"Single-agent processing failed, falling back to team: {e}")
            # HOTFIX: Use unlimited processing for fallback
            return await self._execute_team_processing(input_prompt)

    def _build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Build context-aware input prompt with optimized string construction."""
        # Pre-calculate base components for efficiency
        base = f"Process Thought #{thought_data.thought_number}:\n"
        content = f'\nThought Content: "{thought_data.thought}"'

        # Add context using pattern matching with optimized string building
        match thought_data:
            case ThoughtData(
                is_revision=True, revises_thought=revision_num
            ) if revision_num:
                original = self._session.find_thought_content(revision_num)
                context = f'**REVISION of Thought #{revision_num}** (Original: "{original}")\n'
                return f"{base}{context}{content}"

            case ThoughtData(branch_from=branch_from, branch_id=branch_id) if (
                branch_from and branch_id
            ):
                origin = self._session.find_thought_content(branch_from)
                context = f'**BRANCH (ID: {branch_id}) from Thought #{branch_from}** (Origin: "{origin}")\n'
                return f"{base}{context}{content}"

            case _:
                return f"{base}{content}"

    def _format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Format response with appropriate guidance."""
        guidance = (
            "\n\nGuidance: Look for revision/branch recommendations in the response. Formulate the next logical thought."
            if thought_data.next_needed
            else "\n\nThis is the final thought. Review the synthesis."
        )

        final_response = content + guidance

        # ENHANCED LOGGING: Response formatting details
        logger.info(f"📤 RESPONSE FORMATTING:")
        logger.info(f"  Original content length: {len(content)} chars")
        logger.info(f"  Next needed: {thought_data.next_needed}")
        logger.info(f"  Guidance added: {guidance.strip()}")
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

    pass


def create_validated_thought_data(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    next_needed: bool,
    is_revision: bool = False,
    revises_thought: Optional[int] = None,
    branch_from: Optional[int] = None,
    branch_id: Optional[str] = None,
    needs_more: bool = False,
) -> ThoughtData:
    """Create and validate thought data with enhanced error reporting."""
    try:
        return ThoughtData(
            thought=thought.strip(),
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_needed=next_needed,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from=branch_from,
            branch_id=branch_id.strip() if branch_id else None,
            needs_more=needs_more,
        )
    except ValidationError as e:
        raise ValueError(f"Invalid thought data: {e}") from e


# Global configuration
def get_workflow_enabled() -> bool:
    """Check if Agno workflow routing is enabled."""
    import os
    return os.getenv("USE_AGNO_WORKFLOW", "false").lower() in ("true", "1", "yes", "on")


# Global server state with workflow support
_server_state: Optional[ServerState] = None
_thought_processor: Optional[ThoughtProcessor] = None


async def get_thought_processor() -> ThoughtProcessor:
    """Get or create the global thought processor with workflow support."""
    global _thought_processor, _server_state

    if _thought_processor is None:
        if _server_state is None:
            config = ServerConfig.from_environment()
            _server_state = ServerState()
            await _server_state.initialize(config)

        # Check workflow configuration
        use_workflow = get_workflow_enabled()
        logger.info(f"Initializing ThoughtProcessor with workflow={use_workflow}")

        _thought_processor = ThoughtProcessor(_server_state.session, use_agno_workflow=use_workflow)

    return _thought_processor
