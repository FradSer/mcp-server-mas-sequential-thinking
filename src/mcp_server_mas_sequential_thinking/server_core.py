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
from pydantic import ValidationError

from .modernized_config import check_required_api_keys, get_model_config
from .models import ThoughtData
from .session import SessionMemory
from .unified_team import create_team_by_type
from .utils import setup_logging
from .constants import DefaultValues, DefaultTimeouts
from .adaptive_routing import AdaptiveRouter, ComplexityLevel, ProcessingStrategy
from .circuit_breaker import get_circuit_breaker, CircuitBreakerConfig
from .types import (
    ProcessingMetadata,
    ThoughtProcessingError,
    ConfigurationError,
    TeamCreationError,
    ConfigDict,
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
        self._team_initializer = self._initializers[1]  # Access team initializer

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

    __slots__ = ("_session", "_router", "_circuit_breaker")  # Memory optimization

    def __init__(self, session: SessionMemory) -> None:
        self._session = session
        # HOTFIX: Add adaptive router for complexity-based processing
        self._router = AdaptiveRouter()
        # HOTFIX: Add circuit breaker for failure protection
        provider = os.environ.get("LLM_PROVIDER", "deepseek").lower()
        self._circuit_breaker = get_circuit_breaker(
            f"{provider}_processing",
            CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30)
        )

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        # HOTFIX: Check circuit breaker before processing
        if not self._circuit_breaker.can_proceed():
            error_msg = f"Circuit breaker is OPEN - skipping thought #{thought_data.thought_number} processing"
            logger.warning(error_msg)
            return f"Processing temporarily unavailable due to repeated failures. Please try again later."

        try:
            result = await self._process_thought_internal(thought_data)
            # Record success for circuit breaker
            self._circuit_breaker.record_success()
            return result
        except Exception as e:
            # Record failure for circuit breaker
            self._circuit_breaker.record_failure()
            error_msg = f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thought_number}: {e}"
            logger.error(error_msg, exc_info=True)
            metadata: ProcessingMetadata = {
                "error_count": 1,
                "retry_count": 0,
                "processing_time": 0.0,
            }
            raise ThoughtProcessingError(error_msg, metadata) from e

    async def _process_thought_internal(self, thought_data: ThoughtData) -> str:
        """Internal thought processing logic with structured logging."""
        # HOTFIX: Add performance monitoring
        start_time = time.time()

        # Log with structured data
        logger.info(
            "Processing thought",
            extra={
                "thought_type": thought_data.thought_type.value,
                "thought_number": thought_data.thought_number,
                "total_thoughts": thought_data.total_thoughts,
                "is_revision": thought_data.is_revision,
                "branch_id": thought_data.branch_id,
            },
        )
        logger.debug(thought_data.format_for_log())

        # Add to session
        self._session.add_thought(thought_data)

        # HOTFIX: Use adaptive routing to determine processing strategy
        routing_start = time.time()
        routing_decision = self._router.route_thought(thought_data)
        routing_time = time.time() - routing_start

        logger.info(
            f"Routing decision: {routing_decision.strategy.value} "
            f"(complexity: {routing_decision.complexity_score:.1f}, "
            f"routing_time: {routing_time:.3f}s)"
        )

        # Build context-aware input
        input_prompt = self._build_context_prompt(thought_data)

        # Process based on routing decision with timing
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
        total_time = time.time() - start_time

        # Log performance metrics
        logger.info(
            f"Thought #{thought_data.thought_number} completed: "
            f"strategy={strategy_used}, "
            f"processing_time={processing_time:.3f}s, "
            f"total_time={total_time:.3f}s"
        )

        # Format and return response
        return self._format_response(response, thought_data)

    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Execute team processing with error handling and timeout protection."""
        try:
            # HOTFIX: Add provider-specific timeout protection
            timeout = self._get_provider_timeout()
            response = await asyncio.wait_for(
                self._session.team.arun(input_prompt),
                timeout=timeout
            )
            return getattr(response, "content", "") or str(response)
        except asyncio.TimeoutError:
            raise ThoughtProcessingError(f"Team processing timed out after {timeout}s")
        except Exception as e:
            raise ThoughtProcessingError(f"Team coordination failed: {e}") from e

    async def _execute_team_processing_with_retries(
        self, input_prompt: str, complexity_level: ComplexityLevel
    ) -> str:
        """Execute team processing with adaptive timeout and intelligent retries."""
        max_retries = DefaultTimeouts.MAX_RETRY_ATTEMPTS  # Allow up to 3 total attempts
        last_exception = None

        for retry_count in range(max_retries + 1):
            try:
                # Calculate adaptive timeout for this attempt
                timeout = self._get_adaptive_timeout(complexity_level, retry_count)

                logger.info(
                    f"Processing attempt {retry_count + 1}/{max_retries + 1}: "
                    f"timeout={timeout:.1f}s, complexity={complexity_level.value}"
                )

                response = await asyncio.wait_for(
                    self._session.team.arun(input_prompt),
                    timeout=timeout
                )

                # Success! Log the successful attempt
                if retry_count > 0:
                    logger.info(f"Processing succeeded on retry {retry_count}")

                return getattr(response, "content", "") or str(response)

            except asyncio.TimeoutError as e:
                last_exception = e
                if retry_count < max_retries:
                    # Log timeout and prepare for retry
                    next_timeout = self._get_adaptive_timeout(complexity_level, retry_count + 1)
                    logger.warning(
                        f"Timeout on attempt {retry_count + 1} ({timeout:.1f}s). "
                        f"Retrying with longer timeout ({next_timeout:.1f}s)..."
                    )
                    continue
                else:
                    # Final timeout - calculate total time spent
                    total_time = sum(
                        self._get_adaptive_timeout(complexity_level, i)
                        for i in range(max_retries + 1)
                    )
                    provider = os.environ.get("LLM_PROVIDER", "unknown")

                    raise ThoughtProcessingError(
                        f"Processing failed after {max_retries + 1} attempts "
                        f"(total time: {total_time:.1f}s). "
                        f"This {complexity_level.value} thought exceeded the maximum "
                        f"processing time for {provider}. Consider breaking it into "
                        f"smaller, simpler thoughts or try again later."
                    )

            except Exception as e:
                # Non-timeout errors don't retry
                last_exception = e
                raise ThoughtProcessingError(f"Team coordination failed: {e}") from e

        # This should never be reached, but just in case
        raise ThoughtProcessingError(f"Unexpected error after retries") from last_exception

    async def _execute_single_agent_processing(self, input_prompt: str, routing_decision) -> str:
        """Execute single-agent processing for simple thoughts (hotfix optimization)."""
        try:
            # Use team leader as single agent for simple processing
            model_config = get_model_config()
            single_model = model_config.create_team_model()

            # Create a simple response without multi-agent overhead
            simplified_prompt = f"""Process this thought efficiently:

{input_prompt}

Provide a focused response with clear guidance for the next step."""

            # HOTFIX: Add adaptive timeout protection for single agent
            # Single agent gets simpler complexity level for faster timeout
            single_agent_complexity = ComplexityLevel.SIMPLE
            timeout = self._get_adaptive_timeout(single_agent_complexity, 0) * 0.5  # 50% of base

            response = await asyncio.wait_for(
                single_model.apredict(simplified_prompt),
                timeout=timeout
            )

            logger.info(f"Single-agent processing completed (saved ~{routing_decision.estimated_cost:.4f}$ vs multi-agent)")
            return getattr(response, "content", "") or str(response)

        except asyncio.TimeoutError:
            logger.warning(f"Single-agent processing timed out after {timeout}s, falling back to team")
            return await self._execute_team_processing(input_prompt)
        except Exception as e:
            logger.warning(f"Single-agent processing failed, falling back to team: {e}")
            # Fallback to team processing if single agent fails
            return await self._execute_team_processing(input_prompt)

    def _get_adaptive_timeout(self, complexity_level: ComplexityLevel, retry_count: int = 0) -> float:
        """Get adaptive timeout based on complexity and retry attempts."""
        provider = os.environ.get("LLM_PROVIDER", "").lower()

        # Base timeout by provider
        base_timeouts = {
            "deepseek": DefaultTimeouts.ADAPTIVE_BASE_DEEPSEEK,
            "groq": DefaultTimeouts.ADAPTIVE_BASE_GROQ,
            "openai": DefaultTimeouts.ADAPTIVE_BASE_OPENAI,
            "default": DefaultTimeouts.ADAPTIVE_BASE_DEFAULT
        }
        base_timeout = base_timeouts.get(provider, base_timeouts["default"])

        # Complexity multipliers
        complexity_multipliers = {
            ComplexityLevel.SIMPLE: DefaultTimeouts.COMPLEXITY_SIMPLE_MULTIPLIER,
            ComplexityLevel.MODERATE: DefaultTimeouts.COMPLEXITY_MODERATE_MULTIPLIER,
            ComplexityLevel.COMPLEX: DefaultTimeouts.COMPLEXITY_COMPLEX_MULTIPLIER,
            ComplexityLevel.HIGHLY_COMPLEX: DefaultTimeouts.COMPLEXITY_HIGHLY_COMPLEX_MULTIPLIER
        }
        complexity_multiplier = complexity_multipliers.get(complexity_level, DefaultTimeouts.COMPLEXITY_COMPLEX_MULTIPLIER)

        # Exponential backoff for retries
        retry_multiplier = DefaultTimeouts.RETRY_EXPONENTIAL_BASE ** retry_count

        # Calculate adaptive timeout
        adaptive_timeout = base_timeout * complexity_multiplier * retry_multiplier

        # Safety ceiling - never exceed maximum
        max_timeouts = {
            "deepseek": DefaultTimeouts.MAX_TIMEOUT_DEEPSEEK,
            "default": DefaultTimeouts.MAX_TIMEOUT_DEFAULT
        }
        max_timeout = max_timeouts.get(provider, max_timeouts["default"])

        return min(adaptive_timeout, max_timeout)

    def _get_provider_timeout(self) -> float:
        """Get provider-specific timeout (legacy method for backward compatibility)."""
        provider = os.environ.get("LLM_PROVIDER", "").lower()

        if provider == "deepseek":
            return DefaultTimeouts.DEEPSEEK_PROCESSING_TIMEOUT
        else:
            return DefaultTimeouts.PROCESSING_TIMEOUT

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
        return content + guidance


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
