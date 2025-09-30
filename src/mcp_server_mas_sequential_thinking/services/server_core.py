"""Refactored server core with separated concerns and reduced complexity."""

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Lazy import to break circular dependency
from pydantic import ValidationError

from mcp_server_mas_sequential_thinking.config import (
    DefaultTimeouts,
    DefaultValues,
    PerformanceMetrics,
    check_required_api_keys,
)
from mcp_server_mas_sequential_thinking.core import (
    ConfigurationError,
    SessionMemory,
    ThoughtData,
)
from mcp_server_mas_sequential_thinking.utils import setup_logging

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
    def _log_metrics_block(title: str, metrics: dict[str, Any]) -> None:
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
    """Handles environment validation and setup with comprehensive early validation."""

    async def initialize(self, config: ServerConfig) -> None:
        """Validate environment requirements with enhanced error handling and early failure detection."""
        logger.info(f"Initializing environment with {config.provider} provider")

        try:
            # Early comprehensive validation
            await self._validate_provider_configuration(config)
            await self._validate_system_requirements()
            await self._setup_directories()

        except Exception as e:
            if not isinstance(e, ConfigurationError):
                raise ConfigurationError(
                    f"Environment initialization failed: {e}"
                ) from e
            raise

    async def _validate_provider_configuration(self, config: ServerConfig) -> None:
        """Validate provider-specific configuration with actionable error messages."""
        try:
            # Import here to avoid circular dependency
            from mcp_server_mas_sequential_thinking.config.modernized_config import (
                get_model_config,
                get_available_providers,
            )

            # Check if provider is supported
            available_providers = get_available_providers()
            if config.provider not in available_providers:
                raise ConfigurationError(
                    f"Unsupported provider '{config.provider}'. "
                    f"Available providers: {', '.join(available_providers)}. "
                    f"Set LLM_PROVIDER environment variable to one of the supported providers."
                )

            # Validate provider configuration
            try:
                model_config = get_model_config(config.provider)
                logger.info(f"✓ Provider '{config.provider}' configuration validated")
                logger.info(f"  Enhanced model: {model_config.enhanced_model_id}")
                logger.info(f"  Standard model: {model_config.standard_model_id}")

            except Exception as e:
                provider_specific_help = self._get_provider_setup_help(config.provider)
                raise ConfigurationError(
                    f"Failed to configure provider '{config.provider}': {e}\n\n"
                    f"Setup instructions:\n{provider_specific_help}"
                ) from e

        except ImportError as e:
            raise ConfigurationError(
                f"Failed to import configuration modules: {e}. "
                "This indicates a broken installation. Please reinstall the package."
            ) from e

    async def _validate_system_requirements(self) -> None:
        """Validate system-level requirements."""
        import sys

        # Python version check
        if sys.version_info < (3, 10):
            raise ConfigurationError(
                f"Python 3.10+ required, but found {sys.version_info.major}.{sys.version_info.minor}. "
                "Please upgrade Python or use a compatible environment."
            )

        # Check required packages are available
        required_packages = [
            ("agno", "2.0.5"),
            ("mcp", None),
            ("pydantic", None),
            ("asyncio", None),
        ]

        for package_name, min_version in required_packages:
            try:
                __import__(package_name)
                if min_version and package_name == "agno":
                    import agno
                    if hasattr(agno, "__version__"):
                        from packaging import version
                        if version.parse(agno.__version__) < version.parse(min_version):
                            raise ConfigurationError(
                                f"Agno {min_version}+ required, but found {agno.__version__}. "
                                "Run: uv pip install --upgrade agno"
                            )
            except ImportError as e:
                raise ConfigurationError(
                    f"Required package '{package_name}' not found: {e}. "
                    "Run: uv pip install -e '.[dev]' to install all dependencies."
                ) from e

        logger.info("✓ System requirements validated")

    async def _setup_directories(self) -> None:
        """Setup required directories with proper error handling."""
        # Ensure log directory exists
        log_dir = Path.home() / ".sequential_thinking" / "logs"
        if not log_dir.exists():
            logger.info(f"Creating log directory: {log_dir}")
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
                logger.info("✓ Log directory created successfully")
            except OSError as e:
                raise ConfigurationError(
                    f"Failed to create log directory {log_dir}: {e}. "
                    "Please check file system permissions."
                ) from e
        else:
            logger.info("✓ Log directory already exists")

        # Validate write permissions
        test_file = log_dir / "test_write.tmp"
        try:
            test_file.write_text("test")
            test_file.unlink()
            logger.info("✓ Log directory write permissions validated")
        except OSError as e:
            raise ConfigurationError(
                f"Cannot write to log directory {log_dir}: {e}. "
                "Please check file system permissions."
            ) from e

    def _get_provider_setup_help(self, provider: str) -> str:
        """Get provider-specific setup instructions."""
        help_text = {
            "deepseek": (
                "1. Get API key from https://platform.deepseek.com/api_keys\n"
                "2. Set environment variable: export DEEPSEEK_API_KEY='your-key'\n"
                "3. (Optional) Set models: DEEPSEEK_ENHANCED_MODEL_ID, DEEPSEEK_STANDARD_MODEL_ID"
            ),
            "groq": (
                "1. Get API key from https://console.groq.com/keys\n"
                "2. Set environment variable: export GROQ_API_KEY='your-key'\n"
                "3. (Optional) Set models: GROQ_ENHANCED_MODEL_ID, GROQ_STANDARD_MODEL_ID"
            ),
            "github": (
                "1. Get GitHub token from https://github.com/settings/tokens\n"
                "2. Set environment variable: export GITHUB_TOKEN='your-token'\n"
                "3. (Optional) Set models: GITHUB_ENHANCED_MODEL_ID, GITHUB_STANDARD_MODEL_ID\n"
                "4. Token must have appropriate permissions for GitHub Models API"
            ),
            "anthropic": (
                "1. Get API key from https://console.anthropic.com/\n"
                "2. Set environment variable: export ANTHROPIC_API_KEY='your-key'\n"
                "3. (Optional) Set models: ANTHROPIC_ENHANCED_MODEL_ID, ANTHROPIC_STANDARD_MODEL_ID"
            ),
            "openrouter": (
                "1. Get API key from https://openrouter.ai/keys\n"
                "2. Set environment variable: export OPENROUTER_API_KEY='your-key'\n"
                "3. (Optional) Set models: OPENROUTER_ENHANCED_MODEL_ID, OPENROUTER_STANDARD_MODEL_ID"
            ),
            "ollama": (
                "1. Install Ollama from https://ollama.ai/\n"
                "2. Start Ollama service: ollama serve\n"
                "3. Pull required models: ollama pull llama2\n"
                "4. (Optional) Set models: OLLAMA_ENHANCED_MODEL_ID, OLLAMA_STANDARD_MODEL_ID"
            ),
        }

        return help_text.get(provider, f"Please check documentation for '{provider}' provider setup.")

    async def cleanup(self) -> None:
        """No cleanup needed for environment."""
        logger.info("Environment cleanup completed")


class ServerState:
    """Manages server state with proper lifecycle and separation of concerns."""

    def __init__(self) -> None:
        self._config: ServerConfig | None = None
        self._session: SessionMemory | None = None
        self._initializers = [
            EnvironmentInitializer(),
        ]

    async def initialize(self, config: ServerConfig) -> None:
        """Initialize all server components."""
        self._config = config

        # Ordered initialization prevents dependency conflicts
        for initializer in self._initializers:
            await initializer.initialize(config)

        # Session-based architecture simplifies state management
        self._session = SessionMemory()

        logger.info(
            "Server state initialized successfully with multi-thinking workflow"
        )

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
    """Simplified thought processor that delegates to the refactored implementation.

    This maintains backward compatibility while using the new service-based architecture.
    """

    def __init__(self, session: SessionMemory) -> None:
        # Import and delegate to the refactored implementation
        from .thought_processor_refactored import (
            ThoughtProcessor as RefactoredProcessor,
        )

        self._processor = RefactoredProcessor(session)

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        return await self._processor.process_thought(thought_data)

    # Legacy methods for backward compatibility - delegate to refactored processor
    def _extract_response_content(self, response) -> str:
        """Legacy method - delegates to refactored processor."""
        return self._processor._extract_response_content(response)

    def _build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Legacy method - delegates to refactored processor."""
        return self._processor._build_context_prompt(thought_data)

    def _format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Legacy method - delegates to refactored processor."""
        return self._processor._format_response(content, thought_data)

    async def _execute_single_agent_processing(
        self, input_prompt: str, routing_decision=None
    ) -> str:
        """Legacy method - delegates to refactored processor."""
        return await self._processor._execute_single_agent_processing(
            input_prompt, routing_decision
        )

    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Legacy method - delegates to refactored processor."""
        return await self._processor._execute_team_processing(input_prompt)


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


# Legacy global state removed - use dependency injection through ApplicationContainer
