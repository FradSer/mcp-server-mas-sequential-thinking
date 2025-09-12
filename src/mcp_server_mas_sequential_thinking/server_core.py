"""Refactored server core with separated concerns and reduced complexity."""

import logging
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

logger = setup_logging()


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration with clear defaults."""
    
    provider: str
    team_mode: str = "standard" 
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0
    
    @classmethod
    def from_environment(cls) -> "ServerConfig":
        """Create config from environment with sensible defaults."""
        import os
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "deepseek"),
            team_mode=os.environ.get("TEAM_MODE", "standard").lower(),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            timeout=float(os.environ.get("TIMEOUT", "30.0")),
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
        """Validate environment requirements."""
        logger.info(f"Initializing environment with {config.provider} provider")
        
        # Check required API keys
        missing_keys = check_required_api_keys()
        if missing_keys:
            logger.warning(f"Missing API keys: {', '.join(missing_keys)}")
        
        # Ensure log directory exists
        log_dir = Path.home() / ".sequential_thinking" / "logs"
        if not log_dir.exists():
            logger.info(f"Creating log directory: {log_dir}")
            log_dir.mkdir(parents=True, exist_ok=True)
    
    async def cleanup(self) -> None:
        """No cleanup needed for environment."""
        pass


class TeamInitializer(ServerInitializer):
    """Handles team creation and configuration."""
    
    def __init__(self):
        self._team: Optional[Team] = None
    
    async def initialize(self, config: ServerConfig) -> None:
        """Initialize team based on configuration."""
        logger.info(f"Creating {config.team_mode} team")
        
        # Create team using unified factory
        model_config = get_model_config()
        self._team = create_team_by_type(config.team_mode, model_config)
        
        logger.info(f"Team initialized successfully: {self._team.name}")
    
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


class ProcessingError(Exception):
    """Custom exception for thought processing errors."""
    pass


class ThoughtProcessor:
    """Handles thought processing with optimized performance and error handling."""
    
    __slots__ = ("_session",)  # Memory optimization
    
    def __init__(self, session: SessionMemory) -> None:
        self._session = session
    
    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        try:
            return await self._process_thought_internal(thought_data)
        except Exception as e:
            error_msg = f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thought_number}: {e}"
            logger.error(error_msg, exc_info=True)
            raise ProcessingError(error_msg) from e
    
    async def _process_thought_internal(self, thought_data: ThoughtData) -> str:
        """Internal thought processing logic with structured logging."""
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
        
        # Build context-aware input
        input_prompt = self._build_context_prompt(thought_data)
        
        # Process through team
        response = await self._execute_team_processing(input_prompt)
        
        # Format and return response
        return self._format_response(response, thought_data)
    
    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Execute team processing with error handling."""
        try:
            response = await self._session.team.arun(input_prompt)
            return getattr(response, "content", "") or str(response)
        except Exception as e:
            raise ProcessingError(f"Team coordination failed: {e}") from e
    
    def _build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Build context-aware input prompt with optimized string construction."""
        # Pre-calculate base components for efficiency
        base = f"Process Thought #{thought_data.thought_number}:\n"
        content = f'\nThought Content: "{thought_data.thought}"'
        
        # Add context using pattern matching with optimized string building
        match thought_data:
            case ThoughtData(is_revision=True, revises_thought=revision_num) if revision_num:
                original = self._session.find_thought_content(revision_num)
                context = f'**REVISION of Thought #{revision_num}** (Original: "{original}")\n'
                return f"{base}{context}{content}"
            
            case ThoughtData(branch_from=branch_from, branch_id=branch_id) if branch_from and branch_id:
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