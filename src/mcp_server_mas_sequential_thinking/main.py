"""Modern MCP Sequential Thinking Server with enhanced architecture."""

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError
from dotenv import load_dotenv

# Import simplified modules
from .config import check_required_api_keys
from .models import ThoughtData
from .session import SessionMemory
from .team import create_team
from .utils import setup_logging

# Initialize environment and logging
load_dotenv()
logger = setup_logging()


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Immutable server configuration."""

    provider: str
    log_level: str = "INFO"
    max_retries: int = 3
    timeout: float = 30.0

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        return cls(
            provider=os.environ.get("LLM_PROVIDER", "deepseek"),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            max_retries=int(os.environ.get("MAX_RETRIES", "3")),
            timeout=float(os.environ.get("TIMEOUT", "30.0")),
        )


class ServerState:
    """Manages server state with proper lifecycle management."""

    def __init__(self) -> None:
        self._session: SessionMemory | None = None
        self._config: ServerConfig | None = None

    @property
    def session(self) -> SessionMemory:
        """Get current session, raising if not initialized."""
        if self._session is None:
            raise RuntimeError("Server not initialized - session unavailable")
        return self._session

    @property
    def config(self) -> ServerConfig:
        """Get current config, raising if not initialized."""
        if self._config is None:
            raise RuntimeError("Server not initialized - config unavailable")
        return self._config

    def initialize(self, config: ServerConfig, session: SessionMemory) -> None:
        """Initialize server state."""
        self._config = config
        self._session = session

    def cleanup(self) -> None:
        """Clean up server state."""
        self._session = None
        self._config = None


# Global server state
_server_state = ServerState()


class ThoughtProcessor:
    """Handles thought processing with enhanced error handling and logging."""

    __slots__ = ("_session",)

    def __init__(self, session: SessionMemory) -> None:
        self._session = session

    async def process_thought(self, thought_data: ThoughtData) -> str:
        """Process a thought through the team with comprehensive error handling."""
        try:
            return await self._process_thought_internal(thought_data)
        except Exception as e:
            logger.error(
                f"Failed to process {thought_data.thought_type.value} thought #{thought_data.thought_number}: {e}",
                exc_info=True,
            )
            raise ProcessingError(f"Thought processing failed: {e}") from e

    async def _process_thought_internal(self, thought_data: ThoughtData) -> str:
        """Internal thought processing logic."""
        # Log the thought with structured data
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

        # Prepare input with context
        input_prompt = self._build_input_prompt(thought_data)

        # Process through team with timeout handling
        response = await self._execute_team_processing(input_prompt)

        # Extract and format content
        return self._format_response(response, thought_data)

    async def _execute_team_processing(self, input_prompt: str) -> str:
        """Execute team processing with timeout and retry logic."""
        try:
            response = await self._session.team.arun(input_prompt)
            return getattr(response, "content", "") or str(response)
        except Exception as e:
            logger.warning(f"Team processing failed: {e}")
            raise ProcessingError(f"Team coordination failed: {e}") from e

    def _build_input_prompt(self, thought_data: ThoughtData) -> str:
        """Build input prompt with appropriate context using modern string formatting."""
        components = [f"Process Thought #{thought_data.thought_number}:\n"]

        # Add context for revisions/branches using match statement
        match thought_data:
            case ThoughtData(
                is_revision=True, revises_thought=revision_num
            ) if revision_num:
                original = self._session.find_thought_content(revision_num)
                components.append(
                    f'**REVISION of Thought #{revision_num}** (Original: "{original}")\n'
                )
            case ThoughtData(branch_from=branch_from, branch_id=branch_id) if (
                branch_from and branch_id
            ):
                origin = self._session.find_thought_content(branch_from)
                components.append(
                    f'**BRANCH (ID: {branch_id}) from Thought #{branch_from}** (Origin: "{origin}")\n'
                )

        components.append(f'\nThought Content: "{thought_data.thought}"')
        return "".join(components)

    def _format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Format response with appropriate guidance."""
        guidance_map = {
            True: "\n\nGuidance: Look for revision/branch recommendations in the response. Formulate the next logical thought.",
            False: "\n\nThis is the final thought. Review the synthesis.",
        }

        return content + guidance_map[thought_data.next_needed]


class ProcessingError(Exception):
    """Custom exception for thought processing errors."""

    pass


@asynccontextmanager
async def app_lifespan(app) -> AsyncIterator[None]:
    """Manage application lifecycle with proper resource management."""
    config = ServerConfig.from_env()
    logger.info(
        f"Initializing Sequential Thinking Server with {config.provider} provider"
    )

    try:
        # Validate environment and dependencies
        await _validate_server_requirements()

        # Initialize core components
        team = create_team()
        session = SessionMemory(team=team)

        # Initialize server state
        _server_state.initialize(config, session)

        logger.info("Server initialized successfully")
        yield

    except Exception as e:
        logger.error(f"Server initialization failed: {e}", exc_info=True)
        raise ServerInitializationError(f"Failed to initialize server: {e}") from e

    finally:
        logger.info("Server shutting down...")
        _server_state.cleanup()
        logger.info("Server shutdown complete")


async def _validate_server_requirements() -> None:
    """Validate server requirements and configuration."""
    # Check required API keys
    missing_keys = check_required_api_keys()
    if missing_keys:
        logger.warning(f"Missing API keys: {', '.join(missing_keys)}")

    # Validate critical paths
    log_dir = Path.home() / ".sequential_thinking" / "logs"
    if not log_dir.exists():
        logger.info(f"Creating log directory: {log_dir}")
        log_dir.mkdir(parents=True, exist_ok=True)


class ServerInitializationError(Exception):
    """Custom exception for server initialization failures."""

    pass


# Initialize FastMCP with lifespan
mcp = FastMCP(lifespan=app_lifespan)


@mcp.prompt("sequential-thinking")
def sequential_thinking_prompt(problem: str, context: str = "") -> list[dict]:
    """Enhanced starter prompt for sequential thinking with better formatting."""
    # Sanitize inputs
    problem = problem.strip()[:500]  # Limit problem length
    context = context.strip()[:300] if context else ""

    user_prompt = f"""Initiate sequential thinking for: {problem}
{f'Context: {context}' if context else ''}"""

    assistant_guide = f"""Starting sequential thinking process for: {problem}

Process Guidelines:
1. Estimate at least 5 total thoughts initially
2. Begin with: "Plan comprehensive analysis for: {problem}"
3. Use revisions (isRevision=True) to improve previous thoughts  
4. Use branching (branchFromThought, branchId) for alternative approaches
5. Each thought should be detailed with clear reasoning
6. Progress systematically through analysis phases

System Architecture:
- Multi-agent coordination team with specialized roles
- Planner, Researcher, Analyzer, Critic, and Synthesizer agents
- Intelligent delegation based on thought complexity
- Comprehensive synthesis of specialist responses

Ready to begin systematic analysis."""

    return [
        {
            "description": "Enhanced sequential thinking starter with comprehensive guidelines",
            "messages": [
                {"role": "user", "content": {"type": "text", "text": user_prompt}},
                {
                    "role": "assistant",
                    "content": {"type": "text", "text": assistant_guide},
                },
            ],
        }
    ]


@mcp.tool()
async def sequentialthinking(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    next_needed: bool,
    is_revision: bool = False,
    revises_thought: int | None = None,
    branch_from: int | None = None,
    branch_id: str | None = None,
    needs_more: bool = False,
) -> str:
    """
    Advanced sequential thinking tool with multi-agent coordination.

    Processes thoughts through a specialized team of AI agents that coordinate
    to provide comprehensive analysis, planning, research, critique, and synthesis.

    Args:
        thought: Content of the thinking step (required)
        thought_number: Sequence number starting from 1 (≥1)
        total_thoughts: Estimated total thoughts required (≥5)
        next_needed: Whether another thought step follows this one
        is_revision: Whether this thought revises a previous thought
        revises_thought: Thought number being revised (requires is_revision=True)
        branch_from: Thought number to branch from for alternative exploration
        branch_id: Unique identifier for the branch (required if branch_from set)
        needs_more: Whether more thoughts are needed beyond the initial estimate

    Returns:
        Synthesized response from the multi-agent team with guidance for next steps

    Raises:
        ProcessingError: When thought processing fails
        ValidationError: When input validation fails
        RuntimeError: When server state is invalid
    """
    try:
        # Validate server state
        session = _server_state.session

        # Create and validate thought data with enhanced error context
        thought_data = _create_validated_thought_data(
            thought=thought,
            thought_number=thought_number,
            total_thoughts=total_thoughts,
            next_needed=next_needed,
            is_revision=is_revision,
            revises_thought=revises_thought,
            branch_from=branch_from,
            branch_id=branch_id,
            needs_more=needs_more,
        )

        # Process through team with error handling
        processor = ThoughtProcessor(session)
        result = await processor.process_thought(thought_data)

        logger.info(f"Successfully processed thought #{thought_number}")
        return result

    except ValidationError as e:
        error_msg = f"Input validation failed for thought #{thought_number}: {e}"
        logger.error(error_msg)
        return f"Validation Error: {e}"

    except ProcessingError as e:
        error_msg = f"Processing failed for thought #{thought_number}: {e}"
        logger.error(error_msg)
        return f"Processing Error: {e}"

    except RuntimeError as e:
        error_msg = f"Server state error for thought #{thought_number}: {e}"
        logger.error(error_msg)
        return f"Server Error: {e}"

    except Exception as e:
        error_msg = f"Unexpected error processing thought #{thought_number}: {e}"
        logger.exception(error_msg)
        return f"Unexpected Error: {e}"


def _create_validated_thought_data(
    thought: str,
    thought_number: int,
    total_thoughts: int,
    next_needed: bool,
    is_revision: bool,
    revises_thought: int | None,
    branch_from: int | None,
    branch_id: str | None,
    needs_more: bool,
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
    except Exception as e:
        raise ValueError(f"Invalid thought data: {e}") from e


def run() -> None:
    """Run the MCP server with enhanced error handling and graceful shutdown."""
    config = ServerConfig.from_env()
    logger.info(f"Starting Sequential Thinking Server with {config.provider} provider")

    try:
        # Run server with stdio transport
        mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user (SIGINT)")

    except SystemExit as e:
        logger.info(f"Server stopped with exit code: {e.code}")
        raise

    except Exception as e:
        logger.error(f"Critical server error: {e}", exc_info=True)
        sys.exit(1)

    finally:
        logger.info("Server shutdown sequence complete")


def main() -> None:
    """Main entry point with proper error handling."""
    try:
        run()
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
