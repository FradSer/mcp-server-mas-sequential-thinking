"""Refactored MCP Sequential Thinking Server with separated concerns and reduced complexity."""

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .config import ProcessingDefaults, ValidationLimits
from .core import ThoughtProcessingError

# Import refactored modules
from .services import (
    ServerConfig,
    ServerState,
    ThoughtProcessor,
    create_server_lifespan,
    create_validated_thought_data,
)
from .utils import setup_logging

# Initialize environment and logging
load_dotenv()
logger = setup_logging()

# Global server state with thread safety
_server_state: ServerState | None = None
_thought_processor: ThoughtProcessor | None = None
_processor_lock = asyncio.Lock()


@asynccontextmanager
async def app_lifespan(app) -> AsyncIterator[None]:
    """Simplified application lifespan using refactored server core."""
    global _server_state, _thought_processor

    logger.info("Starting Sequential Thinking Server")

    async with create_server_lifespan() as server_state:
        _server_state = server_state
        logger.info("Server ready for requests")
        yield

    _server_state = None
    _thought_processor = None
    logger.info("Server stopped")


# Initialize FastMCP with lifespan
mcp = FastMCP(lifespan=app_lifespan)


def sanitize_and_validate_input(text: str, max_length: int, field_name: str) -> str:
    """Sanitize and validate input with comprehensive security checks."""
    # Early validation with guard clause
    if not text or not text.strip():
        raise ValueError(f"{field_name} cannot be empty")

    # Strip and normalize whitespace first
    text = text.strip()

    # Check for potential prompt injection patterns
    injection_patterns = [
        # System/role instruction injections
        "system:", "user:", "assistant:", "role:",
        # Prompt escape attempts
        "ignore previous", "ignore all", "disregard",
        # Code execution attempts
        "```python", "```bash", "exec(", "eval(", "__import__",
        # Instruction manipulation
        "new instructions", "override", "instead of",
        # Data extraction attempts
        "print(", "console.log", "alert(", "document.cookie",
    ]

    text_lower = text.lower()
    for pattern in injection_patterns:
        if pattern in text_lower:
            raise ValueError(
                f"Potential security risk detected in {field_name}. "
                f"Input contains suspicious pattern: '{pattern}'"
            )

    # Additional security checks
    if text.count('"') > 10 or text.count("'") > 10:
        raise ValueError(
            f"Excessive quotation marks detected in {field_name}. "
            "This may indicate an injection attempt."
        )

    # Sanitize HTML entities and special characters
    sanitized_text = escape(text)

    # Length validation with descriptive error
    if len(sanitized_text) > max_length:
        raise ValueError(
            f"{field_name} exceeds maximum length of {max_length} characters"
        )

    return sanitized_text


@mcp.prompt("sequential-thinking")
def sequential_thinking_prompt(problem: str, context: str = "") -> list[dict]:
    """Enhanced starter prompt for sequential thinking with better formatting."""
    # Sanitize and validate inputs
    try:
        problem = sanitize_and_validate_input(
            problem, ValidationLimits.MAX_PROBLEM_LENGTH, "Problem statement"
        )
        context = (
            sanitize_and_validate_input(
                context, ValidationLimits.MAX_CONTEXT_LENGTH, "Context"
            )
            if context
            else ""
        )
    except ValueError as e:
        raise ValueError(f"Input validation failed: {e}")

    user_prompt = f"""Initiate sequential thinking for: {problem}
{f"Context: {context}" if context else ""}"""

    assistant_guide = f"""Starting sequential thinking process for: {problem}

Process Guidelines:
1. Estimate appropriate number of total thoughts based on problem complexity
2. Begin with: "Plan comprehensive analysis for: {problem}"
3. Use revisions (isRevision=True) to improve previous thoughts
4. Use branching (branchFromThought, branchId) for alternative approaches
5. Each thought should be detailed with clear reasoning
6. Progress systematically through analysis phases

System Architecture:
- Multi-Thinking methodology with intelligent routing
- Factual, Emotional, Critical, Optimistic, Creative, and Synthesis perspectives
- Adaptive thinking sequence based on thought complexity and type
- Comprehensive integration through Synthesis thinking

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
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: bool,
    branchFromThought: int | None,
    branchId: str | None,
    needsMoreThoughts: bool,
) -> str:
    """Advanced sequential thinking tool with multi-agent coordination.

    Processes thoughts through a specialized team of AI agents that coordinate
    to provide comprehensive analysis, planning, research, critique, and synthesis.

    Args:
        thought: Content of the thinking step (required)
        thoughtNumber: Sequence number starting from {ThoughtProcessingLimits.MIN_THOUGHT_SEQUENCE} (≥{ThoughtProcessingLimits.MIN_THOUGHT_SEQUENCE})
        totalThoughts: Estimated total thoughts required (≥1)
        nextThoughtNeeded: Whether another thought step follows this one
        isRevision: Whether this thought revises a previous thought
        branchFromThought: Thought number to branch from for alternative exploration
        branchId: Unique identifier for the branch (required if branchFromThought set)
        needsMoreThoughts: Whether more thoughts are needed beyond the initial estimate

    Returns:
        Synthesized response from the multi-agent team with guidance for next steps

    Raises:
        ProcessingError: When thought processing fails
        ValidationError: When input validation fails
        RuntimeError: When server state is invalid
    """
    # Capture server state locally to avoid async race conditions
    current_server_state = _server_state
    if current_server_state is None:
        return "Server Error: Server not initialized"

    try:
        # Create and validate thought data using refactored function
        thought_data = create_validated_thought_data(
            thought=thought,
            thoughtNumber=thoughtNumber,
            totalThoughts=totalThoughts,
            nextThoughtNeeded=nextThoughtNeeded,
            isRevision=isRevision,
            branchFromThought=branchFromThought,
            branchId=branchId,
            needsMoreThoughts=needsMoreThoughts,
        )

        # Use captured state directly to avoid race conditions
        global _thought_processor
        async with _processor_lock:
            if _thought_processor is None:
                logger.info("Initializing ThoughtProcessor with Multi-Thinking workflow")
                _thought_processor = ThoughtProcessor(current_server_state.session)

        result = await _thought_processor.process_thought(thought_data)

        logger.info(f"Successfully processed thought #{thoughtNumber}")
        return result

    except ValidationError as e:
        error_msg = f"Input validation failed for thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        return f"Validation Error: {e}"

    except ThoughtProcessingError as e:
        error_msg = f"Processing failed for thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        if hasattr(e, "metadata") and e.metadata:
            logger.exception(f"Error metadata: {e.metadata}")
        return f"Processing Error: {e}"

    except Exception as e:
        error_msg = f"Unexpected error processing thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        return f"Unexpected Error: {e}"


def run() -> None:
    """Run the MCP server with enhanced error handling and graceful shutdown."""
    config = ServerConfig.from_environment()
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
        sys.exit(ProcessingDefaults.EXIT_CODE_ERROR)

    finally:
        logger.info("Server shutdown sequence complete")


def main() -> None:
    """Main entry point with proper error handling."""
    try:
        run()
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(ProcessingDefaults.EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()
