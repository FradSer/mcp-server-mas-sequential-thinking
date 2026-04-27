"""Refactored MCP Sequential Thinking Server with separated concerns and reduced complexity."""

import asyncio
import math
import sys
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from html import escape
from typing import Annotated

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field, ValidationError

from .config import ProcessingDefaults, SecurityConstants, ValidationLimits
from .core import ThoughtProcessingError
from .security import RateLimiter, RateLimitExceeded

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


# Application state container for dependency injection
class ApplicationContainer:
    """Dependency injection container for application state."""

    def __init__(self) -> None:
        self.server_state: ServerState | None = None
        self.thought_processor: ThoughtProcessor | None = None
        self.rate_limiter: RateLimiter = RateLimiter()
        self._processor_lock = asyncio.Lock()

    async def get_thought_processor(self) -> ThoughtProcessor:
        """Get or create thought processor with thread safety."""
        if self.server_state is None:
            raise RuntimeError("Server not initialized")

        async with self._processor_lock:
            if self.thought_processor is None:
                logger.info(
                    "Initializing ThoughtProcessor with Multi-Thinking workflow"
                )
                self.thought_processor = ThoughtProcessor(self.server_state.session)
            return self.thought_processor

    def cleanup(self) -> None:
        """Clean up application state."""
        self.server_state = None
        self.thought_processor = None


# Application container instance
_app_container = ApplicationContainer()


@asynccontextmanager
async def app_lifespan(app) -> AsyncIterator[None]:
    """Application lifespan with dependency injection container."""
    logger.info("Starting Sequential Thinking Server")

    async with create_server_lifespan() as server_state:
        _app_container.server_state = server_state
        logger.info("Server ready for requests")
        yield

    _app_container.cleanup()
    logger.info("Server stopped")


# Initialize FastMCP with lifespan
mcp = FastMCP(lifespan=app_lifespan)


def sanitize_and_validate_input(text: str, max_length: int, field_name: str) -> str:
    """Enhanced sanitization and validation with comprehensive security checks."""
    import re

    # Early validation with guard clause
    if not text or not text.strip():
        raise ValueError(f"{field_name} cannot be empty")

    # Strip and normalize whitespace first
    text = text.strip()

    # Enhanced injection pattern detection using regex
    for pattern in SecurityConstants.INJECTION_PATTERNS:
        if re.search(pattern, text):
            raise ValueError(
                f"Security risk detected in {field_name}: input contains suspicious pattern. "
                f"Please review your input and avoid system commands, code snippets, "
                f"or instruction manipulation attempts."
            )

    # Enhanced quotation mark and bracket validation
    quote_count = text.count('"') + text.count("'")
    if quote_count > SecurityConstants.MAX_QUOTATION_MARKS:
        raise ValueError(
            f"Excessive quotation marks detected in {field_name} ({quote_count} > {SecurityConstants.MAX_QUOTATION_MARKS}). "
            "This may indicate an injection attempt."
        )

    # Control character detection
    control_chars = sum(1 for c in text if ord(c) < 32 or 127 <= ord(c) <= 159)
    if control_chars > SecurityConstants.MAX_CONTROL_CHARACTERS:
        raise ValueError(
            f"Invalid control characters detected in {field_name} ({control_chars} found). "
            "Please use standard text characters only."
        )

    # Shannon entropy check for anomalous input patterns
    if len(text) > 20:  # Only check entropy for longer texts
        entropy = _calculate_shannon_entropy(text)
        if entropy < SecurityConstants.MIN_ENTROPY_THRESHOLD:
            raise ValueError(
                f"Input pattern appears suspicious in {field_name} (low entropy: {entropy:.2f}). "
                "Please provide more natural text input."
            )

    # Multiple bracket/parentheses check
    bracket_count = (
        text.count("(")
        + text.count(")")
        + text.count("[")
        + text.count("]")
        + text.count("{")
        + text.count("}")
    )
    if bracket_count > 20:  # Reasonable threshold for normal text
        raise ValueError(
            f"Excessive brackets/parentheses detected in {field_name} ({bracket_count} found). "
            "This may indicate code injection attempts."
        )

    # Sanitize HTML entities and special characters
    sanitized_text = escape(text)

    # Length validation with descriptive error
    if len(sanitized_text) > max_length:
        raise ValueError(
            f"{field_name} exceeds maximum length of {max_length} characters "
            f"(current: {len(sanitized_text)}). Please shorten your input."
        )

    return sanitized_text


def _calculate_shannon_entropy(text: str) -> float:
    """Calculate Shannon entropy of input text to detect anomalous patterns."""
    if not text:
        return 0.0

    # Count character frequencies
    char_counts = Counter(text.lower())
    total_chars = len(text)

    # Calculate entropy
    entropy = 0.0
    for count in char_counts.values():
        probability = count / total_chars
        if probability > 0:
            entropy -= probability * math.log2(probability)

    # Normalize by maximum possible entropy for this length
    max_entropy = math.log2(min(len(char_counts), total_chars))
    return entropy / max_entropy if max_entropy > 0 else 0.0


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
3. Actively use revisions (isRevision=True) to reflect on and improve previous thoughts
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


class StopReason(str, Enum):
    """Reason code for continuing or stopping thought iteration."""

    NEXT_THOUGHT_REQUIRED = "next_thought_required"
    NEEDS_MORE_THOUGHTS = "needs_more_thoughts"
    THOUGHT_SEQUENCE_COMPLETE = "thought_sequence_complete"
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    RATE_LIMITED = "rate_limited"
    REQUEST_TOO_LARGE = "request_too_large"
    UNEXPECTED_ERROR = "unexpected_error"


class NextCallArguments(BaseModel):
    """Recommended arguments for the next tool call."""

    thoughtNumber: int = Field(
        ...,
        ge=ValidationLimits.MIN_THOUGHT_NUMBER,
        description="Recommended thoughtNumber for the next tool call.",
    )
    totalThoughts: int = Field(
        ...,
        ge=1,
        description="Recommended totalThoughts for the next tool call.",
    )
    nextThoughtNeeded: bool = Field(
        ...,
        description=(
            "Set to true when another step should follow the next call. "
            "Set to false on the final thought."
        ),
    )
    needsMoreThoughts: bool = Field(
        ...,
        description=(
            "Set to true only when you need to exceed totalThoughts and extend "
            "the sequence."
        ),
    )


class SequentialThinkingStructuredContent(BaseModel):
    """Structured control fields returned on every tool response."""

    should_continue: bool = Field(
        ...,
        description=(
            "If true, call sequentialthinking again. The tool is designed for "
            "multi-step reasoning loops."
        ),
    )
    next_thought_number: int | None = Field(
        None,
        ge=ValidationLimits.MIN_THOUGHT_NUMBER,
        description=(
            "Recommended thoughtNumber for the next call. Null means no next "
            "step is currently required."
        ),
    )
    stop_reason: StopReason = Field(
        ...,
        description="Machine-readable reason that explains why to continue or stop.",
    )
    current_thought_number: int = Field(
        ...,
        ge=ValidationLimits.MIN_THOUGHT_NUMBER,
        description="Echo of the current thoughtNumber after normalization.",
    )
    total_thoughts: int = Field(
        ...,
        ge=1,
        description="Echo of current totalThoughts after normalization.",
    )
    next_call_arguments: NextCallArguments | None = Field(
        None,
        description=(
            "Concrete argument recommendations for the next call when the "
            "current run completes successfully and should continue."
        ),
    )
    parameter_usage: dict[str, str] = Field(
        ...,
        description=(
            "Contract reminders for each core parameter to keep multi-step "
            "iterations consistent."
        ),
    )


SEQUENTIALTHINKING_TOOL_DESCRIPTION = (
    "Multi-step sequential reasoning contract.\n"
    "Always treat this tool as iterative: after each response, read "
    "structuredContent.should_continue and continue calling until it is false.\n"
    "Actively use reflection: when a step reveals a flaw, explicitly send a "
    "revision step with isRevision=true.\n"
    "Input contract:\n"
    "- thought: one concrete reasoning step in natural language.\n"
    "- thoughtNumber: 1-based step index; increment by one for each new step.\n"
    "- totalThoughts: target number of steps for the current plan.\n"
    "- nextThoughtNeeded: true while the sequence should continue; false on final step.\n"
    "- isRevision: true only when revising an earlier conclusion.\n"
    "- branchFromThought + branchId: set together to explore an alternative branch.\n"
    "- needsMoreThoughts: true only when extending beyond totalThoughts.\n"
    "Output contract:\n"
    "- structuredContent.should_continue: canonical continuation signal.\n"
    "- structuredContent.next_thought_number: next recommended thoughtNumber.\n"
    "- structuredContent.stop_reason: canonical reason code for orchestration."
)


def _build_parameter_usage_contract() -> dict[str, str]:
    """Build immutable guidance text for input parameter usage."""
    return {
        "thought": "Send one focused reasoning step; avoid bundling multiple steps.",
        "thoughtNumber": "Start from 1 and increase by 1 for each new thought.",
        "totalThoughts": "Initial step budget; increase only with needsMoreThoughts=true.",
        "nextThoughtNeeded": "Set true for intermediate steps and false for final step.",
        "isRevision": (
            "Actively use reflection: set true when correcting or refining an "
            "earlier thought."
        ),
        "branchFromThought": "Set to a prior thought number when creating an alternative path.",
        "branchId": "Required when branchFromThought is set; keep stable on that branch.",
        "needsMoreThoughts": (
            "Set true only when the current plan needs more steps than totalThoughts."
        ),
    }


def _normalize_sequence_values(
    thought_number: int, total_thoughts: int
) -> tuple[int, int]:
    """Normalize sequence counters so output schema remains valid."""
    normalized_thought_number = max(ValidationLimits.MIN_THOUGHT_NUMBER, thought_number)
    normalized_total_thoughts = max(1, total_thoughts, normalized_thought_number)
    return normalized_thought_number, normalized_total_thoughts


def _derive_progress_state(
    thought_number: int,
    total_thoughts: int,
    next_thought_needed: bool,
    needs_more_thoughts: bool,
) -> tuple[bool, int | None, StopReason]:
    """Derive loop-control signals for the MCP structured output."""
    inferred_remaining_steps = thought_number < total_thoughts
    should_continue = (
        needs_more_thoughts or next_thought_needed or inferred_remaining_steps
    )

    if needs_more_thoughts:
        return should_continue, thought_number + 1, StopReason.NEEDS_MORE_THOUGHTS
    if should_continue:
        return should_continue, thought_number + 1, StopReason.NEXT_THOUGHT_REQUIRED
    return False, None, StopReason.THOUGHT_SEQUENCE_COMPLETE


def _build_next_call_arguments(
    should_continue: bool,
    stop_reason: StopReason,
    next_thought_number: int | None,
    total_thoughts: int,
) -> NextCallArguments | None:
    """Build next call recommendations for successful multi-step progress."""
    non_retry_reasons = {
        StopReason.NEXT_THOUGHT_REQUIRED,
        StopReason.NEEDS_MORE_THOUGHTS,
    }
    if (
        not should_continue
        or next_thought_number is None
        or stop_reason not in non_retry_reasons
    ):
        return None

    normalized_next = max(ValidationLimits.MIN_THOUGHT_NUMBER, next_thought_number)
    normalized_total = max(1, total_thoughts, normalized_next)
    needs_more_thoughts = (
        stop_reason == StopReason.NEEDS_MORE_THOUGHTS
        or normalized_next > total_thoughts
    )

    return NextCallArguments(
        thoughtNumber=normalized_next,
        totalThoughts=normalized_total,
        nextThoughtNeeded=normalized_next < normalized_total or needs_more_thoughts,
        needsMoreThoughts=needs_more_thoughts,
    )


def _build_call_result(
    message: str,
    thought_number: int,
    total_thoughts: int,
    should_continue: bool,
    stop_reason: StopReason,
    next_thought_number: int | None,
    *,
    is_error: bool = False,
) -> CallToolResult:
    """Build a CallToolResult with both text and structuredContent."""
    normalized_thought_number, normalized_total_thoughts = _normalize_sequence_values(
        thought_number, total_thoughts
    )

    if next_thought_number is None:
        normalized_next_thought_number = None
    else:
        normalized_next_thought_number = max(
            ValidationLimits.MIN_THOUGHT_NUMBER, next_thought_number
        )

    structured_content = SequentialThinkingStructuredContent(
        should_continue=should_continue,
        next_thought_number=normalized_next_thought_number,
        stop_reason=stop_reason,
        current_thought_number=normalized_thought_number,
        total_thoughts=normalized_total_thoughts,
        next_call_arguments=_build_next_call_arguments(
            should_continue=should_continue,
            stop_reason=stop_reason,
            next_thought_number=normalized_next_thought_number,
            total_thoughts=normalized_total_thoughts,
        ),
        parameter_usage=_build_parameter_usage_contract(),
    )

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        structuredContent=structured_content.model_dump(mode="json"),
        isError=is_error,
    )


@mcp.tool(
    description=SEQUENTIALTHINKING_TOOL_DESCRIPTION,
    structured_output=True,
)
async def sequentialthinking(
    thought: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Current reasoning step text. Keep this to one concrete step."
            ),
        ),
    ],
    thoughtNumber: Annotated[
        int,
        Field(
            ge=ValidationLimits.MIN_THOUGHT_NUMBER,
            description=(
                "1-based sequence index for this thought. Increment for each step."
            ),
        ),
    ],
    totalThoughts: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Estimated total number of steps in the current reasoning plan."
            ),
        ),
    ],
    nextThoughtNeeded: Annotated[
        bool,
        Field(
            description=(
                "Set true when another thought follows this one. Set false on "
                "the final thought."
            ),
        ),
    ],
    isRevision: Annotated[
        bool,
        Field(
            description=("Set true only when this step revises an earlier conclusion."),
        ),
    ],
    branchFromThought: Annotated[
        int | None,
        Field(
            ge=ValidationLimits.MIN_THOUGHT_NUMBER,
            description=(
                "Original thought number for branching. Null for the main path."
            ),
        ),
    ],
    branchId: Annotated[
        str | None,
        Field(
            min_length=1,
            description=(
                "Branch identifier. Required when branchFromThought is not null."
            ),
        ),
    ],
    needsMoreThoughts: Annotated[
        bool,
        Field(
            description=("Set true only when you must continue beyond totalThoughts."),
        ),
    ],
) -> Annotated[CallToolResult, SequentialThinkingStructuredContent]:
    """Advanced multi-step sequential thinking tool.

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
        CallToolResult with:
        - content: human-readable synthesis text
        - structuredContent: continuation control contract for client loops

    Raises:
        ProcessingError: When thought processing fails
        ValidationError: When input validation fails
        RuntimeError: When server state is invalid
    """
    normalized_thought_number, normalized_total_thoughts = _normalize_sequence_values(
        thoughtNumber, totalThoughts
    )

    # Apply rate limiting and DoS protection
    try:
        await _app_container.rate_limiter.check_rate_limit(client_id="mcp_client")
        _app_container.rate_limiter.validate_request_size(thought, "thought")
    except RateLimitExceeded as e:
        error_msg = (
            f"⏱️ Rate Limit Exceeded (Thought #{thoughtNumber}): {e}\n\n"
            f"💡 Action Required:\n"
            f"• Wait {e.retry_after:.1f} seconds before retrying\n"
            f"• You've exceeded the {e.limit_type} limit\n"
            f"• Consider reducing request frequency\n"
            f"• Break complex tasks into smaller chunks"
        )
        logger.warning(
            f"Rate limit exceeded for thought #{thoughtNumber}: {e.limit_type}"
        )
        return _build_call_result(
            message=error_msg,
            thought_number=normalized_thought_number,
            total_thoughts=normalized_total_thoughts,
            should_continue=True,
            stop_reason=StopReason.RATE_LIMITED,
            next_thought_number=normalized_thought_number,
            is_error=True,
        )
    except ValueError as e:
        # Request size validation error
        error_msg = (
            f"❌ Request Size Error (Thought #{thoughtNumber}): {e}\n\n"
            f"💡 Action Required:\n"
            f"• Reduce the size of your thought\n"
            f"• Split into multiple sequential thoughts\n"
            f"• Remove unnecessary details or formatting"
        )
        logger.warning(
            f"Request size validation failed for thought #{thoughtNumber}: {e}"
        )
        return _build_call_result(
            message=error_msg,
            thought_number=normalized_thought_number,
            total_thoughts=normalized_total_thoughts,
            should_continue=True,
            stop_reason=StopReason.REQUEST_TOO_LARGE,
            next_thought_number=normalized_thought_number,
            is_error=True,
        )

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

        # Get thought processor through dependency injection
        thought_processor = await _app_container.get_thought_processor()
        result = await thought_processor.process_thought(thought_data)

        should_continue, next_thought_number, stop_reason = _derive_progress_state(
            thought_number=thought_data.thoughtNumber,
            total_thoughts=thought_data.totalThoughts,
            next_thought_needed=thought_data.nextThoughtNeeded,
            needs_more_thoughts=thought_data.needsMoreThoughts,
        )

        logger.info(f"Successfully processed thought #{thoughtNumber}")
        return _build_call_result(
            message=result,
            thought_number=thought_data.thoughtNumber,
            total_thoughts=thought_data.totalThoughts,
            should_continue=should_continue,
            stop_reason=stop_reason,
            next_thought_number=next_thought_number,
        )

    except (ValidationError, ValueError) as e:
        error_msg = f"Input validation failed for thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        return _build_call_result(
            message=_format_validation_error(e, thoughtNumber),
            thought_number=normalized_thought_number,
            total_thoughts=normalized_total_thoughts,
            should_continue=True,
            stop_reason=StopReason.VALIDATION_ERROR,
            next_thought_number=normalized_thought_number,
            is_error=True,
        )

    except ThoughtProcessingError as e:
        error_msg = f"Processing failed for thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        if hasattr(e, "metadata") and e.metadata:
            logger.exception(f"Error metadata: {e.metadata}")
        return _build_call_result(
            message=_format_processing_error(e, thoughtNumber),
            thought_number=normalized_thought_number,
            total_thoughts=normalized_total_thoughts,
            should_continue=True,
            stop_reason=StopReason.PROCESSING_ERROR,
            next_thought_number=normalized_thought_number,
            is_error=True,
        )

    except Exception as e:
        error_msg = f"Unexpected error processing thought #{thoughtNumber}: {e}"
        logger.exception(error_msg)
        return _build_call_result(
            message=_format_unexpected_error(e, thoughtNumber),
            thought_number=normalized_thought_number,
            total_thoughts=normalized_total_thoughts,
            should_continue=True,
            stop_reason=StopReason.UNEXPECTED_ERROR,
            next_thought_number=normalized_thought_number,
            is_error=True,
        )

    finally:
        # Always release concurrent request slot
        await _app_container.rate_limiter.release_concurrent_slot()


def _format_validation_error(error: Exception, thought_number: int) -> str:
    """Format validation error with actionable guidance."""
    error_str = str(error)

    # Common validation issues and their solutions
    if "empty" in error_str.lower():
        return (
            f"❌ Validation Error (Thought #{thought_number}): {error}\n\n"
            "💡 Action Required:\n"
            "• Ensure the 'thought' field contains meaningful content\n"
            "• Avoid empty or whitespace-only thoughts\n"
            "• Provide a clear, descriptive thought for processing"
        )

    if "length" in error_str.lower():
        return (
            f"❌ Validation Error (Thought #{thought_number}): {error}\n\n"
            "💡 Action Required:\n"
            "• Shorten your input to fit within the allowed limits\n"
            "• Break complex thoughts into multiple sequential steps\n"
            "• Focus on one main idea per thought"
        )

    if "suspicious" in error_str.lower() or "injection" in error_str.lower():
        return (
            f"❌ Security Validation Error (Thought #{thought_number}): {error}\n\n"
            "💡 Action Required:\n"
            "• Review your input for potential security issues\n"
            "• Avoid system commands, code snippets, or instruction manipulation\n"
            "• Use natural language for your thoughts\n"
            "• Remove excessive special characters or brackets"
        )

    if "number" in error_str.lower() or "sequence" in error_str.lower():
        return (
            f"❌ Validation Error (Thought #{thought_number}): {error}\n\n"
            "💡 Action Required:\n"
            "• Ensure thoughtNumber is a positive integer starting from 1\n"
            "• Verify totalThoughts is greater than or equal to thoughtNumber\n"
            "• Check that sequence numbers are consistent and incremental"
        )

    # Generic validation error
    return (
        f"❌ Validation Error (Thought #{thought_number}): {error}\n\n"
        "💡 Action Required:\n"
        "• Review the input parameters for correct format and values\n"
        "• Ensure all required fields are provided\n"
        "• Check that data types match expected formats (strings, numbers, booleans)"
    )


def _format_processing_error(error: ThoughtProcessingError, thought_number: int) -> str:
    """Format processing error with actionable guidance."""
    return f"⚠️ Processing Error (Thought #{thought_number}): {error}"


def _format_unexpected_error(error: Exception, thought_number: int) -> str:
    """Format unexpected error with actionable guidance."""
    return f"💥 Unexpected Error (Thought #{thought_number}): {type(error).__name__}: {error}"


def run() -> None:
    """Run the MCP server with enhanced error handling and graceful shutdown."""
    try:
        config = ServerConfig.from_environment()
        logger.info(
            f"Starting Sequential Thinking Server with {config.provider} provider"
        )
    except Exception:
        sys.exit(ProcessingDefaults.EXIT_CODE_ERROR)

    try:
        # Run server with stdio transport
        mcp.run(transport="stdio")

    except KeyboardInterrupt:
        logger.info("Server stopped by user (SIGINT)")

    except SystemExit as e:
        logger.info(f"Server stopped with exit code: {e.code}")
        raise

    except Exception as e:
        error_msg = _format_server_runtime_error(e)
        logger.exception(error_msg)
        sys.exit(ProcessingDefaults.EXIT_CODE_ERROR)

    finally:
        logger.info("Server shutdown sequence complete")


def _format_startup_error(error: Exception) -> str:
    """Format server startup error with actionable guidance."""
    error_type = type(error).__name__
    error_str = str(error)

    if "configuration" in error_str.lower() or "config" in error_str.lower():
        return (
            f"🔧 Configuration Error: {error}\n\n"
            "💡 Action Required:\n"
            "• Check your environment variables (LLM_PROVIDER, API keys)\n"
            "• Run: uv run python -c 'from mcp_server_mas_sequential_thinking.config import validate_configuration_comprehensive; print(validate_configuration_comprehensive())'\n"
            "• Ensure all required dependencies are installed: uv pip install -e '.[dev]'\n"
            "• Verify your Python version is 3.10 or higher"
        )

    if "import" in error_str.lower() or "module" in error_str.lower():
        return (
            f"📦 Import Error: {error}\n\n"
            "💡 Action Required:\n"
            "• Install dependencies: uv pip install -e '.[dev]'\n"
            "• Ensure you're in the correct directory\n"
            "• Verify Python version compatibility (3.10+)\n"
            "• Check that the installation completed successfully"
        )

    if "permission" in error_str.lower():
        return (
            f"🔐 Permission Error: {error}\n\n"
            "💡 Action Required:\n"
            "• Check file system permissions for log directory\n"
            "• Ensure you have write access to ~/.sequential_thinking/logs/\n"
            "• Run with appropriate user permissions\n"
            "• Consider running: mkdir -p ~/.sequential_thinking/logs && chmod 755 ~/.sequential_thinking"
        )

    return (
        f"💥 Startup Error ({error_type}): {error}\n\n"
        "💡 Action Required:\n"
        "• Check the error message above for specific details\n"
        "• Verify your installation: uv pip install -e '.[dev]'\n"
        "• Ensure all environment variables are set correctly\n"
        "• Try running the configuration validation tool\n\n"
        "🐛 For support, please report this issue at:\n"
        "   https://github.com/anthropics/claude-code/issues"
    )


def _format_server_runtime_error(error: Exception) -> str:
    """Format server runtime error with actionable guidance."""
    error_type = type(error).__name__
    error_str = str(error)

    if "stdio" in error_str.lower() or "transport" in error_str.lower():
        return (
            f"📡 Transport Error: {error}\n\n"
            "💡 Action Required:\n"
            "• This usually indicates a communication issue with the MCP client\n"
            "• Ensure the server is being called correctly through MCP\n"
            "• Check that stdin/stdout are properly configured\n"
            "• Verify the MCP client is compatible with this server version"
        )

    if "agno" in error_str.lower():
        return (
            f"🤖 Agno Framework Error: {error}\n\n"
            "💡 Action Required:\n"
            "• Upgrade Agno: uv pip install --upgrade agno\n"
            "• Ensure Agno version 2.0.5+ is installed\n"
            "• Check Agno compatibility with your AI provider\n"
            "• Verify model configurations are correct"
        )

    return (
        f"⚠️ Server Runtime Error ({error_type}): {error}\n\n"
        "💡 Action Required:\n"
        "• Check the logs for more detailed information\n"
        "• Verify all system dependencies are available\n"
        "• Try restarting the server\n"
        "• Report persistent issues with full error details\n\n"
        "🐛 For support, please report this issue at:\n"
        "   https://github.com/anthropics/claude-code/issues"
    )


def main() -> None:
    """Main entry point with comprehensive error handling."""
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(0)
    except SystemExit:
        # Re-raise SystemExit without additional handling
        raise
    except Exception as e:
        _format_startup_error(e)
        try:
            logger.critical(f"Fatal error in main: {e}", exc_info=True)
        except Exception:
            # If logging fails, still show the error
            pass
        sys.exit(ProcessingDefaults.EXIT_CODE_ERROR)


if __name__ == "__main__":
    main()
