"""Context building service for thought processing.

This service handles building context-aware prompts from thought data,
managing session history, and constructing appropriate inputs for processing.
"""

from mcp_server_mas_sequential_thinking.core import SessionMemory, ThoughtData
from mcp_server_mas_sequential_thinking.utils import setup_logging

logger = setup_logging()


class ContextBuilder:
    """Service responsible for building context-aware prompts and managing thought context."""

    def __init__(self, session: SessionMemory) -> None:
        """Initialize the context builder with session memory.

        Args:
            session: The session memory instance for accessing thought history
        """
        self._session = session

    def build_context_prompt(self, thought_data: ThoughtData) -> str:
        """Build context-aware input prompt with optimized string construction.

        This method creates contextual prompts based on thought type:
        - Revision thoughts include original content
        - Branch thoughts include origin content
        - Sequential thoughts use basic format

        Args:
            thought_data: The thought data to build context for

        Returns:
            Formatted prompt string with appropriate context
        """
        # Pre-calculate base components for efficiency
        base = f"Process Thought #{thought_data.thoughtNumber}:\n"
        content = f'\nThought Content: "{thought_data.thought}"'

        # Add context using pattern matching with optimized string building
        match thought_data:
            case ThoughtData(isRevision=True, branchFromThought=revision_num) if (
                revision_num
            ):
                original = self._find_thought_content_safe(revision_num)
                context = f'**REVISION of Thought #{revision_num}** (Original: "{original}")\n'
                return f"{base}{context}{content}"

            case ThoughtData(branchFromThought=branch_from, branchId=branch_id) if (
                branch_from and branch_id
            ):
                origin = self._find_thought_content_safe(branch_from)
                context = f'**BRANCH (ID: {branch_id}) from Thought #{branch_from}** (Origin: "{origin}")\n'
                return f"{base}{context}{content}"

            case _:
                return f"{base}{content}"

    def _find_thought_content_safe(self, thought_number: int) -> str:
        """Safely find thought content with error handling.

        Args:
            thought_number: The thought number to find

        Returns:
            The thought content or a placeholder if not found
        """
        try:
            return self._session.find_thought_content(thought_number)
        except Exception:
            return "[not found]"

    def log_context_building(
        self, thought_data: ThoughtData, input_prompt: str
    ) -> None:
        """Log context building details for debugging and monitoring.

        Args:
            thought_data: The thought data being processed
            input_prompt: The built prompt
        """
        logger.info("📝 CONTEXT BUILDING:")

        if thought_data.isRevision and thought_data.branchFromThought:
            logger.info(
                f"  Type: Revision of thought #{thought_data.branchFromThought}"
            )
            original = self._find_thought_content_safe(thought_data.branchFromThought)
            logger.info(f"  Original thought: {original}")
        elif thought_data.branchFromThought and thought_data.branchId:
            logger.info(
                f"  Type: Branch '{thought_data.branchId}' from thought #{thought_data.branchFromThought}"
            )
            origin = self._find_thought_content_safe(thought_data.branchFromThought)
            logger.info(f"  Branch origin: {origin}")
        else:
            logger.info(f"  Type: Sequential thought #{thought_data.thoughtNumber}")

        logger.info(f"  Session thoughts: {len(self._session.thought_history)} total")
        logger.info(f"  Input thought: {thought_data.thought}")
        logger.info(f"  Built prompt length: {len(input_prompt)} chars")
        logger.info(f"  Built prompt:\n{input_prompt}")

        # Use field length limits constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import FieldLengthLimits

            separator_length = FieldLengthLimits.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * separator_length}")

    def create_simplified_prompt(self, input_prompt: str) -> str:
        """Create a simplified prompt for single-agent processing.

        Args:
            input_prompt: The original input prompt

        Returns:
            Simplified prompt optimized for single-agent processing
        """
        return f"""Process this thought efficiently:

{input_prompt}

Provide a focused response with clear guidance for the next step."""
