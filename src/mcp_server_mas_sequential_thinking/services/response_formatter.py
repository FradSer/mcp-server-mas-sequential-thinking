"""Response formatting service for thought processing.

This service handles formatting and synthesizing responses from various processing
modes, extracting content from Agno RunOutput objects, and preparing final responses.
"""

from mcp_server_mas_sequential_thinking.core import ThoughtData
from mcp_server_mas_sequential_thinking.utils import setup_logging

logger = setup_logging()


class ResponseFormatter:
    """Service responsible for formatting and synthesizing responses."""

    def __init__(self) -> None:
        """Initialize the response formatter."""

    def format_response(self, content: str, thought_data: ThoughtData) -> str:
        """Format response for MCP - clean content without additional guidance.

        MCP servers should return clean content and let the AI decide next steps.

        Args:
            content: The raw response content to format
            thought_data: The thought data context

        Returns:
            Formatted response string
        """
        # MCP servers should return clean content, let AI decide next steps
        final_response = content

        # Log response formatting details
        self._log_response_formatting(content, thought_data, final_response)

        return final_response

    def extract_response_content(self, response) -> str:
        """Extract clean content from Agno RunOutput objects.

        Handles various response types and extracts text content properly.

        Args:
            response: The response object from Agno processing

        Returns:
            Extracted text content
        """
        # Import ResponseExtractor to handle the extraction
        from mcp_server_mas_sequential_thinking.services.response_processor import (
            ResponseExtractor,
        )

        return ResponseExtractor.extract_content(response)

    def _log_response_formatting(
        self, content: str, thought_data: ThoughtData, final_response: str
    ) -> None:
        """Log response formatting details for debugging and monitoring.

        Args:
            content: The original response content
            thought_data: The thought data context
            final_response: The final formatted response
        """
        logger.info("📤 RESPONSE FORMATTING:")
        logger.info(f"  Original content length: {len(content)} chars")
        logger.info(f"  Next needed: {thought_data.nextThoughtNeeded}")
        logger.info(f"  Final response length: {len(final_response)} chars")
        logger.info(f"  Final response:\n{final_response}")

        # Use field length limits constant if available
        separator_length = 60  # Default fallback
        try:
            from mcp_server_mas_sequential_thinking.config import FieldLengthLimits

            separator_length = FieldLengthLimits.SEPARATOR_LENGTH
        except ImportError:
            pass

        logger.info(f"  {'=' * separator_length}")

    def log_input_details(
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

    def log_output_details(
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


class ResponseExtractor:
    """Utility class for extracting content from various response types."""

    @staticmethod
    def extract_content(response) -> str:
        """Extract clean content from Agno RunOutput objects.

        This method handles various response formats and extracts the actual
        text content that should be returned to the user.

        Args:
            response: The response object from agent processing

        Returns:
            Extracted text content as string
        """
        # Handle string responses directly
        if isinstance(response, str):
            return response.strip()

        # Handle Agno RunOutput objects
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                # Handle list of content items
                text_parts = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif hasattr(item, "text"):
                        text_parts.append(item.text)
                    elif hasattr(item, "content"):
                        text_parts.append(str(item.content))
                return "\n".join(text_parts).strip()

        # Handle objects with text attribute
        if hasattr(response, "text"):
            return response.text.strip()

        # Handle objects with message attribute
        if hasattr(response, "message"):
            message = response.message
            if isinstance(message, str):
                return message.strip()
            if hasattr(message, "content"):
                return str(message.content).strip()

        # Fallback: convert to string
        return str(response).strip()
