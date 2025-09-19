"""Response processing utilities for consistent response handling."""

import logging
from typing import Any, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessedResponse:
    """Standardized response data structure."""

    content: str
    raw_response: Any
    processing_time: Optional[float] = None
    metadata: Optional[dict] = None


class ResponseExtractor:
    """Handles extraction of content from various response types."""

    @staticmethod
    def extract_content(response: Any) -> str:
        """Extract string content from various response formats."""
        if response is None:
            return ""

        # Handle string responses directly
        if isinstance(response, str):
            return response

        # Handle RunOutput from Agno framework
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, str):
                return content
            elif isinstance(content, dict):
                # Extract from dict-based content
                return ResponseExtractor._extract_from_dict(content)
            elif hasattr(content, "__str__"):
                return str(content)

        # Handle dictionary responses
        if isinstance(response, dict):
            return ResponseExtractor._extract_from_dict(response)

        # Handle objects with text/message attributes
        for attr in ["text", "message", "result", "output"]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, str):
                    return value

        # Fallback to string conversion
        logger.warning(f"Unknown response type {type(response)}, converting to string")
        return str(response)

    @staticmethod
    def _extract_from_dict(content_dict: dict) -> str:
        """Extract content from dictionary-based responses."""
        # Common content keys in order of preference
        content_keys = ["content", "text", "message", "result", "output", "response"]

        for key in content_keys:
            if key in content_dict:
                value = content_dict[key]
                if isinstance(value, str):
                    return value
                elif isinstance(value, dict) and "result" in value:
                    # Handle nested result structures
                    return str(value["result"])

        # If no standard key found, try to get first string value
        for value in content_dict.values():
            if isinstance(value, str) and value.strip():
                return value

        # Fallback to string representation
        return str(content_dict)


class ResponseProcessor:
    """Comprehensive response processing with logging and validation."""

    def __init__(self):
        """Initialize response processor."""
        self.extractor = ResponseExtractor()

    def process_response(
        self,
        response: Any,
        processing_time: Optional[float] = None,
        context: str = "processing",
    ) -> ProcessedResponse:
        """Process response with extraction, validation, and logging."""
        content = self.extractor.extract_content(response)

        # Validate content
        if not content or not content.strip():
            logger.warning(f"Empty response content in {context}")
            content = f"[Empty response from {context}]"

        # Create metadata
        metadata = {
            "context": context,
            "response_type": type(response).__name__,
            "content_length": len(content),
            "has_content": bool(content.strip()),
        }

        processed = ProcessedResponse(
            content=content,
            raw_response=response,
            processing_time=processing_time,
            metadata=metadata,
        )

        self._log_response_details(processed, context)
        return processed

    def _log_response_details(self, processed: ProcessedResponse, context: str) -> None:
        """Log detailed response information."""
        logger.info(f"📝 {context.upper()} RESPONSE:")
        logger.info(f"  Type: {processed.metadata['response_type']}")
        logger.info(f"  Length: {processed.metadata['content_length']} chars")

        if processed.processing_time:
            logger.info(f"  Processing time: {processed.processing_time:.3f}s")

        # Log content preview (first 100 chars)
        content_preview = processed.content[:100]
        if len(processed.content) > 100:
            content_preview += "..."

        logger.info(f"  Preview: {content_preview}")


class ResponseFormatter:
    """Formats responses for consistent output."""

    @staticmethod
    def format_for_client(processed: ProcessedResponse) -> str:
        """Format processed response for client consumption."""
        content = processed.content.strip()

        # Ensure content ends with appropriate punctuation
        if content and not content.endswith((".", "!", "?", ":", ";")):
            content += "."

        return content

    @staticmethod
    def format_with_metadata(processed: ProcessedResponse) -> dict:
        """Format response with metadata for debugging."""
        return {
            "content": processed.content,
            "metadata": processed.metadata,
            "processing_time": processed.processing_time,
        }
