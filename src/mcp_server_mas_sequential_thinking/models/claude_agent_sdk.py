"""Claude Agent SDK Model Wrapper for Agno Framework.

This module provides integration between Claude Agent SDK and Agno framework,
allowing the use of local Claude Code as a model provider within Multi-Thinking.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse

logger = logging.getLogger(__name__)


class ClaudeAgentSDKModel(Model):
    """Claude Agent SDK model wrapper that implements Agno Model interface.

    This class bridges Claude Agent SDK (local Claude Code) with the Agno framework,
    enabling it to be used as a model provider in the Multi-Thinking architecture.

    The wrapper:
    - Converts Agno messages to Claude Agent SDK query format
    - Executes queries through local Claude Code
    - Returns responses in Agno ModelResponse format
    - Supports reasoning tools (Think tool in Claude Agent SDK)
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5",  # Default model ID
        name: str | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Claude Agent SDK model.

        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4-5")
            name: Optional human-readable name
            **kwargs: Additional arguments passed to base Model class
        """
        super().__init__(
            id=model_id,
            name=name or "Claude Agent SDK",
            provider="claude-agent-sdk",
            **kwargs,
        )

        # Lazy import to avoid issues if SDK not installed
        try:
            from claude_agent_sdk import query as claude_query  # noqa: PLC0415

            self._claude_query = claude_query
        except ImportError as e:
            logger.exception(
                "claude-agent-sdk not installed. Please install it: "
                "pip install claude-agent-sdk"
            )
            raise ImportError(
                "claude-agent-sdk is required for ClaudeAgentSDKModel. "
                "Install it with: pip install claude-agent-sdk"
            ) from e

        logger.info(
            "Initialized Claude Agent SDK model: %s (provider: claude-agent-sdk)",
            model_id,
        )

    def _convert_messages_to_prompt(self, messages: list[Message]) -> str:
        """Convert Agno messages to a single prompt string for Claude Agent SDK.

        Args:
            messages: List of Agno Message objects

        Returns:
            Combined prompt string
        """
        prompt_parts = []

        for msg in messages:
            role = msg.role if hasattr(msg, "role") else "user"
            content = msg.content if hasattr(msg, "content") else str(msg)

            # Format based on role
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(str(content))

        return "\n\n".join(prompt_parts)

    async def aresponse(
        self,
        messages: list[Message],
        response_format: dict[str, Any] | type | None = None,  # noqa: ARG002
        tools: list[Any] | None = None,  # noqa: ARG002
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        tool_call_limit: int | None = None,  # noqa: ARG002
        run_response: Any = None,  # noqa: ARG002, ANN401
        send_media_to_model: bool = True,  # noqa: ARG002
    ) -> ModelResponse:
        """Generate async response using Claude Agent SDK.

        This is the core method that Agno Agent uses to get model responses.

        Args:
            messages: List of conversation messages
            response_format: Optional response format specification
            tools: Optional list of tools (ReasoningTools mapped to Think tool)
            tool_choice: Optional tool selection strategy
            tool_call_limit: Optional limit on tool calls
            run_response: Optional run response object
            send_media_to_model: Whether to send media content

        Returns:
            ModelResponse with generated content
        """
        try:
            # Convert Agno messages to Claude Agent SDK prompt
            prompt = self._convert_messages_to_prompt(messages)

            logger.debug(
                "Claude Agent SDK query - prompt length: %d chars", len(prompt)
            )

            # Collect response from Claude Agent SDK
            full_response = ""
            async for message in self._claude_query(prompt=prompt):
                # Claude Agent SDK returns message objects
                # Extract text content
                if hasattr(message, "content"):
                    # Handle different content formats
                    if isinstance(message.content, str):
                        full_response += message.content
                    elif isinstance(message.content, list):
                        # Handle content blocks
                        for block in message.content:
                            if hasattr(block, "text"):
                                full_response += block.text
                            elif isinstance(block, dict) and "text" in block:
                                full_response += block["text"]
                else:
                    # Fallback: convert to string
                    full_response += str(message)

            logger.debug(
                "Claude Agent SDK response - length: %d chars", len(full_response)
            )

            # Create Agno ModelResponse
            return ModelResponse(
                role="assistant",
                content=full_response,
                provider_data={
                    "model_id": self.id,
                    "provider": "claude-agent-sdk",
                },
            )

        except Exception as e:
            logger.exception("Claude Agent SDK query failed")
            # Return error response
            return ModelResponse(
                role="assistant",
                content=f"Error querying Claude Agent SDK: {e!s}",
                provider_data={
                    "error": str(e),
                    "model_id": self.id,
                    "provider": "claude-agent-sdk",
                },
            )

    async def aresponse_stream(
        self,
        messages: list[Message],
        response_format: dict[str, Any] | type | None = None,  # noqa: ARG002
        tools: list[Any] | None = None,  # noqa: ARG002
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        tool_call_limit: int | None = None,  # noqa: ARG002
        stream_model_response: bool = True,  # noqa: ARG002
        run_response: Any = None,  # noqa: ARG002, ANN401
        send_media_to_model: bool = True,  # noqa: ARG002
    ) -> AsyncIterator[ModelResponse]:
        """Generate streaming async response using Claude Agent SDK.

        Args:
            messages: List of conversation messages
            response_format: Optional response format specification
            tools: Optional list of tools
            tool_choice: Optional tool selection strategy
            tool_call_limit: Optional limit on tool calls
            stream_model_response: Whether to stream responses
            run_response: Optional run response object
            send_media_to_model: Whether to send media content

        Yields:
            ModelResponse objects as they arrive
        """
        try:
            prompt = self._convert_messages_to_prompt(messages)

            logger.debug(
                "Claude Agent SDK streaming query - prompt length: %d chars",
                len(prompt),
            )

            async for message in self._claude_query(prompt=prompt):
                # Extract content from message
                content = ""
                if hasattr(message, "content"):
                    if isinstance(message.content, str):
                        content = message.content
                    elif isinstance(message.content, list):
                        for block in message.content:
                            if hasattr(block, "text"):
                                content += block.text
                            elif isinstance(block, dict) and "text" in block:
                                content += block["text"]
                else:
                    content = str(message)

                if content:  # Only yield if there's content
                    yield ModelResponse(
                        role="assistant",
                        content=content,
                        provider_data={
                            "model_id": self.id,
                            "provider": "claude-agent-sdk",
                            "streaming": True,
                        },
                    )

        except Exception as e:
            logger.exception("Claude Agent SDK streaming query failed")
            yield ModelResponse(
                role="assistant",
                content=f"Error in Claude Agent SDK streaming: {e!s}",
                provider_data={
                    "error": str(e),
                    "model_id": self.id,
                    "provider": "claude-agent-sdk",
                },
            )

    def response(
        self,
        messages: list[Message],
        response_format: dict[str, Any] | type | None = None,
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        tool_call_limit: int | None = None,
        run_response: Any = None,  # noqa: ANN401
        send_media_to_model: bool = True,
    ) -> ModelResponse:
        """Synchronous response (not implemented for async-only SDK).

        Claude Agent SDK is async-only, so this method raises NotImplementedError.
        Use aresponse() instead.
        """
        raise NotImplementedError(
            "Claude Agent SDK only supports async operations. Use aresponse() instead."
        )

    def get_provider(self) -> str:
        """Get provider name.

        Returns:
            Provider name string
        """
        return "claude-agent-sdk"
