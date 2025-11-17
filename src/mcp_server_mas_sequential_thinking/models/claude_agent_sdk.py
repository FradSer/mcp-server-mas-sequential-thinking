"""Claude Agent SDK Model Wrapper for Agno Framework.

This module provides integration between Claude Agent SDK and Agno framework,
allowing the use of local Claude Code as a model provider within Multi-Thinking.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from agno.models.base import Model
from agno.models.message import Message
from agno.models.response import ModelResponse

logger = logging.getLogger(__name__)


class ClaudeAgentSDKModel(Model):
    """Claude Agent SDK model wrapper that implements Agno Model interface.

    This class bridges Claude Agent SDK (local Claude Code) with the Agno framework,
    enabling it to be used as a model provider in the Multi-Thinking architecture.

    Features:
    - Converts Agno messages to Claude Agent SDK query format
    - Executes queries through local Claude Code
    - Returns responses in Agno ModelResponse format
    - Supports reasoning tools (Think tool in Claude Agent SDK)
    - Tool permission management (allowed_tools, can_use_tool callback)
    - MCP server integration
    - Environment variables and working directory control
    - Event hooks (PreToolUse, PostToolUse, UserPromptSubmit, etc.)
    - Additional directory access for context

    Example:
        Basic usage:
        >>> model = ClaudeAgentSDKModel(
        ...     model_id="claude-sonnet-4-5",
        ...     permission_mode="bypassPermissions"
        ... )

        With MCP servers:
        >>> model = ClaudeAgentSDKModel(
        ...     mcp_servers={"filesystem": {...}},
        ...     env={"DEBUG": "1"},
        ...     add_dirs=["/path/to/project"]
        ... )

        With hooks:
        >>> hooks = {
        ...     "PreToolUse": [lambda ctx: print(f"Using {ctx.tool_name}")],
        ...     "PostToolUse": [lambda ctx: print(f"Completed {ctx.tool_name}")]
        ... }
        >>> model = ClaudeAgentSDKModel(hooks=hooks)

        With permission callback:
        >>> async def check_permission(tool_name, args, context):
        ...     if tool_name == "dangerous_tool":
        ...         return {"allow": False, "reason": "Not allowed"}
        ...     return {"allow": True}
        >>> model = ClaudeAgentSDKModel(can_use_tool=check_permission)
    """

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-5",  # Default model ID
        name: str | None = None,
        permission_mode: Literal[
            "default", "acceptEdits", "plan", "bypassPermissions"
        ] = "bypassPermissions",
        cwd: str | None = None,
        mcp_servers: dict[str, Any] | str | Path | None = None,
        env: dict[str, str] | None = None,
        add_dirs: list[str | Path] | None = None,
        hooks: dict[str, list[Any]] | None = None,
        can_use_tool: Callable[[str, dict[str, Any], Any], Awaitable[Any]]
        | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize Claude Agent SDK model.

        Args:
            model_id: Model identifier (e.g., "claude-sonnet-4-5")
            name: Optional human-readable name
            permission_mode: Permission mode for Claude Code operations
                - 'default': Standard permissions with prompts
                - 'acceptEdits': Auto-accept file edits
                - 'plan': Plan mode for reviewing actions
                - 'bypassPermissions': Bypass all permission checks (default)
            cwd: Working directory for Claude Code (default: current directory)
            mcp_servers: MCP servers configuration (dict, path to config, or None)
            env: Environment variables to pass to Claude Code
            add_dirs: Additional directories for context/file access
            hooks: Event hooks (PreToolUse, PostToolUse, UserPromptSubmit, etc.)
            can_use_tool: Runtime callback for tool permission checks
            **kwargs: Additional arguments passed to base Model class
        """
        super().__init__(
            id=model_id,
            name=name or "Claude Agent SDK",
            provider="claude-agent-sdk",
            **kwargs,
        )

        # Store configuration
        self.permission_mode = permission_mode
        self.cwd = cwd or str(Path.cwd())
        self.mcp_servers = mcp_servers
        self.env = env or {}
        self.add_dirs = [str(d) for d in add_dirs] if add_dirs else []
        self.hooks = hooks
        self.can_use_tool = can_use_tool

        # Lazy import to avoid issues if SDK not installed
        try:
            # Import both classes from claude_agent_sdk
            from claude_agent_sdk import (  # noqa: PLC0415, I001
                ClaudeAgentOptions,
                query as claude_query,
            )

            self._claude_query = claude_query
            self._claude_options_class = ClaudeAgentOptions
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
            "Initialized Claude Agent SDK model: %s (provider: claude-agent-sdk, "
            "permission_mode: %s, cwd: %s, mcp_servers: %s, env_vars: %d, "
            "add_dirs: %d, hooks: %s, can_use_tool: %s)",
            model_id,
            permission_mode,
            self.cwd,
            "configured" if self.mcp_servers else "none",
            len(self.env),
            len(self.add_dirs),
            list(self.hooks.keys()) if self.hooks else "none",
            "configured" if self.can_use_tool else "none",
        )

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, str]:
        """Extract system prompt and convert messages to prompt string.

        System messages are separated and used for ClaudeAgentOptions.system_prompt.
        Other messages are converted to the main prompt.

        Args:
            messages: List of Agno Message objects

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_parts = []
        prompt_parts = []

        for msg in messages:
            role = msg.role if hasattr(msg, "role") else "user"
            content = msg.content if hasattr(msg, "content") else str(msg)

            # Separate system messages for ClaudeAgentOptions
            if role == "system":
                system_parts.append(content)
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(str(content))

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        user_prompt = "\n\n".join(prompt_parts) if prompt_parts else ""

        return system_prompt, user_prompt

    def _map_tools_to_allowed_tools(self, tools: list[Any] | None) -> list[str]:
        """Map Agno tools to Claude Agent SDK allowed_tools list.

        Args:
            tools: List of Agno tools (ReasoningTools, ExaTools, Functions, etc.)

        Returns:
            List of tool names for Claude Agent SDK
        """
        if not tools:
            return []

        allowed_tools = []

        for tool in tools:
            # Handle tool classes
            if hasattr(tool, "__name__"):
                tool_name = tool.__name__
                # Map known Agno tools to Claude Agent SDK tools
                if "reasoning" in tool_name.lower():
                    allowed_tools.append("Think")
                elif "exa" in tool_name.lower():
                    allowed_tools.append("search_exa")
                else:
                    # For other tools, use class name as-is
                    allowed_tools.append(tool_name)
            # Handle Function objects
            elif hasattr(tool, "name"):
                allowed_tools.append(tool.name)
            # Handle dict-based tools
            elif isinstance(tool, dict) and "name" in tool:
                allowed_tools.append(tool["name"])

        logger.debug("Mapped tools to allowed_tools: %s", allowed_tools)
        return allowed_tools

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:  # noqa: ANN401
        """Extract tool calls from Claude Agent SDK message.

        Args:
            message: Message object from Claude Agent SDK

        Returns:
            List of tool call dictionaries
        """
        tool_calls = []

        # Check if message has tool_use blocks
        if hasattr(message, "content") and isinstance(message.content, list):
            for block in message.content:
                # Handle tool_use blocks
                if hasattr(block, "type") and block.type == "tool_use":
                    tool_call = {
                        "id": getattr(block, "id", "unknown"),
                        "type": "tool_use",
                        "name": getattr(block, "name", "unknown"),
                        "input": getattr(block, "input", {}),
                    }
                    tool_calls.append(tool_call)
                elif isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_calls.append(
                        {
                            "id": block.get("id", "unknown"),
                            "type": "tool_use",
                            "name": block.get("name", "unknown"),
                            "input": block.get("input", {}),
                        }
                    )

        return tool_calls

    async def aresponse(  # noqa: PLR0912
        self,
        messages: list[Message],
        response_format: dict[str, Any] | type | None = None,  # noqa: ARG002
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        tool_call_limit: int | None = None,
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
            # Extract system prompt and convert messages
            system_prompt, prompt = self._extract_system_and_messages(messages)

            # Map Agno tools to Claude Agent SDK allowed_tools
            allowed_tools = self._map_tools_to_allowed_tools(tools)

            # Create Claude Agent SDK options
            options_kwargs: dict[str, Any] = {
                "system_prompt": system_prompt,
                "max_turns": tool_call_limit or 10,
                "model": self.id,
                "permission_mode": self.permission_mode,
                "cwd": self.cwd,
            }

            # Add optional parameters if provided
            if allowed_tools:
                options_kwargs["allowed_tools"] = allowed_tools

            if self.mcp_servers is not None:
                options_kwargs["mcp_servers"] = self.mcp_servers

            if self.env:
                options_kwargs["env"] = self.env

            if self.add_dirs:
                options_kwargs["add_dirs"] = self.add_dirs

            if self.hooks is not None:
                options_kwargs["hooks"] = self.hooks

            if self.can_use_tool is not None:
                options_kwargs["can_use_tool"] = self.can_use_tool

            options = self._claude_options_class(**options_kwargs)

            logger.debug(
                "Claude Agent SDK query - prompt: %d chars, system: %d chars, "
                "allowed_tools: %s",
                len(prompt),
                len(system_prompt) if system_prompt else 0,
                allowed_tools,
            )

            # Collect response from Claude Agent SDK
            full_response = ""
            collected_tool_calls = []
            async for message in self._claude_query(prompt=prompt, options=options):
                # Extract tool calls from message
                tool_calls = self._extract_tool_calls(message)
                if tool_calls:
                    collected_tool_calls.extend(tool_calls)

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
                "Claude Agent SDK response - length: %d chars, tool_calls: %d",
                len(full_response),
                len(collected_tool_calls),
            )

            # Create Agno ModelResponse with tool calls
            return ModelResponse(
                role="assistant",
                content=full_response,
                tool_calls=collected_tool_calls if collected_tool_calls else [],
                provider_data={
                    "model_id": self.id,
                    "provider": "claude-agent-sdk",
                    "permission_mode": self.permission_mode,
                    "cwd": self.cwd,
                    "mcp_servers_configured": self.mcp_servers is not None,
                    "env_vars_count": len(self.env),
                    "add_dirs_count": len(self.add_dirs),
                    "hooks_configured": list(self.hooks.keys()) if self.hooks else [],
                    "can_use_tool_configured": self.can_use_tool is not None,
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

    async def aresponse_stream(  # noqa: PLR0912
        self,
        messages: list[Message],
        response_format: dict[str, Any] | type | None = None,  # noqa: ARG002
        tools: list[Any] | None = None,
        tool_choice: str | dict[str, Any] | None = None,  # noqa: ARG002
        tool_call_limit: int | None = None,
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
            # Extract system prompt and convert messages
            system_prompt, prompt = self._extract_system_and_messages(messages)

            # Map Agno tools to Claude Agent SDK allowed_tools
            allowed_tools = self._map_tools_to_allowed_tools(tools)

            # Create Claude Agent SDK options
            options_kwargs: dict[str, Any] = {
                "system_prompt": system_prompt,
                "max_turns": tool_call_limit or 10,
                "model": self.id,
                "permission_mode": self.permission_mode,
                "cwd": self.cwd,
            }

            # Add optional parameters if provided
            if allowed_tools:
                options_kwargs["allowed_tools"] = allowed_tools

            if self.mcp_servers is not None:
                options_kwargs["mcp_servers"] = self.mcp_servers

            if self.env:
                options_kwargs["env"] = self.env

            if self.add_dirs:
                options_kwargs["add_dirs"] = self.add_dirs

            if self.hooks is not None:
                options_kwargs["hooks"] = self.hooks

            if self.can_use_tool is not None:
                options_kwargs["can_use_tool"] = self.can_use_tool

            options = self._claude_options_class(**options_kwargs)

            logger.debug(
                "Claude Agent SDK streaming - prompt: %d chars, system: %d chars, "
                "allowed_tools: %s",
                len(prompt),
                len(system_prompt) if system_prompt else 0,
                allowed_tools,
            )

            async for message in self._claude_query(prompt=prompt, options=options):
                # Extract tool calls from message
                tool_calls = self._extract_tool_calls(message)

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

                # Yield response with content or tool calls
                if content or tool_calls:
                    yield ModelResponse(
                        role="assistant",
                        content=content,
                        tool_calls=tool_calls if tool_calls else [],
                        provider_data={
                            "model_id": self.id,
                            "provider": "claude-agent-sdk",
                            "streaming": True,
                            "permission_mode": self.permission_mode,
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
