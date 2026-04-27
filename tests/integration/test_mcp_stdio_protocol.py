"""Integration tests for MCP stdio using the Python MCP SDK."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _create_server_parameters() -> StdioServerParameters:
    """Create stdio server parameters for this project."""
    repo_root = Path(__file__).resolve().parents[2]
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server_mas_sequential_thinking.main"],
        env={**os.environ, "ENVIRONMENT": "production"},
        cwd=repo_root,
    )


def test_mcp_sdk_initialize_and_list_requests() -> None:
    """Python SDK client should initialize and list server tools/prompts."""

    async def run_assertions() -> None:
        server = _create_server_parameters()
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialize_result = await session.initialize()
                assert initialize_result.serverInfo.name == "FastMCP"

                tools_result = await session.list_tools()
                assert any(
                    tool.name == "sequentialthinking" for tool in tools_result.tools
                )
                sequential_tool = next(
                    tool
                    for tool in tools_result.tools
                    if tool.name == "sequentialthinking"
                )
                assert sequential_tool.description is not None
                assert "multi-step" in sequential_tool.description.lower()
                assert "actively use reflection" in sequential_tool.description.lower()
                assert sequential_tool.outputSchema is not None

                input_properties = sequential_tool.inputSchema.get("properties", {})
                assert "thought" in input_properties
                assert "description" in input_properties["nextThoughtNeeded"]

                output_properties = sequential_tool.outputSchema.get("properties", {})
                assert "should_continue" in output_properties
                assert "next_thought_number" in output_properties
                assert "stop_reason" in output_properties

                prompts_result = await session.list_prompts()
                assert any(
                    prompt.name == "sequential-thinking"
                    for prompt in prompts_result.prompts
                )

    anyio.run(run_assertions)


def test_mcp_sdk_tool_call_request_shape() -> None:
    """Python SDK tool call should return structured content payload."""

    async def run_assertions() -> None:
        server = _create_server_parameters()
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                # Intentionally invalid branch contract to avoid external LLM calls.
                call_result = await session.call_tool(
                    "sequentialthinking",
                    {
                        "thought": "Protocol smoke test",
                        "thoughtNumber": 1,
                        "totalThoughts": 1,
                        "nextThoughtNeeded": False,
                        "isRevision": False,
                        "branchFromThought": None,
                        "branchId": "branch-smoke",
                        "needsMoreThoughts": False,
                    },
                )

                assert isinstance(call_result.content, list)
                assert call_result.content
                assert "text" in call_result.content[0].model_dump()
                assert call_result.structuredContent is not None
                assert call_result.structuredContent["should_continue"] is True
                assert call_result.structuredContent["next_thought_number"] == 1
                assert (
                    call_result.structuredContent["stop_reason"] == "validation_error"
                )

    anyio.run(run_assertions)
