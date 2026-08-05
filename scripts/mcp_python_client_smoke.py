"""Smoke-test this MCP server with the official Python MCP client.

Usage:
  uv run --with mcp --with anyio python scripts/mcp_python_client_smoke.py
"""

from __future__ import annotations

from pathlib import Path

import anyio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def main() -> None:
    """Run initialize/list/call smoke tests via MCP Python client."""
    repo_root = Path(__file__).resolve().parents[1]
    server = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--extra",
            "dev",
            "python",
            "-m",
            "mcp_server_mas_sequential_thinking.main",
        ],
        env={"ENVIRONMENT": "production"},
        cwd=repo_root,
    )

    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            initialize_result = await session.initialize()
            print("initialize:", initialize_result)

            tools_result = await session.list_tools()
            print("tools:", [tool.name for tool in tools_result.tools])

            prompts_result = await session.list_prompts()
            print("prompts:", [prompt.name for prompt in prompts_result.prompts])

            # Intentionally invalid thoughtNumber keeps this as protocol-level smoke test
            # without requiring external LLM/API credentials.
            call_result = await session.call_tool(
                "sequentialthinking",
                {
                    "thought": "Protocol smoke test",
                    "thoughtNumber": 0,
                    "totalThoughts": 1,
                    "nextThoughtNeeded": False,
                    "isRevision": False,
                    "branchFromThought": None,
                    "branchId": None,
                    "needsMoreThoughts": False,
                },
            )
            print("call_tool is_error:", call_result.is_error)
            if call_result.content:
                print("call_tool first content:", call_result.content[0])


if __name__ == "__main__":
    anyio.run(main)
