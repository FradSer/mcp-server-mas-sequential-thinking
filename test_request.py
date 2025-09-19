#!/usr/bin/env python3
"""Test script to validate the MCP server with user input."""

import asyncio
import os
from src.mcp_server_mas_sequential_thinking.main import sequentialthinking
from src.mcp_server_mas_sequential_thinking.server_core import create_server_lifespan

async def test_user_request():
    """Test the exact user request that was failing."""

    # Test data from user
    test_input = {
        "thought": "如果生命终将结束，我们为什么要活着？",
        "thought_number": 1,
        "total_thoughts": 3,
        "next_needed": True
    }

    print("🧪 Testing MCP Server with user input...")
    print(f"   Input: {test_input}")

    try:
        # Initialize server first
        print("🚀 Initializing server...")
        async with create_server_lifespan() as server_state:
            # Set global server state (like in main.py)
            import src.mcp_server_mas_sequential_thinking.main as main_module
            main_module._server_state = server_state

            print("✅ Server initialized successfully")

            # Test the sequentialthinking function
            print("📝 Processing thought...")
            result = await sequentialthinking(**test_input)

            print("\n✅ SUCCESS! Server processed the request")
            print(f"📤 Response length: {len(result)} chars")
            print(f"📝 Response preview: {result[:200]}...")

            # Check if it's not an error response
            if "failed" not in result.lower() and "error" not in result.lower():
                print("✅ Response appears to be successful (no error keywords)")
                return result
            else:
                print("⚠️  Response may contain error - check full output")
                return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Set minimal environment for testing
    os.environ.setdefault("LLM_PROVIDER", "deepseek")
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-677f151a095040e4945b514998f24863")
    os.environ.setdefault("EXA_API_KEY", "33a8ff99-425d-424f-9de2-510dfc2dc79b")

    result = asyncio.run(test_user_request())

    if result:
        print(f"\n📄 FULL RESPONSE:\n{result}")