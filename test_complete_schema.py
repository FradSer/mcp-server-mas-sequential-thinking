#!/usr/bin/env python3
"""Test complete MCP schema with exact format required."""

import asyncio
import os
from src.mcp_server_mas_sequential_thinking.main import sequentialthinking
from src.mcp_server_mas_sequential_thinking.server_core import create_server_lifespan

async def test_complete_schema():
    """Test the exact MCP input format required."""

    # Exact format as specified by user
    test_input = {
        "thought": "开始思考'什么是人'这个哲学问题的本质",
        "nextThoughtNeeded": True,
        "thoughtNumber": 1,
        "totalThoughts": 5,
        "isRevision": False,
        "branchFromThought": None,
        "branchId": None,
        "needsMoreThoughts": False
    }

    print("🧪 Testing complete MCP schema...")
    print(f"📋 Input format: {test_input}")

    try:
        # Initialize server first
        print("🚀 Initializing server...")
        async with create_server_lifespan() as server_state:
            # Set global server state
            import src.mcp_server_mas_sequential_thinking.main as main_module
            main_module._server_state = server_state

            print("✅ Server initialized successfully")

            # Test the sequentialthinking function with exact parameters
            print("📝 Testing MCP tool call...")
            result = await sequentialthinking(
                thought=test_input["thought"],
                thoughtNumber=test_input["thoughtNumber"],
                totalThoughts=test_input["totalThoughts"],
                nextThoughtNeeded=test_input["nextThoughtNeeded"],
                isRevision=test_input["isRevision"],
                branchFromThought=test_input["branchFromThought"],
                branchId=test_input["branchId"],
                needsMoreThoughts=test_input["needsMoreThoughts"]
            )

            print("\n✅ SUCCESS! MCP schema works correctly")
            print(f"📤 Response length: {len(result)} chars")
            print(f"📝 Response preview: {result[:200]}...")

            return result

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_schema_validation():
    """Test schema validation separately."""

    print("\n🔍 Testing schema validation...")

    from src.mcp_server_mas_sequential_thinking.models import ThoughtData

    try:
        # Test with exact camelCase format
        thought_data = ThoughtData(
            thought="开始思考'什么是人'这个哲学问题的本质",
            thoughtNumber=1,
            totalThoughts=5,
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False
        )

        print("✅ Schema validation: PASSED")
        print(f"📊 Thought type: {thought_data.thought_type.value}")
        print(f"📝 Formatted log:\n{thought_data.format_for_log()}")

        return True

    except Exception as e:
        print(f"❌ Schema validation: FAILED - {e}")
        return False

if __name__ == "__main__":
    # Set minimal environment for testing
    os.environ.setdefault("LLM_PROVIDER", "deepseek")
    os.environ.setdefault("DEEPSEEK_API_KEY", "sk-677f151a095040e4945b514998f24863")
    os.environ.setdefault("EXA_API_KEY", "33a8ff99-425d-424f-9de2-510dfc2dc79b")

    # Test schema validation first
    schema_ok = test_schema_validation()

    if schema_ok:
        # Test complete MCP functionality
        result = asyncio.run(test_complete_schema())

        if result:
            print(f"\n📄 FULL RESPONSE:\n{result}")
    else:
        print("\n❌ Schema validation failed, skipping MCP test")