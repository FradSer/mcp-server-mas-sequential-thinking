#!/usr/bin/env python3
"""Test MCP tool schema generation."""

import inspect

from src.mcp_server_mas_sequential_thinking.main import sequentialthinking


def test_mcp_schema():
    """Test MCP tool function signature."""
    print("🔍 Testing MCP tool function schema...")

    # Get function signature
    sig = inspect.signature(sequentialthinking)

    print("\n📋 Function parameters:")
    for name, param in sig.parameters.items():
        has_default = param.default != inspect.Parameter.empty
        param_type = param.annotation if param.annotation != inspect.Parameter.empty else "Any"

        print(f"  {name}: {param_type} {'(optional)' if has_default else '(required)'}")

        if has_default:
            print(f"    default: {param.default}")

    print("\n🎯 Required parameters (no defaults):")
    required_params = [
        name for name, param in sig.parameters.items()
        if param.default == inspect.Parameter.empty
    ]

    for param in required_params:
        print(f"  ✓ {param}")

    print(f"\n📊 Total parameters: {len(sig.parameters)}")
    print(f"📊 Required parameters: {len(required_params)}")

    # Expected required parameters
    expected_required = [
        "thought", "thoughtNumber", "totalThoughts", "nextThoughtNeeded",
        "isRevision", "branchFromThought", "branchId", "needsMoreThoughts"
    ]

    print(f"\n🎯 Expected required: {len(expected_required)}")
    print("Expected required parameters:")
    for param in expected_required:
        print(f"  ✓ {param}")

    # Validation
    if set(required_params) == set(expected_required):
        print("\n✅ SUCCESS: All expected parameters are required!")
        return True
    missing = set(expected_required) - set(required_params)
    unexpected = set(required_params) - set(expected_required)

    if missing:
        print(f"\n❌ Missing required parameters: {missing}")
    if unexpected:
        print(f"\n❌ Unexpected required parameters: {unexpected}")

    print("\n❌ FAILED: Parameter requirements don't match!")
    return False

if __name__ == "__main__":
    test_mcp_schema()
