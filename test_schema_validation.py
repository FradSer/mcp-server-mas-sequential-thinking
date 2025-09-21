#!/usr/bin/env python3
"""Test schema validation for required fields."""

from pydantic import ValidationError
from src.mcp_server_mas_sequential_thinking.models import ThoughtData


def test_required_fields():
    """Test that all fields are required."""
    print("🧪 Testing required field validation...")

    # Test 1: Valid complete input
    try:
        valid_data = ThoughtData(
            thought="测试思考",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        print("✅ Valid complete input: PASSED")
    except Exception as e:
        print(f"❌ Valid complete input: FAILED - {e}")

    # Test 2: Missing required field (should fail)
    try:
        ThoughtData(
            thought="测试思考",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            # Missing isRevision - should fail
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        print("❌ Missing isRevision: FAILED - Should have raised ValidationError")
    except ValidationError:
        print("✅ Missing isRevision: PASSED - Correctly raised ValidationError")
    except Exception as e:
        print(f"❌ Missing isRevision: FAILED - Wrong error type: {e}")

    # Test 3: Missing another required field (should fail)
    try:
        ThoughtData(
            thought="测试思考",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            # Missing branchId - should fail
            needsMoreThoughts=False,
        )
        print("❌ Missing branchId: FAILED - Should have raised ValidationError")
    except ValidationError:
        print("✅ Missing branchId: PASSED - Correctly raised ValidationError")
    except Exception as e:
        print(f"❌ Missing branchId: FAILED - Wrong error type: {e}")

    # Test 4: Test with None values (should pass)
    try:
        valid_with_nulls = ThoughtData(
            thought="测试思考",
            thoughtNumber=1,
            totalThoughts=3,
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )
        print("✅ Explicit None values: PASSED")
    except Exception as e:
        print(f"❌ Explicit None values: FAILED - {e}")

    print("\n🎯 Schema validation test complete!")

if __name__ == "__main__":
    test_required_fields()
