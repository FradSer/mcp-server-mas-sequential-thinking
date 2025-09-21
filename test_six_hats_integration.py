#!/usr/bin/env python3
"""
Test script for Six Thinking Hats integration.

Tests the complete integration from problem input to Six Hats processing,
including the philosophical question example that was causing "synthesis + review" issues.
"""

import asyncio
import logging
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.mcp_server_mas_sequential_thinking.models import ThoughtData
from src.mcp_server_mas_sequential_thinking.six_hats_router import (
    SixHatsIntelligentRouter, route_thought_to_hats
)
from src.mcp_server_mas_sequential_thinking.six_hats_processor import (
    SixHatsSequentialProcessor, process_with_six_hats
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_philosophical_question():
    """Test the philosophical question that was causing synthesis+review separation."""

    print("🎩" + "="*70)
    print("Testing Six Hats with Philosophical Question")
    print("="*71)

    # The exact question that was causing issues
    philosophical_question = """如果生命终将结束，我们为什么要活着？"""

    thought_data = ThoughtData(
        thought=philosophical_question,
        thoughtNumber=1,
        totalThoughts=1,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )

    print(f"📝 Question: {philosophical_question}")
    print()

    # Test 1: Router Decision
    print("🧭 STEP 1: Testing Router Decision")
    print("-" * 40)

    try:
        routing_decision = route_thought_to_hats(thought_data)

        print(f"✅ Strategy Selected: {routing_decision.strategy.name}")
        print(f"🎨 Hat Sequence: {' → '.join(hat.value for hat in routing_decision.strategy.hat_sequence)}")
        print(f"📊 Complexity Score: {routing_decision.complexity_metrics.complexity_score:.1f}")
        print(f"💰 Cost Reduction: {routing_decision.estimated_cost_reduction:.1f}%")
        print(f"🎯 Problem Type: {routing_decision.problem_characteristics.primary_type.value}")

    except Exception as e:
        print(f"❌ Router test failed: {e}")
        return False

    print()

    # Test 2: Full Processing
    print("🎩 STEP 2: Testing Full Six Hats Processing")
    print("-" * 45)

    try:
        result = await process_with_six_hats(thought_data, "")

        print(f"✅ Processing Complete: {result.strategy_used}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f}s")
        print(f"🎨 Hats Used: {' → '.join(result.hat_sequence)}")
        print(f"💰 Cost Reduction: {result.cost_reduction:.1f}%")
        print()

        print("📋 FINAL OUTPUT:")
        print("-" * 20)
        print(result.content[:500] + "..." if len(result.content) > 500 else result.content)
        print()

        # Check if output is unified (no separate synthesis + review)
        has_synthesis_review_separation = (
            "综合" in result.content and "评审" in result.content and
            result.content.count("###") > 1  # Multiple sections
        )

        if has_synthesis_review_separation:
            print("⚠️  WARNING: Still showing synthesis + review separation")
            return False
        else:
            print("✅ SUCCESS: Unified output achieved (no synthesis + review separation)")

    except Exception as e:
        print(f"❌ Processing test failed: {e}")
        return False

    return True


async def test_simple_question():
    """Test a simple question that should use single hat mode."""

    print("\n🎩" + "="*70)
    print("Testing Six Hats with Simple Question")
    print("="*71)

    simple_question = "What is the capital of France?"

    thought_data = ThoughtData(
        thought=simple_question,
        thoughtNumber=1,
        totalThoughts=1,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )

    print(f"📝 Question: {simple_question}")
    print()

    try:
        routing_decision = route_thought_to_hats(thought_data)

        print(f"✅ Strategy Selected: {routing_decision.strategy.name}")
        print(f"🎨 Hat Sequence: {' → '.join(hat.value for hat in routing_decision.strategy.hat_sequence)}")
        print(f"📊 Complexity Score: {routing_decision.complexity_metrics.complexity_score:.1f}")

        # Should be single hat for simple factual question
        if routing_decision.strategy.complexity.value == "single":
            print("✅ SUCCESS: Simple question routed to single hat mode")
            return True
        else:
            print("⚠️  WARNING: Simple question not using single hat mode")
            return False

    except Exception as e:
        print(f"❌ Simple question test failed: {e}")
        return False


async def test_creative_question():
    """Test a creative question that should use green hat + others."""

    print("\n🎩" + "="*70)
    print("Testing Six Hats with Creative Question")
    print("="*71)

    creative_question = "How can we innovate education to better prepare students for the future?"

    thought_data = ThoughtData(
        thought=creative_question,
        thoughtNumber=1,
        totalThoughts=1,
        nextThoughtNeeded=True,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )

    print(f"📝 Question: {creative_question}")
    print()

    try:
        routing_decision = route_thought_to_hats(thought_data)

        print(f"✅ Strategy Selected: {routing_decision.strategy.name}")
        print(f"🎨 Hat Sequence: {' → '.join(hat.value for hat in routing_decision.strategy.hat_sequence)}")
        print(f"📊 Complexity Score: {routing_decision.complexity_metrics.complexity_score:.1f}")
        print(f"🎯 Problem Type: {routing_decision.problem_characteristics.primary_type.value}")

        # Should include green hat for creative question
        has_green_hat = any(hat.value == "green" for hat in routing_decision.strategy.hat_sequence)
        if has_green_hat:
            print("✅ SUCCESS: Creative question includes Green Hat")
            return True
        else:
            print("⚠️  WARNING: Creative question doesn't include Green Hat")
            return False

    except Exception as e:
        print(f"❌ Creative question test failed: {e}")
        return False


async def main():
    """Run all tests."""

    print("🎩 SIX THINKING HATS INTEGRATION TESTS")
    print("=" * 71)
    print()

    test_results = []

    # Run tests
    test_results.append(await test_philosophical_question())
    test_results.append(await test_simple_question())
    test_results.append(await test_creative_question())

    # Summary
    print("\n🎩" + "="*70)
    print("TEST SUMMARY")
    print("="*71)

    passed = sum(test_results)
    total = len(test_results)

    print(f"✅ Tests Passed: {passed}/{total}")
    print(f"❌ Tests Failed: {total - passed}/{total}")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Six Hats integration is working correctly.")
        print("   Key achievements:")
        print("   • ✅ Unified output (no synthesis + review separation)")
        print("   • ✅ Intelligent routing based on problem type")
        print("   • ✅ Cost optimization through appropriate strategy selection")
    else:
        print(f"\n⚠️  {total - passed} tests failed. See details above.")

    print()


if __name__ == "__main__":
    asyncio.run(main())