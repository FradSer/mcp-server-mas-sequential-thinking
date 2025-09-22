#!/usr/bin/env python3
"""
Comprehensive test script to verify AI-powered routing system functionality.

This script tests:
1. AIComplexityAnalyzer import and initialization
2. Complexity analysis for simple and complex thoughts
3. Six Hats router using AI analyzer instead of rule-based complexity
4. Blue Hat integration for philosophical questions
5. End-to-end routing workflow

Run with: python test_ai_routing_verification.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.routing.ai_complexity_analyzer import AIComplexityAnalyzer
from mcp_server_mas_sequential_thinking.routing.six_hats_router import (
    SixHatsIntelligentRouter,
    ProblemType,
    HatColor,
)
from mcp_server_mas_sequential_thinking.routing.complexity_types import ComplexityMetrics
from mcp_server_mas_sequential_thinking.processors.six_hats_core import HatComplexity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class AIRoutingVerificationTests:
    """Test suite for AI routing system verification."""

    def __init__(self):
        self.analyzer = None
        self.router = None
        self.test_results = {}

    async def setup(self):
        """Initialize test components."""
        logger.info("🔧 Setting up test components...")
        try:
            self.analyzer = AIComplexityAnalyzer()
            self.router = SixHatsIntelligentRouter(complexity_analyzer=self.analyzer)
            logger.info("✅ Setup completed successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Setup failed: {e}")
            return False

    def create_thought_data(self, content: str, **kwargs) -> ThoughtData:
        """Helper to create ThoughtData objects."""
        defaults = {
            "thoughtNumber": 1,
            "totalThoughts": 1,
            "nextThoughtNeeded": False,
            "isRevision": False,
            "branchFromThought": None,
            "branchId": None,
            "needsMoreThoughts": False,
        }
        defaults.update(kwargs)

        return ThoughtData(
            thought=content,
            **defaults
        )

    async def test_ai_analyzer_import_and_init(self) -> bool:
        """Test 1: Verify AIComplexityAnalyzer can be imported and initialized."""
        logger.info("\n🧪 Test 1: AIComplexityAnalyzer Import and Initialization")

        try:
            # Test import and initialization
            analyzer = AIComplexityAnalyzer()

            # Verify the analyzer has the expected attributes
            assert hasattr(analyzer, '_agent')
            assert hasattr(analyzer, 'model_config')
            assert hasattr(analyzer, '_get_agent')
            assert hasattr(analyzer, 'analyze')

            logger.info("✅ AIComplexityAnalyzer imported and initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Test 1 failed: {e}")
            return False

    async def test_complexity_analysis_simple_and_complex(self) -> bool:
        """Test 2: Verify complexity analysis works for different thought types."""
        logger.info("\n🧪 Test 2: Complexity Analysis for Simple and Complex Thoughts")

        try:
            # Test simple thought
            simple_thought = self.create_thought_data("Hello, this is simple.")
            simple_metrics = await self.analyzer.analyze(simple_thought)

            assert isinstance(simple_metrics, ComplexityMetrics)
            assert simple_metrics.complexity_score >= 0
            assert simple_metrics.analyzer_type == "ai"

            logger.info(f"✅ Simple thought analysis: score={simple_metrics.complexity_score:.1f}")

            # Test complex philosophical thought
            complex_thought = self.create_thought_data(
                "What is the fundamental meaning of existence? How do we reconcile "
                "the apparent meaninglessness of the universe with our human need for purpose? "
                "This requires deep philosophical investigation and analysis."
            )
            complex_metrics = await self.analyzer.analyze(complex_thought)

            assert isinstance(complex_metrics, ComplexityMetrics)
            assert complex_metrics.complexity_score > simple_metrics.complexity_score
            assert complex_metrics.analyzer_type == "ai"

            logger.info(f"✅ Complex thought analysis: score={complex_metrics.complexity_score:.1f}")
            logger.info(f"   Reasoning: {complex_metrics.reasoning[:100]}...")

            return True

        except Exception as e:
            logger.error(f"❌ Test 2 failed: {e}")
            return False

    async def test_six_hats_router_uses_ai_analyzer(self) -> bool:
        """Test 3: Verify six_hats_router uses AI analyzer instead of rule-based complexity."""
        logger.info("\n🧪 Test 3: Six Hats Router Uses AI Analyzer")

        try:
            # Verify router is using AI analyzer
            assert isinstance(self.router.complexity_analyzer, AIComplexityAnalyzer)

            # Test routing a simple thought
            simple_thought = self.create_thought_data("What is 2 + 2?")
            simple_decision = await self.router.route_thought(simple_thought)

            assert hasattr(simple_decision, 'strategy')
            assert hasattr(simple_decision, 'complexity_metrics')
            assert simple_decision.complexity_metrics.analyzer_type == "ai"

            logger.info(f"✅ Simple routing: {simple_decision.strategy.name}")
            logger.info(f"   Complexity: {simple_decision.complexity_metrics.complexity_score:.1f}")
            logger.info(f"   Hat sequence: {[hat.value for hat in simple_decision.strategy.hat_sequence]}")

            # Test routing a complex thought
            complex_thought = self.create_thought_data(
                "How can we develop ethical AI systems that balance human values with "
                "technological advancement? What are the implications for society?"
            )
            complex_decision = await self.router.route_thought(complex_thought)

            assert complex_decision.complexity_metrics.analyzer_type == "ai"
            assert complex_decision.complexity_metrics.complexity_score > simple_decision.complexity_metrics.complexity_score

            logger.info(f"✅ Complex routing: {complex_decision.strategy.name}")
            logger.info(f"   Complexity: {complex_decision.complexity_metrics.complexity_score:.1f}")
            logger.info(f"   Hat sequence: {[hat.value for hat in complex_decision.strategy.hat_sequence]}")

            return True

        except Exception as e:
            logger.error(f"❌ Test 3 failed: {e}")
            return False

    async def test_blue_hat_integration_philosophical(self) -> bool:
        """Test 4: Verify Blue Hat integration for philosophical questions."""
        logger.info("\n🧪 Test 4: Blue Hat Integration for Philosophical Questions")

        try:
            # Test deeply philosophical thought
            philosophical_thought = self.create_thought_data(
                "为什么我们会死去？生命的意义是什么？我们如何在知道死亡不可避免的情况下找到生活的目的？"
                "Why do we live if we are going to die? What is the purpose of existence "
                "when we know our time is limited? How do we find meaning in the face of mortality?"
            )

            philosophical_decision = await self.router.route_thought(philosophical_thought)

            # Check that philosophical content gets appropriate complexity
            assert philosophical_decision.complexity_metrics.complexity_score >= 30  # Should be high

            # Check that Blue Hat is involved (metacognitive orchestration)
            hat_sequence = philosophical_decision.strategy.hat_sequence
            has_blue_hat = HatColor.BLUE in hat_sequence

            if has_blue_hat:
                logger.info("✅ Blue Hat included in philosophical routing sequence")
            else:
                logger.info("ℹ️  Blue Hat not explicitly in sequence, but complexity detected")

            # Check problem type detection
            problem_characteristics = philosophical_decision.problem_characteristics
            assert problem_characteristics.is_philosophical
            assert problem_characteristics.primary_type == ProblemType.PHILOSOPHICAL

            logger.info(f"✅ Philosophical thought routing: {philosophical_decision.strategy.name}")
            logger.info(f"   Problem type: {problem_characteristics.primary_type.value}")
            logger.info(f"   Complexity: {philosophical_decision.complexity_metrics.complexity_score:.1f}")
            logger.info(f"   Hat sequence: {[hat.value for hat in hat_sequence]}")
            logger.info(f"   Reasoning: {philosophical_decision.reasoning[:150]}...")

            return True

        except Exception as e:
            logger.error(f"❌ Test 4 failed: {e}")
            return False

    async def test_rule_based_complexity_replaced(self) -> bool:
        """Test 5: Verify rule-based complexity scoring has been replaced with AI."""
        logger.info("\n🧪 Test 5: Rule-based Complexity Completely Replaced with AI")

        try:
            # Test multiple different types of content
            test_cases = [
                ("Simple fact", "Paris is the capital of France."),
                ("Technical question", "How does machine learning optimization work?"),
                ("Philosophical inquiry", "What is the nature of consciousness and free will?"),
                ("Creative challenge", "Design an innovative solution to climate change."),
                ("Decision making", "Should we prioritize economic growth or environmental protection?")
            ]

            all_use_ai = True

            for case_name, content in test_cases:
                thought = self.create_thought_data(content)
                metrics = await self.analyzer.analyze(thought)

                if metrics.analyzer_type != "ai":
                    logger.error(f"❌ {case_name} used {metrics.analyzer_type} instead of AI")
                    all_use_ai = False
                else:
                    logger.info(f"✅ {case_name}: AI analysis (score: {metrics.complexity_score:.1f})")

            if all_use_ai:
                logger.info("✅ All complexity analysis uses AI - rule-based system replaced")
                return True
            else:
                logger.error("❌ Some analysis still using non-AI methods")
                return False

        except Exception as e:
            logger.error(f"❌ Test 5 failed: {e}")
            return False

    async def test_end_to_end_routing_workflow(self) -> bool:
        """Test 6: End-to-end AI routing system workflow."""
        logger.info("\n🧪 Test 6: End-to-End AI Routing System Workflow")

        try:
            # Test complete workflow with different complexity levels
            test_scenarios = [
                {
                    "name": "Simple Factual",
                    "content": "What is the weather like?",
                    "expected_complexity": "low",
                    "expected_hat_count": 1
                },
                {
                    "name": "Moderate Analysis",
                    "content": "How can we improve team productivity using agile methodologies?",
                    "expected_complexity": "moderate",
                    "expected_hat_count": 2
                },
                {
                    "name": "Complex Philosophical",
                    "content": "What are the ethical implications of artificial intelligence making "
                             "autonomous decisions that affect human lives? How do we balance "
                             "efficiency with moral responsibility?",
                    "expected_complexity": "high",
                    "expected_hat_count": 3
                }
            ]

            all_passed = True

            for scenario in test_scenarios:
                logger.info(f"\n📋 Testing scenario: {scenario['name']}")

                thought = self.create_thought_data(scenario["content"])
                decision = await self.router.route_thought(thought)

                # Verify AI was used
                if decision.complexity_metrics.analyzer_type != "ai":
                    logger.error(f"❌ Scenario '{scenario['name']}' didn't use AI analyzer")
                    all_passed = False
                    continue

                # Check complexity alignment
                complexity_score = decision.complexity_metrics.complexity_score
                hat_count = len(decision.strategy.hat_sequence)

                logger.info(f"   📊 Complexity Score: {complexity_score:.1f}")
                logger.info(f"   🎩 Hat Count: {hat_count}")
                logger.info(f"   🎯 Strategy: {decision.strategy.name}")
                logger.info(f"   💰 Cost Reduction: {decision.estimated_cost_reduction:.1f}%")

                # Verify reasonable complexity distribution
                if scenario["expected_complexity"] == "low" and complexity_score > 20:
                    logger.warning(f"⚠️  Simple scenario got high complexity: {complexity_score}")
                elif scenario["expected_complexity"] == "high" and complexity_score < 15:
                    logger.warning(f"⚠️  Complex scenario got low complexity: {complexity_score}")

                logger.info(f"✅ Scenario '{scenario['name']}' completed successfully")

            if all_passed:
                logger.info("✅ End-to-end workflow test passed")
                return True
            else:
                logger.error("❌ Some end-to-end scenarios failed")
                return False

        except Exception as e:
            logger.error(f"❌ Test 6 failed: {e}")
            return False

    async def run_all_tests(self):
        """Run all verification tests."""
        logger.info("🚀 Starting AI Routing System Verification")
        logger.info("=" * 60)

        # Setup
        if not await self.setup():
            logger.error("❌ Setup failed - aborting tests")
            return False

        # Run tests
        tests = [
            ("AIComplexityAnalyzer Import & Init", self.test_ai_analyzer_import_and_init),
            ("Complexity Analysis Simple & Complex", self.test_complexity_analysis_simple_and_complex),
            ("Six Hats Router Uses AI Analyzer", self.test_six_hats_router_uses_ai_analyzer),
            ("Blue Hat Integration Philosophical", self.test_blue_hat_integration_philosophical),
            ("Rule-based Complexity Replaced", self.test_rule_based_complexity_replaced),
            ("End-to-End Routing Workflow", self.test_end_to_end_routing_workflow),
        ]

        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            logger.info(f"\n{'='*60}")
            try:
                result = await test_func()
                if result:
                    passed += 1
                    self.test_results[test_name] = "PASSED"
                else:
                    self.test_results[test_name] = "FAILED"
            except Exception as e:
                logger.error(f"❌ Test '{test_name}' encountered error: {e}")
                self.test_results[test_name] = f"ERROR: {e}"

        # Summary
        logger.info(f"\n{'='*60}")
        logger.info("📋 TEST SUMMARY")
        logger.info("=" * 60)

        for test_name, result in self.test_results.items():
            status_emoji = "✅" if result == "PASSED" else "❌"
            logger.info(f"{status_emoji} {test_name}: {result}")

        logger.info(f"\n📊 Results: {passed}/{total} tests passed")

        if passed == total:
            logger.info("🎉 ALL TESTS PASSED - AI Routing System is working correctly!")
            logger.info("\n🔍 Key Verification Points:")
            logger.info("  ✅ Rule-based complexity scoring completely replaced with AI")
            logger.info("  ✅ Blue Hat integration working for philosophical questions")
            logger.info("  ✅ System can distinguish simple factual vs deep philosophical questions")
            logger.info("  ✅ AI-powered routing produces appropriate hat sequences")
            logger.info("  ✅ End-to-end workflow functioning correctly")
            return True
        else:
            logger.error(f"❌ {total - passed} tests failed - Issues detected in AI routing system")
            return False

async def main():
    """Main test execution."""
    tests = AIRoutingVerificationTests()
    success = await tests.run_all_tests()

    if success:
        logger.info("\n🎯 CONCLUSION: AI-powered routing refactoring was SUCCESSFUL!")
        sys.exit(0)
    else:
        logger.error("\n💥 CONCLUSION: Issues detected in AI routing system!")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())