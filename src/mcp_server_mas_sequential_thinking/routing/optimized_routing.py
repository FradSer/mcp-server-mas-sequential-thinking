"""Optimized routing system with intelligent complexity thresholds and cost awareness."""

# Lazy import to break circular dependency
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.processors.six_hats_core import HatColor

from .six_hats_router import (
    HatComplexity,
    HatSequenceStrategy,
    ProblemCharacteristics,
    ProblemType,
    RoutingDecision,
)

# logger already defined above


class ProcessingMode(Enum):
    """Processing modes with different cost/quality tradeoffs."""
    FAST = "fast"  # Prioritize speed over depth
    BALANCED = "balanced"  # Balance cost and quality
    DEEP = "deep"  # Prioritize depth over cost


@dataclass
class OptimizedThresholds:
    """Optimized complexity thresholds based on real-world usage patterns."""

    # Much higher thresholds to prevent over-processing
    SINGLE_MAX = 8.0      # Increased from 3.0
    DOUBLE_MAX = 18.0     # Increased from 10.0
    TRIPLE_MAX = 35.0     # Increased from 20.0
    FULL_MIN = 40.0       # Increased from 20.0

    # Time-based thresholds (seconds)
    ACCEPTABLE_TIME = 45.0
    SLOW_WARNING_TIME = 120.0
    UNACCEPTABLE_TIME = 300.0

    # Cost-based thresholds (USD)
    LOW_COST_MAX = 0.005
    MEDIUM_COST_MAX = 0.015
    HIGH_COST_WARNING = 0.05


class OptimizedSequenceLibrary:
    """Optimized sequence library with cost-aware strategies."""

    STRATEGIES = {
        # Single hat strategies (0-8 complexity)
        "single_white": HatSequenceStrategy(
            name="事实查询",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.WHITE],
            estimated_time_seconds=15,
            description="快速事实收集，适用于简单查询",
            recommended_for=[ProblemType.FACTUAL]
        ),

        "single_green": HatSequenceStrategy(
            name="创意思考",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.GREEN],
            estimated_time_seconds=20,
            description="创意生成，适用于简单创新需求",
            recommended_for=[ProblemType.CREATIVE]
        ),

        "single_blue": HatSequenceStrategy(
            name="整合分析",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.BLUE],
            estimated_time_seconds=25,
            description="元认知整合，适用于需要总结的任务",
            recommended_for=[ProblemType.EVALUATIVE]
        ),

        # Efficient double strategies (8-18 complexity)
        "efficient_factual": HatSequenceStrategy(
            name="高效事实分析",
            complexity=HatComplexity.DOUBLE,
            hat_sequence=[HatColor.WHITE, HatColor.BLUE],
            estimated_time_seconds=45,
            description="事实收集+整合，高效处理复杂查询",
            recommended_for=[ProblemType.FACTUAL, ProblemType.EVALUATIVE]
        ),

        "efficient_creative": HatSequenceStrategy(
            name="高效创新流程",
            complexity=HatComplexity.DOUBLE,
            hat_sequence=[HatColor.GREEN, HatColor.YELLOW],
            estimated_time_seconds=50,
            description="创意+优化，平衡创新与实用性",
            recommended_for=[ProblemType.CREATIVE]
        ),

        # Balanced triple strategies (18-35 complexity)
        "balanced_analysis": HatSequenceStrategy(
            name="平衡分析流程",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.WHITE, HatColor.YELLOW, HatColor.BLUE],
            estimated_time_seconds=90,
            description="事实→价值→整合，适用于决策分析",
            recommended_for=[ProblemType.DECISION, ProblemType.EVALUATIVE]
        ),

        "balanced_philosophy": HatSequenceStrategy(
            name="哲学思辨流程",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.WHITE, HatColor.GREEN, HatColor.BLUE],
            estimated_time_seconds=120,
            description="事实→创造→整合，适用于哲学问题",
            recommended_for=[ProblemType.PHILOSOPHICAL]
        ),

        # Reserved for truly complex tasks (35+ complexity)
        "deep_exploration": HatSequenceStrategy(
            name="深度探索序列",
            complexity=HatComplexity.FULL,
            hat_sequence=[HatColor.BLUE, HatColor.WHITE, HatColor.YELLOW,
                         HatColor.BLACK, HatColor.GREEN, HatColor.BLUE],
            estimated_time_seconds=240,
            description="深度分析，仅用于极复杂问题",
            recommended_for=[ProblemType.DECISION, ProblemType.PHILOSOPHICAL]
        )
    }

    @classmethod
    def get_strategy(cls, strategy_name: str) -> HatSequenceStrategy | None:
        """Get strategy by name."""
        return cls.STRATEGIES.get(strategy_name)

    @classmethod
    def get_strategies_by_complexity(cls, complexity: HatComplexity) -> list[HatSequenceStrategy]:
        """Get all strategies for a complexity level."""
        return [strategy for strategy in cls.STRATEGIES.values()
                if strategy.complexity == complexity]


class CostAwareRouter:
    """Cost-aware router that prevents expensive processing unless justified."""

    def __init__(self, mode: ProcessingMode = ProcessingMode.BALANCED) -> None:
        self.mode = mode
        self.thresholds = OptimizedThresholds()
        self.sequence_library = OptimizedSequenceLibrary()

        # Cost multipliers based on mode
        self.cost_multipliers = {
            ProcessingMode.FAST: 0.5,      # Prefer cheaper options
            ProcessingMode.BALANCED: 1.0,   # Normal cost awareness
            ProcessingMode.DEEP: 2.0        # Allow higher costs for quality
        }

    def route_thought(self, thought_data: ThoughtData, force_complexity: float | None = None) -> RoutingDecision:
        """Route thought with cost awareness and intelligent thresholds."""
        # Calculate or use forced complexity
        if force_complexity is not None:
            complexity_score = force_complexity
        else:
            complexity_score = self._calculate_smart_complexity(thought_data)

        # Determine processing level with new thresholds
        processing_level = self._determine_processing_level(complexity_score)

        # Select strategy with cost awareness
        strategy = self._select_cost_aware_strategy(thought_data, processing_level, complexity_score)

        # Generate decision with cost justification
        reasoning = self._generate_cost_justification(strategy, complexity_score)

        # Calculate actual cost reduction vs naive approach
        cost_reduction = self._calculate_cost_reduction(strategy, complexity_score)

        return RoutingDecision(
            strategy=strategy,
            reasoning=reasoning,
            problem_characteristics=self._analyze_problem_quickly(thought_data),
            complexity_metrics=self._create_complexity_metrics(complexity_score),
            estimated_cost_reduction=cost_reduction
        )

    def _calculate_smart_complexity(self, thought_data: ThoughtData) -> float:
        """Calculate complexity with bias toward efficiency."""
        content = thought_data.thought.lower()

        # Base complexity from content length and structure
        base_complexity = min(len(content) / 50, 15.0)

        # Complexity indicators (reduced weights)
        complexity_indicators = 0

        # Question complexity (reduced impact)
        question_words = ["如何", "为什么", "怎样", "什么", "哪个", "how", "why", "what", "which"]
        complexity_indicators += sum(2 for word in question_words if word in content)

        # Philosophical indicators (reduced impact)
        philosophical_words = ["存在", "意义", "价值", "哲学", "思考", "existence", "meaning", "philosophy"]
        complexity_indicators += sum(1.5 for word in philosophical_words if word in content)

        # Academic indicators (reduced impact)
        academic_words = ["分析", "研究", "理论", "框架", "模型", "analysis", "research", "theory"]
        complexity_indicators += sum(1 for word in academic_words if word in content)

        # Creative indicators
        creative_words = ["创新", "创造", "设计", "想象", "创意", "creative", "innovation", "design"]
        complexity_indicators += sum(1 for word in creative_words if word in content)

        # Final complexity with cap to prevent over-processing
        final_complexity = base_complexity + complexity_indicators

        # Apply mode-based adjustment
        if self.mode == ProcessingMode.FAST:
            final_complexity *= 0.7  # Bias toward simpler processing
        elif self.mode == ProcessingMode.DEEP:
            final_complexity *= 1.3  # Allow more complex processing

        return min(final_complexity, 50.0)  # Hard cap at 50

    def _determine_processing_level(self, complexity_score: float) -> HatComplexity:
        """Determine processing level with optimized thresholds."""
        if complexity_score <= self.thresholds.SINGLE_MAX:
            return HatComplexity.SINGLE
        if complexity_score <= self.thresholds.DOUBLE_MAX:
            return HatComplexity.DOUBLE
        if complexity_score <= self.thresholds.TRIPLE_MAX:
            return HatComplexity.TRIPLE
        return HatComplexity.FULL

    def _select_cost_aware_strategy(
        self,
        thought_data: ThoughtData,
        level: HatComplexity,
        complexity_score: float
    ) -> HatSequenceStrategy:
        """Select strategy with cost awareness."""
        # Get available strategies for this level
        candidates = self.sequence_library.get_strategies_by_complexity(level)

        if not candidates:
            # Fallback to simpler level
            if level == HatComplexity.FULL:
                return self._select_cost_aware_strategy(thought_data, HatComplexity.TRIPLE, complexity_score)
            if level == HatComplexity.TRIPLE:
                return self._select_cost_aware_strategy(thought_data, HatComplexity.DOUBLE, complexity_score)
            if level == HatComplexity.DOUBLE:
                return self._select_cost_aware_strategy(thought_data, HatComplexity.SINGLE, complexity_score)
            return self.sequence_library.get_strategy("single_blue")  # Ultimate fallback

        # Select based on content type and cost awareness
        content = thought_data.thought.lower()

        # Content type detection
        is_factual = any(word in content for word in ["什么", "谁", "哪里", "什么时候", "事实", "数据"])
        is_creative = any(word in content for word in ["创新", "创造", "设计", "想象", "如何改进"])
        is_philosophical = any(word in content for word in ["为什么", "意义", "价值", "存在", "生命"])
        is_evaluative = any(word in content for word in ["分析", "比较", "评估", "判断"])

        # Strategy selection with cost bias
        if level == HatComplexity.SINGLE:
            if is_factual:
                return self.sequence_library.get_strategy("single_white")
            if is_creative:
                return self.sequence_library.get_strategy("single_green")
            return self.sequence_library.get_strategy("single_blue")

        if level == HatComplexity.DOUBLE:
            if is_factual or is_evaluative:
                return self.sequence_library.get_strategy("efficient_factual")
            return self.sequence_library.get_strategy("efficient_creative")

        if level == HatComplexity.TRIPLE:
            if is_philosophical:
                return self.sequence_library.get_strategy("balanced_philosophy")
            return self.sequence_library.get_strategy("balanced_analysis")

        # Only use deep exploration for truly complex scenarios
        if complexity_score > 40 and self.mode != ProcessingMode.FAST:
            return self.sequence_library.get_strategy("deep_exploration")
        # Downgrade to triple for cost efficiency
        return self.sequence_library.get_strategy("balanced_philosophy")

    def _generate_cost_justification(self, strategy: HatSequenceStrategy, complexity_score: float) -> str:
        """Generate reasoning that explains cost-benefit tradeoff."""
        time_estimate = strategy.estimated_time_seconds

        if time_estimate <= self.thresholds.ACCEPTABLE_TIME:
            efficiency = "高效"
        elif time_estimate <= self.thresholds.SLOW_WARNING_TIME:
            efficiency = "适中"
        else:
            efficiency = "深度"

        return (f"复杂度{complexity_score:.1f}→{strategy.complexity.value}级处理，"
                f"预计{time_estimate}秒({efficiency})，"
                f"优化成本效益比")

    def _calculate_cost_reduction(self, strategy: HatSequenceStrategy, complexity_score: float) -> float:
        """Calculate cost reduction vs naive full processing."""
        # Estimate what naive routing would have cost
        naive_time = 600  # Old system averaged 10 minutes
        actual_time = strategy.estimated_time_seconds

        return max(0, (naive_time - actual_time) / naive_time * 100)

    def _analyze_problem_quickly(self, thought_data: ThoughtData) -> ProblemCharacteristics:
        """Quick problem analysis for routing decision."""
        content = thought_data.thought.lower()

        # Determine primary type
        if any(word in content for word in ["什么", "事实", "数据", "信息"]):
            primary_type = ProblemType.FACTUAL
        elif any(word in content for word in ["创新", "创造", "设计", "想象"]):
            primary_type = ProblemType.CREATIVE
        elif any(word in content for word in ["决策", "选择", "判断", "比较"]):
            primary_type = ProblemType.DECISION
        elif any(word in content for word in ["为什么", "意义", "价值", "存在"]):
            primary_type = ProblemType.PHILOSOPHICAL
        else:
            primary_type = ProblemType.EVALUATIVE

        return ProblemCharacteristics(
            primary_type=primary_type,
            complexity_indicators=[],
            factual_indicators=1 if primary_type == ProblemType.FACTUAL else 0,
            creative_indicators=1 if primary_type == ProblemType.CREATIVE else 0,
            question_count=content.count("?") + content.count("？"),
            is_philosophical=primary_type == ProblemType.PHILOSOPHICAL
        )

    def _create_complexity_metrics(self, score: float):
        """Create minimal complexity metrics object."""
        class SimpleComplexityMetrics:
            def __init__(self, score: float) -> None:
                self.complexity_score = score

        return SimpleComplexityMetrics(score)


# Factory function for easy usage
def create_optimized_router(mode: ProcessingMode = ProcessingMode.BALANCED) -> CostAwareRouter:
    """Create an optimized router with specified processing mode."""
    return CostAwareRouter(mode)


# Quick test function
def test_routing_with_examples() -> None:
    """Test the optimized routing with example inputs."""
    router = create_optimized_router(ProcessingMode.BALANCED)

    test_cases = [
        "什么是人工智能？",  # Should be single
        "如何创新教育方法？",  # Should be double
        "分析哲学问题：如果生命终将结束，我们为什么要活着？",  # Should be triple
        "设计一个复杂的多维决策框架，整合经济学、心理学、哲学多个领域的理论，用于解决现代社会的道德伦理冲突"  # Should be full
    ]

    for i, thought_text in enumerate(test_cases, 1):
        thought_data = ThoughtData(
            thought=thought_text,
            thoughtNumber=i,
            totalThoughts=len(test_cases),
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False
        )

        router.route_thought(thought_data)


if __name__ == "__main__":
    test_routing_with_examples()
