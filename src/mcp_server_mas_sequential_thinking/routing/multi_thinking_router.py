"""Multi-Thinking Intelligent Router.

基于问题复杂度和类型的智能路由系统，支持：
- 单向模式：简单问题快速处理
- 双向序列：中等问题平衡处理
- 三向核心：标准问题深度处理
- 完整多向：复杂问题全面处理
"""

# Lazy import to break circular dependency
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .ai_complexity_analyzer import AIComplexityAnalyzer
from .complexity_types import ComplexityMetrics

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.processors.multi_thinking_core import (
    ProcessingDepth,
    ThinkingDirection,
)

# logger already defined above


@dataclass
class ThinkingSequenceStrategy:
    """思维序列策略."""

    name: str
    complexity: ProcessingDepth
    thinking_sequence: list[ThinkingDirection]
    estimated_time_seconds: int
    description: str


class ThinkingSequenceLibrary:
    """思维序列策略库."""

    # 预定义的思维序列策略
    STRATEGIES = {
        # 单向模式策略
        "single_factual": ThinkingSequenceStrategy(
            name="单向事实模式",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.FACTUAL],
            estimated_time_seconds=120,
            description="纯事实收集，快速信息处理",
        ),
        "single_intuitive": ThinkingSequenceStrategy(
            name="单向直觉模式",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.EMOTIONAL],
            estimated_time_seconds=30,
            description="快速直觉反应，30秒情感判断",
        ),
        "single_creative": ThinkingSequenceStrategy(
            name="单向创意模式",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.CREATIVE],
            estimated_time_seconds=240,
            description="创意生成模式，自由创新思考",
        ),
        "single_critical": ThinkingSequenceStrategy(
            name="单向批判模式",
            complexity=ProcessingDepth.SINGLE,
            thinking_sequence=[ThinkingDirection.CRITICAL],
            estimated_time_seconds=120,
            description="风险识别，快速批判分析",
        ),
        # 双向序列策略
        "evaluate_idea": ThinkingSequenceStrategy(
            name="想法评估序列",
            complexity=ProcessingDepth.DOUBLE,
            thinking_sequence=[ThinkingDirection.OPTIMISTIC, ThinkingDirection.CRITICAL],
            estimated_time_seconds=240,
            description="先看优点，再看风险，平衡评估",
        ),
        "improve_design": ThinkingSequenceStrategy(
            name="设计改进序列",
            complexity=ProcessingDepth.DOUBLE,
            thinking_sequence=[ThinkingDirection.CRITICAL, ThinkingDirection.CREATIVE],
            estimated_time_seconds=360,
            description="识别问题，然后创新改进",
        ),
        "fact_and_judge": ThinkingSequenceStrategy(
            name="事实判断序列",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[ThinkingDirection.FACTUAL, ThinkingDirection.CRITICAL, ThinkingDirection.SYNTHESIS],
            estimated_time_seconds=360,
            description="收集事实，批判验证，综合整合结论",
        ),
        # 三向核心序列策略
        "problem_solving": ThinkingSequenceStrategy(
            name="问题解决序列",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[ThinkingDirection.FACTUAL, ThinkingDirection.CREATIVE, ThinkingDirection.CRITICAL],
            estimated_time_seconds=480,
            description="事实→创意→评估，标准问题解决",
        ),
        "decision_making": ThinkingSequenceStrategy(
            name="决策制定序列",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[ThinkingDirection.EMOTIONAL, ThinkingDirection.OPTIMISTIC, ThinkingDirection.CRITICAL],
            estimated_time_seconds=390,
            description="直觉→价值→风险，快速决策",
        ),
        "philosophical_thinking": ThinkingSequenceStrategy(
            name="哲学思考序列",
            complexity=ProcessingDepth.TRIPLE,
            thinking_sequence=[ThinkingDirection.FACTUAL, ThinkingDirection.CREATIVE, ThinkingDirection.SYNTHESIS],
            estimated_time_seconds=540,
            description="事实→创造→整合，深度哲学思考（解决综合+评审分离问题）",
        ),
        # 完整多向序列
        "full_exploration": ThinkingSequenceStrategy(
            name="全面探索序列",
            complexity=ProcessingDepth.FULL,
            thinking_sequence=[
                ThinkingDirection.SYNTHESIS,
                ThinkingDirection.FACTUAL,
                ThinkingDirection.EMOTIONAL,
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=780,
            description="完整多向序列，全面深度分析",
        ),
        "creative_innovation": ThinkingSequenceStrategy(
            name="创新发展序列",
            complexity=ProcessingDepth.FULL,
            thinking_sequence=[
                ThinkingDirection.SYNTHESIS,
                ThinkingDirection.EMOTIONAL,
                ThinkingDirection.CREATIVE,
                ThinkingDirection.FACTUAL,
                ThinkingDirection.OPTIMISTIC,
                ThinkingDirection.CRITICAL,
                ThinkingDirection.SYNTHESIS,
            ],
            estimated_time_seconds=840,
            description="创新优先的完整流程",
        ),
    }

    @classmethod
    def get_strategy(cls, strategy_name: str) -> ThinkingSequenceStrategy | None:
        """获取指定策略."""
        return cls.STRATEGIES.get(strategy_name)


    @classmethod
    def get_strategies_by_complexity(
        cls, complexity: ProcessingDepth
    ) -> list[ThinkingSequenceStrategy]:
        """按复杂度获取策略."""
        return [
            strategy
            for strategy in cls.STRATEGIES.values()
            if strategy.complexity == complexity
        ]


@dataclass
class RoutingDecision:
    """路由决策结果."""

    strategy: ThinkingSequenceStrategy
    reasoning: str
    complexity_metrics: ComplexityMetrics
    estimated_cost_reduction: float  # 相比原系统的成本降低百分比
    problem_type: str  # AI-determined problem type
    thinking_modes_needed: list[str]  # AI-recommended thinking modes


class MultiThinkingIntelligentRouter:
    """多向思维智能路由器."""

    def __init__(self, complexity_analyzer: AIComplexityAnalyzer | None = None) -> None:
        self.complexity_analyzer = complexity_analyzer or AIComplexityAnalyzer()
        self.sequence_library = ThinkingSequenceLibrary()

        # 复杂度阈值配置
        self.complexity_thresholds = {
            ProcessingDepth.SINGLE: (0, 3),
            ProcessingDepth.DOUBLE: (3, 10),
            ProcessingDepth.TRIPLE: (10, 20),
            ProcessingDepth.FULL: (20, 100),
        }

    async def route_thought(self, thought_data: "ThoughtData") -> RoutingDecision:
        """AI-powered intelligent routing to optimal thinking sequence."""
        logger.info("🤖 AI-DRIVEN MULTI-THINKING ROUTING:")
        logger.info(f"  📝 Input: {thought_data.thought[:100]}...")

        # Step 1: AI analysis (complexity + problem type + thinking modes)
        complexity_metrics = await self.complexity_analyzer.analyze(thought_data)
        complexity_score = complexity_metrics.complexity_score

        # Extract AI analysis results
        ai_result = complexity_metrics.get("ai_analysis", {})
        problem_type = ai_result.get("primary_problem_type", "GENERAL")
        thinking_modes_needed = ai_result.get("thinking_modes_needed", ["SYNTHESIS"])

        logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")
        logger.info(f"  🎯 AI Problem Type: {problem_type}")
        logger.info(f"  🧠 Thinking Modes: {thinking_modes_needed}")

        # Step 2: Determine complexity level
        complexity_level = self._determine_complexity_level(complexity_score)
        logger.info(f"  📈 Complexity Level: {complexity_level.value}")

        # Step 3: AI-driven strategy selection
        strategy = self._select_optimal_strategy(
            complexity_level, problem_type, thinking_modes_needed, complexity_score
        )

        # Step 4: Generate reasoning
        reasoning = self._generate_reasoning(
            strategy, problem_type, thinking_modes_needed, complexity_metrics
        )

        # Step 5: Estimate cost reduction
        cost_reduction = self._estimate_cost_reduction(strategy, complexity_score)

        decision = RoutingDecision(
            strategy=strategy,
            reasoning=reasoning,
            complexity_metrics=complexity_metrics,
            estimated_cost_reduction=cost_reduction,
            problem_type=problem_type,
            thinking_modes_needed=thinking_modes_needed,
        )

        logger.info(f"  ✅ Selected Strategy: {strategy.name}")
        logger.info(
            f"  🎨 Thinking Sequence: {[direction.value for direction in strategy.thinking_sequence]}"
        )
        logger.info(f"  💰 Cost Reduction: {cost_reduction:.1f}%")

        return decision

    def _determine_complexity_level(self, score: float) -> ProcessingDepth:
        """根据复杂度分数确定处理级别."""
        for level, (min_score, max_score) in self.complexity_thresholds.items():
            if min_score <= score < max_score:
                return level
        return ProcessingDepth.FULL

    def _select_optimal_strategy(
        self,
        complexity_level: ProcessingDepth,
        problem_type: str,
        thinking_modes_needed: list[str],
        complexity_score: float,
    ) -> ThinkingSequenceStrategy:
        """AI-driven strategy selection."""
        # Get strategies by complexity level
        candidate_strategies = self.sequence_library.get_strategies_by_complexity(
            complexity_level
        )

        if not candidate_strategies:
            logger.warning(
                f"No strategies found for complexity {complexity_level}, using fallback"
            )
            return self._get_fallback_strategy(complexity_level)

        # AI-driven selection based on problem type and thinking modes
        return self._select_by_ai_analysis(
            candidate_strategies, problem_type, thinking_modes_needed, complexity_score
        )

    def _select_by_ai_analysis(
        self,
        strategies: list[ThinkingSequenceStrategy],
        problem_type: str,
        thinking_modes_needed: list[str],
        complexity_score: float,
    ) -> ThinkingSequenceStrategy:
        """AI-driven strategy selection logic."""
        # Single mode AI-driven selection
        if strategies[0].complexity == ProcessingDepth.SINGLE:
            if "FACTUAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_factual")
            if "CREATIVE" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_creative")
            if "EMOTIONAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_intuitive")
            if "CRITICAL" in thinking_modes_needed:
                return self.sequence_library.get_strategy("single_critical")
            return self.sequence_library.get_strategy("single_factual")  # Default

        # For other complexity levels: intelligent selection based on problem type
        if problem_type == "PHILOSOPHICAL":
            return self.sequence_library.get_strategy("philosophical_thinking")
        if problem_type == "DECISION":
            return self.sequence_library.get_strategy("decision_making")
        if problem_type == "CREATIVE":
            return self.sequence_library.get_strategy("creative_innovation") or strategies[0]
        if problem_type == "EVALUATIVE":
            return self.sequence_library.get_strategy("evaluate_idea") or strategies[0]

        # Default: return first strategy
        return strategies[0]

    def _get_fallback_strategy(
        self, complexity_level: ProcessingDepth
    ) -> ThinkingSequenceStrategy:
        """获取降级策略."""
        fallback_map = {
            ProcessingDepth.SINGLE: "single_factual",
            ProcessingDepth.DOUBLE: "fact_and_judge",
            ProcessingDepth.TRIPLE: "problem_solving",
            ProcessingDepth.FULL: "full_exploration",
        }
        strategy_name = fallback_map.get(complexity_level, "single_factual")
        return self.sequence_library.get_strategy(strategy_name)

    def _generate_reasoning(
        self,
        strategy: ThinkingSequenceStrategy,
        problem_type: str,
        thinking_modes_needed: list[str],
        metrics: ComplexityMetrics,
    ) -> str:
        """Generate AI-driven routing decision reasoning."""
        reasoning_parts = [
            f"Strategy: {strategy.name}",
            f"AI Problem Type: {problem_type}",
            f"Complexity: {metrics.complexity_score:.1f}/100",
            f"Thinking Sequence: {' → '.join(direction.value for direction in strategy.thinking_sequence)}",
            f"Estimated Time: {strategy.estimated_time_seconds}s",
            f"AI Recommended Modes: {', '.join(thinking_modes_needed)}",
        ]

        # Add AI insights
        if "PHILOSOPHICAL" in problem_type:
            reasoning_parts.append("Deep philosophical analysis required")
        if "CREATIVE" in thinking_modes_needed:
            reasoning_parts.append("Creative thinking essential")
        if "SYNTHESIS" in thinking_modes_needed:
            reasoning_parts.append("Integration and synthesis needed")

        return " | ".join(reasoning_parts)

    def _estimate_cost_reduction(
        self, strategy: ThinkingSequenceStrategy, complexity_score: float
    ) -> float:
        """估算相比原系统的成本降低."""
        # 原系统成本估算（基于复杂度）
        if complexity_score < 5:
            original_cost = 100  # 单agent成本基准
        elif complexity_score < 15:
            original_cost = 300  # 混合team成本
        else:
            original_cost = 600  # 完整多agent成本

        # 新系统成本（基于思维方向数量和时间）
        thinking_count = len(strategy.thinking_sequence)
        new_cost = thinking_count * 50 + strategy.estimated_time_seconds * 0.1

        # 计算降低百分比
        if original_cost > 0:
            reduction = max(0, (original_cost - new_cost) / original_cost * 100)
        else:
            reduction = 0

        return min(reduction, 85)  # 最大85%的降低


# 便利函数
def create_multi_thinking_router(complexity_analyzer=None) -> MultiThinkingIntelligentRouter:
    """创建多向思维智能路由器."""
    return MultiThinkingIntelligentRouter(complexity_analyzer)


async def route_thought_to_thinking(thought_data: "ThoughtData") -> RoutingDecision:
    """将思想路由到最佳思维序列."""
    router = MultiThinkingIntelligentRouter()
    return await router.route_thought(thought_data)
