"""Six Thinking Hats Intelligent Router.

基于问题复杂度和类型的智能路由系统，支持：
- 单帽模式：简单问题快速处理
- 双帽序列：中等问题平衡处理
- 三帽核心：标准问题深度处理
- 完整六帽：复杂问题全面处理
"""

# Lazy import to break circular dependency
import logging
from dataclasses import dataclass, field
from enum import Enum

from .ai_complexity_analyzer import AIComplexityAnalyzer
from .complexity_types import ComplexityMetrics

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.processors.six_hats_core import (
    HatColor,
    HatComplexity,
)

# logger already defined above


class ProblemType(Enum):
    """问题类型分类."""
    FACTUAL = "factual"              # 事实性问题
    EMOTIONAL = "emotional"          # 情感性问题
    CREATIVE = "creative"            # 创造性问题
    EVALUATIVE = "evaluative"        # 评估性问题
    PHILOSOPHICAL = "philosophical"  # 哲学性问题
    DECISION = "decision"            # 决策性问题
    GENERAL = "general"              # 一般性问题


@dataclass
class ProblemCharacteristics:
    """问题特征分析结果."""
    primary_type: ProblemType
    secondary_types: list[ProblemType] = field(default_factory=list)

    # 特征标记
    is_factual: bool = False
    is_creative: bool = False
    is_evaluative: bool = False
    is_philosophical: bool = False
    is_decision: bool = False
    needs_judgment: bool = False
    needs_improvement: bool = False

    # 文本特征
    question_count: int = 0
    complexity_indicators: int = 0
    creative_indicators: int = 0
    factual_indicators: int = 0


@dataclass
class HatSequenceStrategy:
    """帽子序列策略."""
    name: str
    complexity: HatComplexity
    hat_sequence: list[HatColor]
    estimated_time_seconds: int
    description: str
    recommended_for: list[ProblemType] = field(default_factory=list)


class ProblemAnalyzer:
    """问题类型和特征分析器."""

    # 问题类型识别关键词（支持中英文）
    TYPE_INDICATORS = {
        ProblemType.FACTUAL: {
            # 英文关键词
            "what", "when", "where", "who", "how many", "statistics", "data", "facts",
            "information", "definition", "explain", "describe", "list",
            # 中文关键词
            "什么", "何时", "哪里", "谁", "多少", "统计", "数据", "事实",
            "信息", "定义", "解释", "描述", "列出", "介绍"
        },
        ProblemType.EMOTIONAL: {
            # 英文关键词
            "feel", "emotion", "sense", "intuition", "gut", "heart", "passion",
            "worry", "excited", "concerned", "hopeful", "afraid",
            # 中文关键词
            "感觉", "情感", "感受", "直觉", "内心", "担心", "兴奋", "关心", "希望", "害怕"
        },
        ProblemType.CREATIVE: {
            # 英文关键词
            "creative", "innovative", "brainstorm", "alternative", "new idea",
            "think outside", "novel", "original", "imagination", "possibility",
            # 中文关键词
            "创造", "创新", "头脑风暴", "替代", "新想法", "新颖", "原创", "想象", "可能性"
        },
        ProblemType.EVALUATIVE: {
            # 英文关键词
            "evaluate", "assess", "compare", "judge", "rate", "pros and cons",
            "advantages", "disadvantages", "better", "worse", "best",
            # 中文关键词
            "评估", "评价", "比较", "判断", "评级", "优缺点", "优势", "劣势", "更好", "最好"
        },
        ProblemType.PHILOSOPHICAL: {
            # 英文关键词
            "meaning", "purpose", "existence", "philosophy", "ethics", "moral",
            "values", "beliefs", "truth", "reality", "consciousness",
            # 中文关键词
            "意义", "目的", "存在", "哲学", "伦理", "道德", "价值观", "信念", "真理", "现实",
            "意识", "生命", "死亡", "自由", "选择", "存在主义", "本质"
        },
        ProblemType.DECISION: {
            # 英文关键词
            "decide", "choose", "select", "option", "should", "recommend",
            "which", "pick", "decision", "choice", "dilemma",
            # 中文关键词
            "决定", "选择", "挑选", "选项", "应该", "推荐", "哪个", "决策", "两难"
        }
    }

    def analyze_problem(self, thought_data: ThoughtData) -> ProblemCharacteristics:
        """分析问题类型和特征."""
        text = thought_data.thought.lower()

        # 分析各种类型的指标
        type_scores = {}
        for problem_type, keywords in self.TYPE_INDICATORS.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text)
            type_scores[problem_type] = score

        # 确定主要类型
        primary_type = max(type_scores, key=type_scores.get)
        if type_scores[primary_type] == 0:
            primary_type = ProblemType.GENERAL

        # 确定次要类型（得分 > 0）
        secondary_types = [
            ptype for ptype, score in type_scores.items()
            if score > 0 and ptype != primary_type
        ]

        # 计算特征标记
        characteristics = ProblemCharacteristics(
            primary_type=primary_type,
            secondary_types=secondary_types,
            is_factual=type_scores[ProblemType.FACTUAL] > 0,
            is_creative=type_scores[ProblemType.CREATIVE] > 0,
            is_evaluative=type_scores[ProblemType.EVALUATIVE] > 0,
            is_philosophical=type_scores[ProblemType.PHILOSOPHICAL] > 0,
            is_decision=type_scores[ProblemType.DECISION] > 0,
            needs_judgment=any(word in text for word in ["judge", "evaluate", "assess", "判断", "评价"]),
            needs_improvement=any(word in text for word in ["improve", "better", "enhance", "改进", "改善"]),
            question_count=text.count("?") + text.count("？"),
            complexity_indicators=type_scores[ProblemType.PHILOSOPHICAL] + type_scores[ProblemType.DECISION],
            creative_indicators=type_scores[ProblemType.CREATIVE],
            factual_indicators=type_scores[ProblemType.FACTUAL]
        )

        logger.info(f"Problem analysis: primary={primary_type.value}, secondary={[t.value for t in secondary_types]}")
        return characteristics


class SixHatsSequenceLibrary:
    """六帽序列策略库."""

    # 预定义的帽子序列策略
    STRATEGIES = {
        # 单帽模式策略
        "single_factual": HatSequenceStrategy(
            name="单帽事实模式",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.WHITE],
            estimated_time_seconds=120,
            description="纯事实收集，快速信息处理",
            recommended_for=[ProblemType.FACTUAL]
        ),

        "single_intuitive": HatSequenceStrategy(
            name="单帽直觉模式",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.RED],
            estimated_time_seconds=30,
            description="快速直觉反应，30秒情感判断",
            recommended_for=[ProblemType.EMOTIONAL]
        ),

        "single_creative": HatSequenceStrategy(
            name="单帽创意模式",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.GREEN],
            estimated_time_seconds=240,
            description="创意生成模式，自由创新思考",
            recommended_for=[ProblemType.CREATIVE]
        ),

        "single_critical": HatSequenceStrategy(
            name="单帽批判模式",
            complexity=HatComplexity.SINGLE,
            hat_sequence=[HatColor.BLACK],
            estimated_time_seconds=120,
            description="风险识别，快速批判分析",
            recommended_for=[ProblemType.EVALUATIVE]
        ),

        # 双帽序列策略
        "evaluate_idea": HatSequenceStrategy(
            name="想法评估序列",
            complexity=HatComplexity.DOUBLE,
            hat_sequence=[HatColor.YELLOW, HatColor.BLACK],
            estimated_time_seconds=240,
            description="先看优点，再看风险，平衡评估",
            recommended_for=[ProblemType.EVALUATIVE]
        ),

        "improve_design": HatSequenceStrategy(
            name="设计改进序列",
            complexity=HatComplexity.DOUBLE,
            hat_sequence=[HatColor.BLACK, HatColor.GREEN],
            estimated_time_seconds=360,
            description="识别问题，然后创新改进",
            recommended_for=[ProblemType.CREATIVE, ProblemType.EVALUATIVE]
        ),

        "fact_and_judge": HatSequenceStrategy(
            name="事实判断序列",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.WHITE, HatColor.BLACK, HatColor.BLUE],
            estimated_time_seconds=360,
            description="收集事实，批判验证，蓝帽整合结论",
            recommended_for=[ProblemType.FACTUAL, ProblemType.EVALUATIVE]
        ),

        # 三帽核心序列策略
        "problem_solving": HatSequenceStrategy(
            name="问题解决序列",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.WHITE, HatColor.GREEN, HatColor.BLACK],
            estimated_time_seconds=480,
            description="事实→创意→评估，标准问题解决",
            recommended_for=[ProblemType.GENERAL, ProblemType.CREATIVE]
        ),

        "decision_making": HatSequenceStrategy(
            name="决策制定序列",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.RED, HatColor.YELLOW, HatColor.BLACK],
            estimated_time_seconds=390,
            description="直觉→价值→风险，快速决策",
            recommended_for=[ProblemType.DECISION]
        ),

        "philosophical_thinking": HatSequenceStrategy(
            name="哲学思考序列",
            complexity=HatComplexity.TRIPLE,
            hat_sequence=[HatColor.WHITE, HatColor.GREEN, HatColor.BLUE],
            estimated_time_seconds=540,
            description="事实→创造→整合，深度哲学思考（解决综合+评审分离问题）",
            recommended_for=[ProblemType.PHILOSOPHICAL]
        ),

        # 完整六帽序列
        "full_exploration": HatSequenceStrategy(
            name="全面探索序列",
            complexity=HatComplexity.FULL,
            hat_sequence=[HatColor.BLUE, HatColor.WHITE, HatColor.RED,
                         HatColor.YELLOW, HatColor.BLACK, HatColor.GREEN, HatColor.BLUE],
            estimated_time_seconds=780,
            description="完整六帽序列，全面深度分析",
            recommended_for=[ProblemType.DECISION, ProblemType.PHILOSOPHICAL]
        ),

        "creative_innovation": HatSequenceStrategy(
            name="创新发展序列",
            complexity=HatComplexity.FULL,
            hat_sequence=[HatColor.BLUE, HatColor.RED, HatColor.GREEN,
                         HatColor.WHITE, HatColor.YELLOW, HatColor.BLACK, HatColor.BLUE],
            estimated_time_seconds=840,
            description="创新优先的完整流程",
            recommended_for=[ProblemType.CREATIVE]
        )
    }

    @classmethod
    def get_strategy(cls, strategy_name: str) -> HatSequenceStrategy | None:
        """获取指定策略."""
        return cls.STRATEGIES.get(strategy_name)

    @classmethod
    def get_strategies_for_problem(cls, problem_type: ProblemType) -> list[HatSequenceStrategy]:
        """获取适合特定问题类型的策略."""
        return [
            strategy for strategy in cls.STRATEGIES.values()
            if problem_type in strategy.recommended_for
        ]

    @classmethod
    def get_strategies_by_complexity(cls, complexity: HatComplexity) -> list[HatSequenceStrategy]:
        """按复杂度获取策略."""
        return [
            strategy for strategy in cls.STRATEGIES.values()
            if strategy.complexity == complexity
        ]


@dataclass
class RoutingDecision:
    """路由决策结果."""
    strategy: HatSequenceStrategy
    reasoning: str
    problem_characteristics: ProblemCharacteristics
    complexity_metrics: ComplexityMetrics
    estimated_cost_reduction: float  # 相比原系统的成本降低百分比


class SixHatsIntelligentRouter:
    """六帽智能路由器."""

    def __init__(self, complexity_analyzer: AIComplexityAnalyzer | None = None) -> None:
        self.complexity_analyzer = complexity_analyzer or AIComplexityAnalyzer()
        self.problem_analyzer = ProblemAnalyzer()
        self.sequence_library = SixHatsSequenceLibrary()

        # 复杂度阈值配置
        self.complexity_thresholds = {
            HatComplexity.SINGLE: (0, 3),
            HatComplexity.DOUBLE: (3, 10),
            HatComplexity.TRIPLE: (10, 20),
            HatComplexity.FULL: (20, 100)
        }

    async def route_thought(self, thought_data: ThoughtData) -> RoutingDecision:
        """智能路由思想到最佳帽子序列."""
        logger.info("🎩 SIX HATS INTELLIGENT ROUTING:")
        logger.info(f"  📝 Input: {thought_data.thought[:100]}...")

        # 步骤1: 复杂度分析 (AI-powered)
        complexity_metrics = await self.complexity_analyzer.analyze(thought_data)
        complexity_score = complexity_metrics.complexity_score

        logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")

        # 步骤2: 问题特征分析
        problem_characteristics = self.problem_analyzer.analyze_problem(thought_data)

        logger.info(f"  🎯 Problem Type: {problem_characteristics.primary_type.value}")

        # 步骤3: 确定复杂度级别
        complexity_level = self._determine_complexity_level(complexity_score)

        logger.info(f"  📈 Complexity Level: {complexity_level.value}")

        # 步骤4: 策略选择
        strategy = self._select_optimal_strategy(
            complexity_level, problem_characteristics, complexity_score
        )

        # 步骤5: 生成决策说明
        reasoning = self._generate_reasoning(
            strategy, problem_characteristics, complexity_metrics
        )

        # 步骤6: 估算成本节约
        cost_reduction = self._estimate_cost_reduction(strategy, complexity_score)

        decision = RoutingDecision(
            strategy=strategy,
            reasoning=reasoning,
            problem_characteristics=problem_characteristics,
            complexity_metrics=complexity_metrics,
            estimated_cost_reduction=cost_reduction
        )

        logger.info(f"  ✅ Selected Strategy: {strategy.name}")
        logger.info(f"  🎨 Hat Sequence: {[hat.value for hat in strategy.hat_sequence]}")
        logger.info(f"  💰 Cost Reduction: {cost_reduction:.1f}%")

        return decision

    def _determine_complexity_level(self, score: float) -> HatComplexity:
        """根据复杂度分数确定处理级别."""
        for level, (min_score, max_score) in self.complexity_thresholds.items():
            if min_score <= score < max_score:
                return level
        return HatComplexity.FULL

    def _select_optimal_strategy(
        self,
        complexity_level: HatComplexity,
        problem_characteristics: ProblemCharacteristics,
        complexity_score: float
    ) -> HatSequenceStrategy:
        """选择最优策略."""
        # 获取该复杂度级别的所有策略
        candidate_strategies = self.sequence_library.get_strategies_by_complexity(complexity_level)

        # 如果没有找到策略，使用降级处理
        if not candidate_strategies:
            logger.warning(f"No strategies found for complexity {complexity_level}, using fallback")
            return self._get_fallback_strategy(complexity_level)

        # 根据问题类型筛选推荐策略
        recommended_strategies = [
            strategy for strategy in candidate_strategies
            if problem_characteristics.primary_type in strategy.recommended_for
        ]

        # 如果有推荐策略，选择第一个
        if recommended_strategies:
            return recommended_strategies[0]

        # 否则使用特殊逻辑选择
        return self._select_by_special_logic(
            candidate_strategies, problem_characteristics, complexity_score
        )

    def _select_by_special_logic(
        self,
        strategies: list[HatSequenceStrategy],
        characteristics: ProblemCharacteristics,
        complexity_score: float
    ) -> HatSequenceStrategy:
        """使用特殊逻辑选择策略."""
        # 单帽模式的特殊选择逻辑
        if strategies[0].complexity == HatComplexity.SINGLE:
            if characteristics.factual_indicators > characteristics.creative_indicators:
                return self.sequence_library.get_strategy("single_factual")
            if characteristics.creative_indicators > 0:
                return self.sequence_library.get_strategy("single_creative")
            if characteristics.needs_judgment:
                return self.sequence_library.get_strategy("single_intuitive")
            return self.sequence_library.get_strategy("single_factual")  # 默认

        # 其他复杂度：返回第一个策略
        return strategies[0]

    def _get_fallback_strategy(self, complexity_level: HatComplexity) -> HatSequenceStrategy:
        """获取降级策略."""
        fallback_map = {
            HatComplexity.SINGLE: "single_factual",
            HatComplexity.DOUBLE: "fact_and_judge",
            HatComplexity.TRIPLE: "problem_solving",
            HatComplexity.FULL: "full_exploration"
        }
        strategy_name = fallback_map.get(complexity_level, "single_factual")
        return self.sequence_library.get_strategy(strategy_name)

    def _generate_reasoning(
        self,
        strategy: HatSequenceStrategy,
        characteristics: ProblemCharacteristics,
        metrics: ComplexityMetrics
    ) -> str:
        """生成路由决策推理."""
        reasoning_parts = [
            f"Strategy: {strategy.name}",
            f"Problem type: {characteristics.primary_type.value}",
            f"Complexity: {metrics.complexity_score:.1f}/100",
            f"Hat sequence: {' → '.join(hat.value for hat in strategy.hat_sequence)}",
            f"Estimated time: {strategy.estimated_time_seconds}s"
        ]

        # 添加特征说明
        if characteristics.is_philosophical:
            reasoning_parts.append("Philosophical depth detected")
        if characteristics.is_creative:
            reasoning_parts.append("Creative thinking required")
        if characteristics.question_count > 0:
            reasoning_parts.append(f"{characteristics.question_count} questions found")

        return " | ".join(reasoning_parts)

    def _estimate_cost_reduction(self, strategy: HatSequenceStrategy, complexity_score: float) -> float:
        """估算相比原系统的成本降低."""
        # 原系统成本估算（基于复杂度）
        if complexity_score < 5:
            original_cost = 100  # 单agent成本基准
        elif complexity_score < 15:
            original_cost = 300  # 混合team成本
        else:
            original_cost = 600  # 完整多agent成本

        # 新系统成本（基于帽子数量和时间）
        hat_count = len(strategy.hat_sequence)
        new_cost = hat_count * 50 + strategy.estimated_time_seconds * 0.1

        # 计算降低百分比
        if original_cost > 0:
            reduction = max(0, (original_cost - new_cost) / original_cost * 100)
        else:
            reduction = 0

        return min(reduction, 85)  # 最大85%的降低


# 便利函数
def create_six_hats_router(complexity_analyzer=None) -> SixHatsIntelligentRouter:
    """创建六帽智能路由器."""
    return SixHatsIntelligentRouter(complexity_analyzer)


async def route_thought_to_hats(thought_data: ThoughtData) -> RoutingDecision:
    """将思想路由到最佳帽子序列."""
    router = SixHatsIntelligentRouter()
    return await router.route_thought(thought_data)
