"""Adaptive routing system for intelligent complexity-based agent selection."""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .models import ThoughtData

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Thought complexity levels for routing decisions."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


class ProcessingStrategy(Enum):
    """Available processing strategies."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    HYBRID = "hybrid"


@dataclass
class ComplexityMetrics:
    """Metrics for thought complexity analysis."""

    word_count: int
    sentence_count: int
    question_count: int
    technical_terms: int
    branching_references: int
    research_indicators: int
    analysis_depth: int

    @property
    def complexity_score(self) -> float:
        """Calculate overall complexity score (0-100)."""
        # Weighted scoring system
        score = (
            min(self.word_count / 20, 15)  # Word count (max 15 points)
            + min(self.sentence_count * 2, 10)  # Sentence count (max 10 points)
            + min(self.question_count * 3, 15)  # Questions (max 15 points)
            + min(self.technical_terms * 2, 20)  # Technical terms (max 20 points)
            + min(self.branching_references * 5, 15)  # Branching (max 15 points)
            + min(self.research_indicators * 3, 15)  # Research needs (max 15 points)
            + min(self.analysis_depth * 2, 10)  # Analysis depth (max 10 points)
        )
        return min(score, 100.0)


@dataclass
class RoutingDecision:
    """Decision result from adaptive routing."""

    strategy: ProcessingStrategy
    complexity_level: ComplexityLevel
    complexity_score: float
    reasoning: str
    estimated_token_usage: Tuple[int, int]  # (min, max)
    estimated_cost: float
    specialist_recommendations: List[str] = field(default_factory=list)


class ComplexityAnalyzer(ABC):
    """Abstract base class for complexity analysis."""

    @abstractmethod
    def analyze(self, thought_data: ThoughtData) -> ComplexityMetrics:
        """Analyze thought complexity and return metrics."""
        pass


class BasicComplexityAnalyzer(ComplexityAnalyzer):
    """Basic complexity analyzer using text analysis with Chinese language support."""

    # Technical terms that indicate complexity (English + Chinese)
    TECHNICAL_TERMS = {
        # English terms
        "algorithm",
        "implementation",
        "architecture",
        "framework",
        "optimization",
        "scalability",
        "performance",
        "security",
        "integration",
        "methodology",
        "paradigm",
        "pattern",
        "structure",
        "analysis",
        "synthesis",
        "evaluation",
        "research",
        "investigation",
        "comparison",
        "assessment",
        "validation",
        # Chinese philosophical and academic terms
        "哲学", "心理学", "存在主义", "萨特", "加缪", "尼采", "康德", "亚里士多德",
        "佛教", "道家", "儒家", "意义", "存在", "本质", "自由", "选择", "死亡", "生命",
        "理论", "框架", "方法论", "模式", "体系", "结构", "分析", "综合", "评估",
        "研究", "调查", "比较", "评价", "验证", "实证", "逻辑", "推理", "论证",
        "概念", "范式", "维度", "层面", "角度", "视角", "观点", "立场", "态度",
        "价值观", "世界观", "人生观", "道德", "伦理", "美学", "认识论", "本体论",
        "现象学", "解构主义", "后现代", "结构主义", "功能主义", "行为主义",
        "认知", "意识", "潜意识", "心理", "精神", "情感", "感知", "体验", "经验",
        "直觉", "理性", "感性", "主观", "客观", "相对", "绝对", "必然", "偶然",
        "因果", "关系", "联系", "影响", "作用", "机制", "过程", "发展", "变化",
        "趋势", "规律", "原则", "标准", "准则", "依据", "根据", "基础", "前提",
        "假设", "条件", "环境", "背景", "语境", "情境", "场景", "状况", "情况"
    }

    # Research indicators (English + Chinese)
    RESEARCH_INDICATORS = {
        # English terms
        "find out",
        "investigate",
        "research",
        "look up",
        "explore",
        "discover",
        "analyze",
        "study",
        "examine",
        "compare",
        "evaluate",
        "assess",
        # Chinese research terms
        "研究", "调查", "探索", "发现", "分析", "学习", "研读", "考察", "检视",
        "比较", "评估", "评价", "审视", "观察", "调研", "探讨", "探究", "深入",
        "了解", "理解", "掌握", "查找", "搜索", "寻找", "找出", "查明", "弄清",
        "揭示", "揭露", "挖掘", "阐述", "阐释", "解释", "说明", "论述", "论证"
    }

    # Branching indicators (English + Chinese)
    BRANCHING_INDICATORS = {
        # English terms
        "branch",
        "alternative",
        "option",
        "possibility",
        "scenario",
        "approach",
        "strategy",
        "method",
        "way",
        "path",
        "route",
        "direction",
        # Chinese branching terms
        "分支", "分叉", "替代", "选择", "可能性", "情况", "场景", "方案", "策略",
        "方法", "方式", "途径", "路径", "路线", "方向", "角度", "视角", "层面",
        "维度", "方面", "思路", "思考", "考虑", "权衡", "比较", "对比", "或者",
        "或", "还是", "另外", "另一", "其他", "别的", "不同", "多种", "各种"
    }

    def analyze(self, thought_data: ThoughtData) -> ComplexityMetrics:
        """Analyze thought complexity using basic text analysis with Chinese support."""
        text = thought_data.thought.lower()

        # Basic text metrics with Chinese support
        # For Chinese: estimate word count by character count / 2 (rough approximation)
        # For mixed content: use both space-split and character-based counting
        space_words = text.split()
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        estimated_chinese_words = chinese_chars // 2
        total_words = len(space_words) + estimated_chinese_words

        # Sentence splitting with Chinese punctuation support
        sentences = re.split(r"[.!?。！？]+", text)
        questions = text.count("?") + text.count("？")

        # Advanced analysis - check both individual words and substrings for Chinese
        technical_terms = 0
        for term in self.TECHNICAL_TERMS:
            if term in text:
                technical_terms += 1

        research_indicators = sum(
            1 for phrase in self.RESEARCH_INDICATORS if phrase in text
        )
        branching_references = sum(
            1 for phrase in self.BRANCHING_INDICATORS if phrase in text
        )

        # Analysis depth indicators with Chinese support
        analysis_depth = (
            # English connectors
            text.count("because") + text.count("therefore") + text.count("however") +
            text.count("moreover") + text.count("furthermore") + text.count("consequently") +
            # Chinese connectors
            text.count("因为") + text.count("所以") + text.count("因此") + text.count("然而") +
            text.count("但是") + text.count("不过") + text.count("而且") + text.count("并且") +
            text.count("另外") + text.count("此外") + text.count("由于") + text.count("既然") +
            text.count("如果") + text.count("假如") + text.count("虽然") + text.count("尽管")
        )

        # Check for branching context
        if thought_data.branch_from is not None:
            branching_references += 2  # Bonus for actual branching

        return ComplexityMetrics(
            word_count=total_words,
            sentence_count=len([s for s in sentences if s.strip()]),
            question_count=questions,
            technical_terms=technical_terms,
            branching_references=branching_references,
            research_indicators=research_indicators,
            analysis_depth=analysis_depth,
        )


class CostEstimator:
    """Estimates token usage and costs for different processing strategies."""

    # Token usage estimates based on historical data
    TOKEN_ESTIMATES = {
        ProcessingStrategy.SINGLE_AGENT: {
            ComplexityLevel.SIMPLE: (400, 800),
            ComplexityLevel.MODERATE: (600, 1200),
            ComplexityLevel.COMPLEX: (800, 1600),
            ComplexityLevel.HIGHLY_COMPLEX: (1000, 2000),
        },
        ProcessingStrategy.MULTI_AGENT: {
            ComplexityLevel.SIMPLE: (2000, 4000),
            ComplexityLevel.MODERATE: (3000, 6000),
            ComplexityLevel.COMPLEX: (4000, 8000),
            ComplexityLevel.HIGHLY_COMPLEX: (5000, 10000),
        },
        ProcessingStrategy.HYBRID: {
            ComplexityLevel.SIMPLE: (400, 800),  # Same as single agent
            ComplexityLevel.MODERATE: (1200, 2400),  # Partial multi-agent
            ComplexityLevel.COMPLEX: (3000, 6000),  # Full multi-agent
            ComplexityLevel.HIGHLY_COMPLEX: (4000, 8000),
        },
    }

    # Cost per 1K tokens (approximate averages)
    COST_PER_1K_TOKENS = {
        "deepseek": 0.0002,
        "groq": 0.0001,  # Often free tier
        "openrouter": 0.001,
        "github": 0.0005,
        "ollama": 0.0000,  # Local
    }

    def estimate_cost(
        self,
        strategy: ProcessingStrategy,
        complexity_level: ComplexityLevel,
        provider: str = "deepseek",
    ) -> Tuple[Tuple[int, int], float]:
        """Estimate token usage and cost."""
        token_range = self.TOKEN_ESTIMATES[strategy][complexity_level]
        cost_per_token = self.COST_PER_1K_TOKENS.get(provider, 0.0002) / 1000
        estimated_cost = (token_range[0] + token_range[1]) / 2 * cost_per_token

        return token_range, estimated_cost


class AdaptiveRouter:
    """Main adaptive routing system for intelligent agent selection."""

    def __init__(
        self,
        analyzer: Optional[ComplexityAnalyzer] = None,
        cost_estimator: Optional[CostEstimator] = None,
        budget_limit: Optional[float] = None,
    ):
        self.analyzer = analyzer or BasicComplexityAnalyzer()
        self.cost_estimator = cost_estimator or CostEstimator()
        self.budget_limit = budget_limit

        # Routing thresholds (can be configured)
        self.thresholds = {
            ComplexityLevel.SIMPLE: (0, 25),
            ComplexityLevel.MODERATE: (25, 50),
            ComplexityLevel.COMPLEX: (50, 75),
            ComplexityLevel.HIGHLY_COMPLEX: (75, 100),
        }

        # Strategy selection rules
        self.strategy_rules = {
            ComplexityLevel.SIMPLE: ProcessingStrategy.SINGLE_AGENT,
            ComplexityLevel.MODERATE: ProcessingStrategy.HYBRID,
            ComplexityLevel.COMPLEX: ProcessingStrategy.MULTI_AGENT,
            ComplexityLevel.HIGHLY_COMPLEX: ProcessingStrategy.MULTI_AGENT,
        }

    def route_thought(
        self,
        thought_data: ThoughtData,
        provider: str = "deepseek",
        budget_remaining: Optional[float] = None,
    ) -> RoutingDecision:
        """Route thought to appropriate processing strategy."""

        # Analyze complexity
        metrics = self.analyzer.analyze(thought_data)
        complexity_level = self._determine_complexity_level(metrics.complexity_score)

        # Select initial strategy
        strategy = self.strategy_rules[complexity_level]

        # Cost-aware strategy adjustment
        if budget_remaining is not None or self.budget_limit is not None:
            strategy = self._adjust_for_budget(
                strategy, complexity_level, provider, budget_remaining
            )

        # Estimate costs
        token_range, estimated_cost = self.cost_estimator.estimate_cost(
            strategy, complexity_level, provider
        )

        # Generate specialist recommendations
        specialist_recommendations = self._recommend_specialists(
            metrics, complexity_level, strategy
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            metrics, complexity_level, strategy, estimated_cost
        )

        decision = RoutingDecision(
            strategy=strategy,
            complexity_level=complexity_level,
            complexity_score=metrics.complexity_score,
            reasoning=reasoning,
            estimated_token_usage=token_range,
            estimated_cost=estimated_cost,
            specialist_recommendations=specialist_recommendations,
        )

        logger.info(
            f"Routing decision: {strategy.value} for {complexity_level.value} "
            f"thought (score: {metrics.complexity_score:.1f}, "
            f"estimated cost: ${estimated_cost:.4f})"
        )

        return decision

    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score."""
        for level, (min_score, max_score) in self.thresholds.items():
            if min_score <= score < max_score:
                return level
        return ComplexityLevel.HIGHLY_COMPLEX

    def _adjust_for_budget(
        self,
        strategy: ProcessingStrategy,
        complexity_level: ComplexityLevel,
        provider: str,
        budget_remaining: Optional[float],
    ) -> ProcessingStrategy:
        """Adjust strategy based on budget constraints."""
        if budget_remaining is None:
            return strategy

        _, estimated_cost = self.cost_estimator.estimate_cost(
            strategy, complexity_level, provider
        )

        # If cost exceeds budget, try cheaper alternatives
        if estimated_cost > budget_remaining:
            if strategy == ProcessingStrategy.MULTI_AGENT:
                # Try hybrid first
                _, hybrid_cost = self.cost_estimator.estimate_cost(
                    ProcessingStrategy.HYBRID, complexity_level, provider
                )
                if hybrid_cost <= budget_remaining:
                    logger.info(f"Budget constraint: switching to hybrid strategy")
                    return ProcessingStrategy.HYBRID

                # Fall back to single agent
                _, single_cost = self.cost_estimator.estimate_cost(
                    ProcessingStrategy.SINGLE_AGENT, complexity_level, provider
                )
                if single_cost <= budget_remaining:
                    logger.info(
                        f"Budget constraint: switching to single-agent strategy"
                    )
                    return ProcessingStrategy.SINGLE_AGENT

            elif strategy == ProcessingStrategy.HYBRID:
                # Fall back to single agent
                _, single_cost = self.cost_estimator.estimate_cost(
                    ProcessingStrategy.SINGLE_AGENT, complexity_level, provider
                )
                if single_cost <= budget_remaining:
                    logger.info(
                        f"Budget constraint: switching to single-agent strategy"
                    )
                    return ProcessingStrategy.SINGLE_AGENT

        return strategy

    def _recommend_specialists(
        self,
        metrics: ComplexityMetrics,
        complexity_level: ComplexityLevel,
        strategy: ProcessingStrategy,
    ) -> List[str]:
        """Recommend which specialists to use based on analysis."""
        recommendations = []

        if strategy == ProcessingStrategy.SINGLE_AGENT:
            return ["general"]  # Use general-purpose single agent

        # Multi-agent or hybrid recommendations
        if metrics.research_indicators > 0:
            recommendations.append("researcher")

        if metrics.analysis_depth > 2 or complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.HIGHLY_COMPLEX,
        ]:
            recommendations.append("analyzer")

        if metrics.technical_terms > 2:
            recommendations.append("planner")

        # Always include synthesizer for multi-agent
        if strategy == ProcessingStrategy.MULTI_AGENT:
            recommendations.append("synthesizer")

        # Add critic for complex thoughts
        if complexity_level in [
            ComplexityLevel.COMPLEX,
            ComplexityLevel.HIGHLY_COMPLEX,
        ]:
            recommendations.append("critic")

        return recommendations or ["planner", "synthesizer"]  # Default fallback

    def _generate_reasoning(
        self,
        metrics: ComplexityMetrics,
        complexity_level: ComplexityLevel,
        strategy: ProcessingStrategy,
        estimated_cost: float,
    ) -> str:
        """Generate human-readable reasoning for the routing decision."""
        reasoning_parts = [
            f"Complexity analysis: {complexity_level.value} "
            f"(score: {metrics.complexity_score:.1f}/100)"
        ]

        if metrics.word_count > 50:
            reasoning_parts.append(f"High word count ({metrics.word_count} words)")

        if metrics.technical_terms > 0:
            reasoning_parts.append(
                f"Contains {metrics.technical_terms} technical terms"
            )

        if metrics.research_indicators > 0:
            reasoning_parts.append(
                f"Requires research ({metrics.research_indicators} indicators)"
            )

        if metrics.branching_references > 0:
            reasoning_parts.append(
                f"Involves branching ({metrics.branching_references} references)"
            )

        reasoning_parts.append(f"Selected {strategy.value} strategy")
        reasoning_parts.append(f"Estimated cost: ${estimated_cost:.4f}")

        return " | ".join(reasoning_parts)


# Convenience functions for easy integration
def create_adaptive_router(budget_limit: Optional[float] = None) -> AdaptiveRouter:
    """Create a configured adaptive router."""
    return AdaptiveRouter(budget_limit=budget_limit)


def route_thought_adaptive(
    thought_data: ThoughtData,
    provider: str = "deepseek",
    budget_remaining: Optional[float] = None,
    budget_limit: Optional[float] = None,
) -> RoutingDecision:
    """Quick routing function for thought processing."""
    router = AdaptiveRouter(budget_limit=budget_limit)
    return router.route_thought(thought_data, provider, budget_remaining)
