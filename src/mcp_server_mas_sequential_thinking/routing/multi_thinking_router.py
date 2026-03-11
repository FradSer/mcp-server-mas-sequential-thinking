"""Multi-thinking router with a fixed full-exploration strategy.

This router intentionally removes legacy single/double/triple routing branches.
All thoughts are processed through the same full multi-thinking sequence.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp_server_mas_sequential_thinking.processors.multi_thinking_core import (
    ProcessingDepth,
    ThinkingDirection,
)

from .ai_complexity_analyzer import AIComplexityAnalyzer
from .complexity_types import ComplexityMetrics

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)

FULL_EXPLORATION_STRATEGY_KEY = "full_exploration"
FULL_EXPLORATION_SEQUENCE: tuple[ThinkingDirection, ...] = (
    ThinkingDirection.SYNTHESIS,
    ThinkingDirection.FACTUAL,
    ThinkingDirection.EMOTIONAL,
    ThinkingDirection.OPTIMISTIC,
    ThinkingDirection.CRITICAL,
    ThinkingDirection.CREATIVE,
    ThinkingDirection.METACOGNITIVE,
    ThinkingDirection.SYNTHESIS,
)


@dataclass(frozen=True)
class ThinkingSequenceStrategy:
    """Single fixed strategy definition."""

    key: str
    name: str
    complexity: ProcessingDepth
    thinking_sequence: tuple[ThinkingDirection, ...]
    estimated_time_seconds: int
    description: str


FULL_EXPLORATION_STRATEGY = ThinkingSequenceStrategy(
    key=FULL_EXPLORATION_STRATEGY_KEY,
    name="Full Exploration Sequence",
    complexity=ProcessingDepth.FULL,
    thinking_sequence=FULL_EXPLORATION_SEQUENCE,
    estimated_time_seconds=780,
    description="Default full multi-thinking analysis sequence",
)


@dataclass
class RoutingDecision:
    """Routing decision result."""

    strategy: ThinkingSequenceStrategy
    reasoning: str
    complexity_metrics: ComplexityMetrics
    estimated_cost_reduction: float
    problem_type: str
    thinking_modes_needed: list[str]


class MultiThinkingIntelligentRouter:
    """Router that always returns the full exploration strategy."""

    def __init__(self, complexity_analyzer: AIComplexityAnalyzer | None = None) -> None:
        self.complexity_analyzer = complexity_analyzer or AIComplexityAnalyzer()

    async def route_thought(
        self,
        thought_data: "ThoughtData",
        complexity_metrics: ComplexityMetrics | None = None,
    ) -> RoutingDecision:
        """Route thought to the fixed full-exploration strategy."""
        logger.info("Fixed multi-thinking routing started")
        if logger.isEnabledFor(logging.INFO):
            logger.info("Input preview: %s", thought_data.thought[:100])

        if complexity_metrics is None:
            complexity_metrics = await self.complexity_analyzer.analyze(thought_data)

        problem_type = complexity_metrics.primary_problem_type
        thinking_modes_needed = complexity_metrics.thinking_modes_needed or ["SYNTHESIS"]
        cost_reduction = self._estimate_cost_reduction(complexity_metrics.complexity_score)

        reasoning = (
            "Fixed routing strategy: full_exploration is mandatory for all thoughts."
        )

        return RoutingDecision(
            strategy=FULL_EXPLORATION_STRATEGY,
            reasoning=reasoning,
            complexity_metrics=complexity_metrics,
            estimated_cost_reduction=cost_reduction,
            problem_type=problem_type,
            thinking_modes_needed=thinking_modes_needed,
        )

    async def route_thought_with_strategy(
        self,
        thought_data: "ThoughtData",
        strategy_name: str,
        complexity_metrics: ComplexityMetrics | None = None,
    ) -> RoutingDecision:
        """Route thought with explicit strategy validation.

        Only full_exploration is accepted in the simplified routing model.
        """
        if strategy_name != FULL_EXPLORATION_STRATEGY_KEY:
            raise ValueError(
                "Unsupported strategy. Only 'full_exploration' is allowed."
            )
        return await self.route_thought(thought_data, complexity_metrics=complexity_metrics)

    def _estimate_cost_reduction(self, complexity_score: float) -> float:
        """Estimate cost reduction using the previous baseline model."""
        if complexity_score < 5:
            original_cost = 100
        elif complexity_score < 15:
            original_cost = 300
        else:
            original_cost = 600

        new_cost = len(FULL_EXPLORATION_STRATEGY.thinking_sequence) * 50 + (
            FULL_EXPLORATION_STRATEGY.estimated_time_seconds * 0.1
        )
        if original_cost <= 0:
            return 0.0
        reduction = max(0.0, (original_cost - new_cost) / original_cost * 100.0)
        return min(reduction, 85.0)


def create_multi_thinking_router(
    complexity_analyzer: AIComplexityAnalyzer | None = None,
) -> MultiThinkingIntelligentRouter:
    """Create a fixed full-exploration router."""
    return MultiThinkingIntelligentRouter(complexity_analyzer)


async def route_thought_to_thinking(thought_data: "ThoughtData") -> RoutingDecision:
    """Route thought using the fixed full-exploration strategy."""
    router = MultiThinkingIntelligentRouter()
    return await router.route_thought(thought_data)
