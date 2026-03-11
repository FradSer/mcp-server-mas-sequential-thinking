"""Unit tests for simplified multi-thinking router behavior."""

import pytest

from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.processors.multi_thinking_processor import (
    MultiThinkingSequentialProcessor,
)
from mcp_server_mas_sequential_thinking.routing.complexity_types import ComplexityMetrics


@pytest.fixture
def sample_thought() -> ThoughtData:
    """Return a minimal valid thought payload."""
    return ThoughtData(
        thought="How should we redesign the workflow?",
        thoughtNumber=1,
        totalThoughts=1,
        nextThoughtNeeded=False,
        isRevision=False,
        branchFromThought=None,
        branchId=None,
        needsMoreThoughts=False,
    )


@pytest.fixture
def sample_metrics() -> ComplexityMetrics:
    """Return deterministic complexity metrics for routing tests."""
    return ComplexityMetrics(
        complexity_score=12.0,
        primary_problem_type="FACTUAL",
        thinking_modes_needed=["FACTUAL", "SYNTHESIS"],
        analyzer_type="test",
        reasoning="test fixture",
    )


@pytest.mark.asyncio
async def test_route_thought_defaults_to_full_exploration(
    sample_thought: ThoughtData,
    sample_metrics: ComplexityMetrics,
):
    """Default routing should always return the full exploration sequence."""
    router = MultiThinkingSequentialProcessor().router

    decision = await router.route_thought(
        sample_thought, complexity_metrics=sample_metrics
    )

    assert decision.strategy.name == "Full Exploration Sequence"
    assert [direction.value for direction in decision.strategy.thinking_sequence] == [
        "synthesis",
        "factual",
        "emotional",
        "optimistic",
        "critical",
        "creative",
        "metacognitive",
        "synthesis",
    ]


@pytest.mark.asyncio
async def test_route_thought_with_strategy_rejects_legacy_strategies(
    sample_thought: ThoughtData,
    sample_metrics: ComplexityMetrics,
):
    """Legacy forced strategies should fail instead of silently falling back."""
    router = MultiThinkingSequentialProcessor().router

    with pytest.raises(ValueError, match="full_exploration"):
        await router.route_thought_with_strategy(
            sample_thought,
            "single_factual",
            complexity_metrics=sample_metrics,
        )


@pytest.mark.asyncio
async def test_route_thought_with_strategy_accepts_only_full_exploration(
    sample_thought: ThoughtData,
    sample_metrics: ComplexityMetrics,
):
    """Full exploration is the only valid forced strategy."""
    router = MultiThinkingSequentialProcessor().router

    decision = await router.route_thought_with_strategy(
        sample_thought,
        "full_exploration",
        complexity_metrics=sample_metrics,
    )

    assert decision.strategy.name == "Full Exploration Sequence"
