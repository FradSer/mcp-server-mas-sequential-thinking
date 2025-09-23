"""Core complexity analysis types and enums.

This file contains the essential types needed for complexity analysis,
extracted from the old adaptive_routing.py to support AI-first architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData


class ProcessingStrategy(Enum):
    """Processing strategy types."""

    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    HYBRID = "hybrid"


class ComplexityLevel(Enum):
    """Complexity levels for thought processing."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"


@dataclass
class ComplexityMetrics:
    """Metrics for thought complexity analysis."""

    # Core metrics (used by both AI and fallback analyzers)
    complexity_score: float  # Primary score (0-100)

    # Detailed breakdown (optional, for debugging/analysis)
    word_count: int = 0
    sentence_count: int = 0
    question_count: int = 0
    technical_terms: int = 0
    branching_references: int = 0
    research_indicators: int = 0
    analysis_depth: int = 0
    philosophical_depth_boost: int = 0

    # Analysis metadata
    analyzer_type: str = "unknown"  # "ai" or "basic"
    reasoning: str = ""  # Why this score was assigned


@dataclass
class RoutingDecision:
    """Decision result from routing analysis."""

    strategy: ProcessingStrategy
    complexity_level: ComplexityLevel
    complexity_score: float
    reasoning: str
    estimated_token_usage: tuple[int, int]  # (min, max)
    estimated_cost: float
    specialist_recommendations: list[str] | None = None

    def __post_init__(self):
        if self.specialist_recommendations is None:
            self.specialist_recommendations = []


class ComplexityAnalyzer(ABC):
    """Abstract base class for complexity analysis."""

    @abstractmethod
    async def analyze(self, thought_data: "ThoughtData") -> ComplexityMetrics:
        """Analyze thought complexity and return metrics."""
