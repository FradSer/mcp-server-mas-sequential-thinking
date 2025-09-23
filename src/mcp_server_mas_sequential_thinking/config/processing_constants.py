"""Processing Constants for MAS Sequential Thinking System.

Centralized constants to eliminate magic numbers and improve maintainability.
"""

from dataclasses import dataclass
from typing import Final

from mcp_server_mas_sequential_thinking.routing.complexity_types import ComplexityLevel


@dataclass(frozen=True)
class QualityThresholds:
    """Quality evaluation thresholds for output assessment."""

    MIN_CONTENT_LENGTH: Final[int] = 100
    """Minimum content length to avoid 'too short' quality issues."""

    INSUFFICIENT_DEPTH_THRESHOLD: Final[int] = 200
    """Threshold for identifying insufficient analysis depth."""

    MAX_ERROR_INDICATORS: Final[int] = 3
    """Maximum number of error indicators before quality concern."""


@dataclass(frozen=True)
class ComplexityThresholds:
    """Complexity scoring thresholds for routing decisions."""

    SIMPLE_MAX: Final[float] = 5.0
    """Maximum score for simple complexity level."""

    MODERATE_MAX: Final[float] = 15.0
    """Maximum score for moderate complexity level."""

    COMPLEX_MAX: Final[float] = 25.0
    """Maximum score for complex complexity level (above is highly complex)."""


@dataclass(frozen=True)
class RetryConfiguration:
    """Configuration for retry mechanisms and fallback logic."""

    SIMPLE_CONTENT_LENGTH_THRESHOLD: Final[int] = 50
    """Content length threshold for simple retry analysis."""

    MODERATE_CONTENT_LENGTH_THRESHOLD: Final[int] = 200
    """Content length threshold for moderate retry analysis."""

    MAX_KEYWORD_COUNT_SIMPLE: Final[int] = 1
    """Maximum complex keywords for simple classification."""

    MAX_KEYWORD_COUNT_MODERATE: Final[int] = 3
    """Maximum complex keywords for moderate classification."""


@dataclass(frozen=True)
class MultiThinkingConfiguration:
    """Configuration constants for Multi-Thinking processing."""

    SUITABILITY_MIN_INDICATORS: Final[int] = 2
    """Minimum indicators required for Multi-Thinking suitability."""

    MIN_COMPLEX_LENGTH: Final[int] = 50
    """Minimum content length for complex Multi-Thinking processing."""

    COST_REDUCTION_MAX: Final[float] = 85.0
    """Maximum cost reduction percentage cap."""


@dataclass(frozen=True)
class CostEstimation:
    """Cost estimation constants for different processing strategies."""

    SINGLE_AGENT_BASE_COST: Final[int] = 100
    """Base cost for single agent processing."""

    HYBRID_TEAM_BASE_COST: Final[int] = 300
    """Base cost for hybrid team processing."""

    MULTI_AGENT_BASE_COST: Final[int] = 600
    """Base cost for multi-agent processing."""

    THINKING_PROCESSING_COST: Final[int] = 50
    """Cost per thinking direction in Multi-Thinking processing."""

    TIME_COST_MULTIPLIER: Final[float] = 0.1
    """Multiplier for time-based cost calculation."""


@dataclass(frozen=True)
class LoggingLimits:
    """Limits for logging output to prevent excessive verbosity."""

    MAX_INPUT_LOG_LENGTH: Final[int] = 100
    """Maximum length for logging input content."""

    MAX_OUTPUT_LOG_LENGTH: Final[int] = 200
    """Maximum length for logging output content."""

    MAX_ERROR_LOG_LENGTH: Final[int] = 500
    """Maximum length for logging error messages."""


@dataclass(frozen=True)
class ProcessingLimits:
    """Processing limits and timeouts."""

    MAX_PROCESSING_RETRIES: Final[int] = 3
    """Maximum number of processing retries before giving up."""

    DEFAULT_TIMEOUT_SECONDS: Final[int] = 300
    """Default timeout for processing operations in seconds."""

    MAX_CONTENT_PREVIEW_LENGTH: Final[int] = 500
    """Maximum length for content previews."""


# Complex keywords used in retry analysis
COMPLEX_KEYWORDS: Final[tuple[str, ...]] = (
    "analyze",
    "consider",
    "evaluate",
    "implications",
    "factors",
    "compare",
    "contrast",
    "examine",
    "assess",
    "investigate",
    "综合",
    "分析",
    "评估",
    "考虑",
    "研究",
    "比较",
    "检查",
    "调查",
)

# Error indicators for quality assessment
ERROR_INDICATORS: Final[tuple[str, ...]] = (
    "error",
    "failed",
    "exception",
    "timeout",
    "invalid",
    "错误",
    "失败",
    "异常",
    "超时",
    "无效",
)

# Multi-Thinking suitability indicators
MULTI_THINKING_INDICATORS: Final[tuple[str, ...]] = (
    # Creative/innovative thinking
    "creative",
    "innovative",
    "brainstorm",
    "idea",
    "solution",
    "alternative",
    # Decision making
    "decide",
    "choose",
    "should",
    "option",
    "which",
    "better",
    # Evaluation/assessment
    "evaluate",
    "assess",
    "pros",
    "cons",
    "advantages",
    "disadvantages",
    # Complex philosophical questions
    "meaning",
    "purpose",
    "philosophy",
    "ethical",
    "moral",
    "value",
    # Problem solving
    "problem",
    "challenge",
    "issue",
    "difficulty",
    "obstacle",
    # Chinese equivalents
    "创新",
    "创意",
    "解决",
    "选择",
    "评估",
    "哲学",
    "意义",
    "价值",
    "问题",
)


def get_complexity_level_name(score: float) -> str:
    """Get human-readable complexity level name from score."""
    if score < ComplexityThresholds.SIMPLE_MAX:
        return ComplexityLevel.SIMPLE.value
    if score < ComplexityThresholds.MODERATE_MAX:
        return ComplexityLevel.MODERATE.value
    if score < ComplexityThresholds.COMPLEX_MAX:
        return ComplexityLevel.COMPLEX.value
    return ComplexityLevel.HIGHLY_COMPLEX.value


def is_content_sufficient_quality(content: str) -> bool:
    """Check if content meets basic quality requirements."""
    if len(content.strip()) < QualityThresholds.MIN_CONTENT_LENGTH:
        return False

    error_count = sum(
        1 for indicator in ERROR_INDICATORS if indicator.lower() in content.lower()
    )
    return error_count <= QualityThresholds.MAX_ERROR_INDICATORS


def count_complex_keywords(text: str) -> int:
    """Count complex keywords in text for retry analysis."""
    text_lower = text.lower()
    return sum(1 for keyword in COMPLEX_KEYWORDS if keyword in text_lower)


def is_suitable_for_multi_thinking(text: str) -> bool:
    """Quick heuristic to determine if Multi-Thinking would be beneficial."""
    text_lower = text.lower()
    indicator_count = sum(
        1 for indicator in MULTI_THINKING_INDICATORS if indicator in text_lower
    )

    has_questions = "?" in text or "？" in text
    is_complex_length = len(text) > MultiThinkingConfiguration.MIN_COMPLEX_LENGTH
    has_multiple_indicators = (
        indicator_count >= MultiThinkingConfiguration.SUITABILITY_MIN_INDICATORS
    )

    return has_multiple_indicators or (has_questions and is_complex_length)
