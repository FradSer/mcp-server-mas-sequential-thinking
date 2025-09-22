"""Smart logging system with minimal verbosity and maximum insight."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from .models import ThoughtData

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Smart log levels for different types of information."""
    CRITICAL_ONLY = "critical"  # Only errors and critical performance issues
    PERFORMANCE = "performance"  # Include performance metrics
    ROUTING = "routing"  # Include routing decisions
    DEBUG = "debug"  # All details


@dataclass
class ProcessingSnapshot:
    """Lightweight snapshot of processing state."""
    thought_id: int
    complexity_score: float
    strategy_name: str
    processing_time: float
    token_estimate: tuple[int, int]
    cost_estimate: float
    efficiency_score: float

    @property
    def is_slow(self) -> bool:
        """Check if processing is unacceptably slow."""
        return self.processing_time > 60.0  # More than 1 minute

    @property
    def is_expensive(self) -> bool:
        """Check if processing is unexpectedly expensive."""
        return self.cost_estimate > 0.01  # More than 1 cent

    @property
    def is_inefficient(self) -> bool:
        """Check if processing efficiency is poor."""
        return self.efficiency_score < 0.7


class SmartLogger:
    """Intelligent logging that adapts verbosity based on issues detected."""

    def __init__(self, log_level: LogLevel = LogLevel.PERFORMANCE):
        self.log_level = log_level
        self.session_snapshots: list[ProcessingSnapshot] = []

    def log_thought_start(self, thought_data: ThoughtData) -> None:
        """Log thought processing start with minimal noise."""
        if self.log_level in [LogLevel.ROUTING, LogLevel.DEBUG]:
            logger.info(f"Processing thought #{thought_data.thoughtNumber}: {thought_data.thought[:60]}...")

    def log_routing_decision(
        self,
        complexity_score: float,
        strategy_name: str,
        estimated_time: float,
        reasoning: Optional[str] = None
    ) -> None:
        """Log routing decision with smart verbosity."""
        # Always log if it's a potentially expensive decision
        if estimated_time > 120 or complexity_score > 30:
            logger.warning(
                f"🚨 Expensive routing: {strategy_name} (complexity: {complexity_score:.1f}, "
                f"est. time: {estimated_time:.0f}s)"
            )
            if reasoning and self.log_level == LogLevel.DEBUG:
                logger.info(f"Routing reasoning: {reasoning}")

        elif self.log_level in [LogLevel.ROUTING, LogLevel.DEBUG]:
            logger.info(f"Route: {strategy_name} (complexity: {complexity_score:.1f})")

    def log_processing_complete(self, snapshot: ProcessingSnapshot) -> None:
        """Log completion with adaptive verbosity based on performance."""
        self.session_snapshots.append(snapshot)

        # Always log performance issues
        issues = []
        if snapshot.is_slow:
            issues.append(f"SLOW ({snapshot.processing_time:.1f}s)")
        if snapshot.is_expensive:
            issues.append(f"EXPENSIVE (${snapshot.cost_estimate:.4f})")
        if snapshot.is_inefficient:
            issues.append(f"INEFFICIENT ({snapshot.efficiency_score:.2f})")

        if issues:
            logger.warning(
                f"⚠️ Thought #{snapshot.thought_id} completed with issues: {', '.join(issues)}"
            )
        elif self.log_level in [LogLevel.PERFORMANCE, LogLevel.ROUTING, LogLevel.DEBUG]:
            logger.info(
                f"✅ Thought #{snapshot.thought_id} completed "
                f"({snapshot.processing_time:.1f}s, eff: {snapshot.efficiency_score:.2f})"
            )

    def log_session_summary(self) -> None:
        """Log session summary with key insights."""
        if not self.session_snapshots:
            return

        total_time = sum(s.processing_time for s in self.session_snapshots)
        avg_efficiency = sum(s.efficiency_score for s in self.session_snapshots) / len(self.session_snapshots)
        slow_thoughts = [s for s in self.session_snapshots if s.is_slow]
        expensive_thoughts = [s for s in self.session_snapshots if s.is_expensive]

        logger.info(f"📊 Session Summary: {len(self.session_snapshots)} thoughts, "
                   f"{total_time:.1f}s total, {avg_efficiency:.2f} avg efficiency")

        if slow_thoughts:
            logger.warning(f"🐌 {len(slow_thoughts)} slow thoughts detected")

        if expensive_thoughts:
            logger.warning(f"💰 {len(expensive_thoughts)} expensive thoughts detected")

    def log_response_quality(self, content: str, thought_number: int) -> None:
        """Log response quality issues."""
        content_length = len(content)

        # Check for academic over-complexity
        academic_indicators = content.count("$$") + content.count("\\(") + content.count("###")
        if academic_indicators > 5:
            logger.warning(f"🎓 Thought #{thought_number}: Overly academic response detected")

        # Check for reasonable length
        if content_length > 2000:
            logger.warning(f"📝 Thought #{thought_number}: Very long response ({content_length} chars)")
        elif self.log_level == LogLevel.DEBUG:
            logger.debug(f"Response length: {content_length} chars")

    def force_debug_next(self) -> None:
        """Force debug level for next operations (for troubleshooting)."""
        self.log_level = LogLevel.DEBUG


class PerformanceMonitor:
    """Monitor and alert on performance degradation."""

    def __init__(self):
        self.baseline_efficiency = 0.8
        self.baseline_time_per_thought = 30.0
        self.recent_snapshots: list[ProcessingSnapshot] = []
        self.max_recent_count = 10

    def record_snapshot(self, snapshot: ProcessingSnapshot) -> None:
        """Record a processing snapshot and check for degradation."""
        self.recent_snapshots.append(snapshot)

        # Keep only recent snapshots
        if len(self.recent_snapshots) > self.max_recent_count:
            self.recent_snapshots.pop(0)

        self._check_performance_degradation()

    def _check_performance_degradation(self) -> None:
        """Check if performance is degrading."""
        if len(self.recent_snapshots) < 3:
            return

        recent_efficiency = sum(s.efficiency_score for s in self.recent_snapshots[-3:]) / 3
        recent_avg_time = sum(s.processing_time for s in self.recent_snapshots[-3:]) / 3

        if recent_efficiency < self.baseline_efficiency * 0.7:
            logger.warning(f"📉 Performance degradation detected: efficiency dropped to {recent_efficiency:.2f}")

        if recent_avg_time > self.baseline_time_per_thought * 2:
            logger.warning(f"🐌 Processing time increased significantly: {recent_avg_time:.1f}s avg")

    def get_performance_recommendation(self) -> Optional[str]:
        """Get recommendation for performance improvement."""
        if not self.recent_snapshots:
            return None

        slow_count = sum(1 for s in self.recent_snapshots if s.is_slow)
        expensive_count = sum(1 for s in self.recent_snapshots if s.is_expensive)

        if slow_count > len(self.recent_snapshots) * 0.5:
            return "Consider reducing complexity thresholds or simplifying routing logic"

        if expensive_count > len(self.recent_snapshots) * 0.3:
            return "Consider using more cost-effective strategies for moderate complexity tasks"

        return None


# Global instances
smart_logger = SmartLogger()
performance_monitor = PerformanceMonitor()


def configure_smart_logging(level: LogLevel) -> None:
    """Configure smart logging level."""
    smart_logger.log_level = level


def log_thought_processing(
    thought_data: ThoughtData,
    complexity_score: float,
    strategy_name: str,
    processing_time: float,
    efficiency_score: float,
    content: str,
    token_estimate: tuple[int, int] = (0, 0),
    cost_estimate: float = 0.0,
    reasoning: Optional[str] = None
) -> None:
    """Unified logging function for thought processing."""
    # Log start
    smart_logger.log_thought_start(thought_data)

    # Log routing
    estimated_time = 60 if "全面探索" in strategy_name else 30
    smart_logger.log_routing_decision(complexity_score, strategy_name, estimated_time, reasoning)

    # Create snapshot
    snapshot = ProcessingSnapshot(
        thought_id=thought_data.thoughtNumber,
        complexity_score=complexity_score,
        strategy_name=strategy_name,
        processing_time=processing_time,
        token_estimate=token_estimate,
        cost_estimate=cost_estimate,
        efficiency_score=efficiency_score
    )

    # Log completion
    smart_logger.log_processing_complete(snapshot)

    # Log response quality
    smart_logger.log_response_quality(content, thought_data.thoughtNumber)

    # Monitor performance
    performance_monitor.record_snapshot(snapshot)


def log_session_end() -> None:
    """Log session end summary."""
    smart_logger.log_session_summary()

    recommendation = performance_monitor.get_performance_recommendation()
    if recommendation:
        logger.info(f"💡 Performance tip: {recommendation}")