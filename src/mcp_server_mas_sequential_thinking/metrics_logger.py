"""Metrics logging utilities for consistent performance tracking."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .constants import FieldLengthLimits, PerformanceMetrics
from .models import ThoughtData

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log levels for different types of metrics."""

    INFO = "info"
    DEBUG = "debug"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class MetricsConfig:
    """Configuration for metrics logging."""

    separator_length: int = PerformanceMetrics.SEPARATOR_LENGTH
    log_level: LogLevel = LogLevel.INFO
    include_separators: bool = True


class MetricsLogger:
    """Centralized metrics logging with consistent formatting."""

    def __init__(self, config: MetricsConfig = None):
        """Initialize metrics logger with configuration."""
        self.config = config or MetricsConfig()

    def log_section_header(
        self, title: str, separator_length: int | None = None
    ) -> None:
        """Log a section header with consistent formatting."""
        length = separator_length or self.config.separator_length
        self._log_with_level(title, self.config.log_level)

        if self.config.include_separators:
            self.log_separator(length)

    def log_metrics_block(self, title: str, metrics: dict[str, Any]) -> None:
        """Log a block of metrics with consistent formatting."""
        self._log_with_level(title, self.config.log_level)

        for key, value in metrics.items():
            formatted_value = self._format_metric_value(value)
            self._log_with_level(f"  {key}: {formatted_value}", self.config.log_level)

    def log_separator(self, length: int | None = None) -> None:
        """Log a separator line."""
        sep_length = length or self.config.separator_length
        separator = f"  {'=' * sep_length}"
        self._log_with_level(separator, self.config.log_level)

    def log_completion_summary(
        self,
        thought_data: ThoughtData,
        strategy: str,
        specialists_count: int,
        processing_time: float,
        total_time: float,
        confidence: float,
        final_response: str,
    ) -> None:
        """Log comprehensive completion summary."""
        # Completion info
        completion_info = {
            f"Thought #{thought_data.thoughtNumber}": "completed",
            "Strategy": strategy,
            "Specialists": specialists_count,
            "Processing time": f"{processing_time:.3f}s",
            "Total time": f"{total_time:.3f}s",
            "Confidence": confidence,
        }
        self.log_metrics_block("💫 COMPLETION SUMMARY:", completion_info)

        # Performance metrics
        performance_metrics = self._calculate_performance_metrics(
            processing_time, strategy, final_response
        )
        self.log_metrics_block("📊 PERFORMANCE METRICS:", performance_metrics)

        # Final summary
        final_summary = {
            f"Thought #{thought_data.thoughtNumber}": "processed successfully",
            "Strategy used": strategy,
            "Processing time": f"{processing_time:.3f}s",
            "Total time": f"{total_time:.3f}s",
            "Response length": f"{len(final_response)} chars",
        }
        self.log_metrics_block("🎯 PROCESSING COMPLETE:", final_summary)
        self.log_separator(FieldLengthLimits.SEPARATOR_LENGTH)

    def log_team_details(self, team_info: dict[str, Any]) -> None:
        """Log team processing details."""
        team_details = {
            "Team": f"{team_info.get('name', 'unknown')} ({team_info.get('member_count', 0)} agents)",
            "Leader": f"{team_info.get('leader_class', 'unknown')} (model: {team_info.get('leader_model', 'unknown')})",
            "Members": team_info.get("member_names", "unknown"),
        }
        self.log_metrics_block("🏢 MULTI-AGENT TEAM CALL:", team_details)

    def log_input_details(self, input_prompt: str, max_preview: int = 200) -> None:
        """Log input processing details."""
        preview = input_prompt[:max_preview]
        if len(input_prompt) > max_preview:
            preview += "..."

        input_info = {"Input length": f"{len(input_prompt)} chars", "Preview": preview}
        self.log_metrics_block("📥 INPUT DETAILS:", input_info)

    def log_output_details(self, response_content: str, processing_time: float) -> None:
        """Log output processing details."""
        output_info = {
            "Response length": f"{len(response_content)} chars",
            "Processing time": f"{processing_time:.3f}s",
            "Success": "✅",
        }
        self.log_metrics_block("📤 OUTPUT DETAILS:", output_info)

    def _calculate_performance_metrics(
        self, processing_time: float, strategy: str, final_response: str
    ) -> dict[str, Any]:
        """Calculate performance metrics for logging."""
        # Calculate efficiency score
        efficiency_score = self._calculate_efficiency_score(processing_time)

        # Calculate execution consistency
        execution_consistency = self._calculate_execution_consistency(bool(strategy))

        return {
            "Execution Consistency": execution_consistency,
            "Efficiency Score": efficiency_score,
            "Response Length": f"{len(final_response)} chars",
            "Strategy Executed": strategy,
        }

    def _calculate_efficiency_score(self, processing_time: float) -> float:
        """Calculate efficiency score based on processing time."""
        if processing_time < PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD:
            return PerformanceMetrics.PERFECT_EFFICIENCY_SCORE
        return max(
            PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
            PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / processing_time,
        )

    def _calculate_execution_consistency(self, success: bool) -> float:
        """Calculate execution consistency score."""
        return (
            PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY
            if success
            else PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY
        )

    def _format_metric_value(self, value: Any) -> str:
        """Format metric values for consistent display."""
        if isinstance(value, float):
            # Format floats to 3 decimal places
            if value == int(value):
                return str(int(value))
            return f"{value:.3f}"
        if isinstance(value, bool):
            return "✅" if value else "❌"
        if isinstance(value, (list, tuple)):
            if len(value) <= 3:
                return ", ".join(str(item) for item in value)
            return f"{', '.join(str(item) for item in value[:3])}... ({len(value)} total)"
        return str(value)

    def _log_with_level(self, message: str, level: LogLevel) -> None:
        """Log message with specified level."""
        if level == LogLevel.DEBUG:
            logger.debug(message)
        elif level == LogLevel.WARNING:
            logger.warning(message)
        elif level == LogLevel.ERROR:
            logger.error(message)
        else:
            logger.info(message)


class PerformanceTracker:
    """Tracks and reports performance metrics over time."""

    def __init__(self):
        """Initialize performance tracker."""
        self.metrics_logger = MetricsLogger()
        self.processing_times = []
        self.success_count = 0
        self.total_attempts = 0

    def record_processing(self, processing_time: float, success: bool) -> None:
        """Record processing metrics."""
        self.processing_times.append(processing_time)
        self.total_attempts += 1
        if success:
            self.success_count += 1

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.processing_times:
            return {"status": "no_data"}

        avg_time = sum(self.processing_times) / len(self.processing_times)
        success_rate = (
            self.success_count / self.total_attempts if self.total_attempts > 0 else 0
        )

        return {
            "average_processing_time": f"{avg_time:.3f}s",
            "success_rate": f"{success_rate:.1%}",
            "total_attempts": self.total_attempts,
            "min_time": f"{min(self.processing_times):.3f}s",
            "max_time": f"{max(self.processing_times):.3f}s",
        }

    def log_performance_summary(self) -> None:
        """Log current performance summary."""
        summary = self.get_performance_summary()
        if summary.get("status") != "no_data":
            self.metrics_logger.log_metrics_block("📈 PERFORMANCE SUMMARY:", summary)
