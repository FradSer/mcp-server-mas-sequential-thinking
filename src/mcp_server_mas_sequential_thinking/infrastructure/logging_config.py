"""Complete logging system - unified configuration and functionality.

This module provides the entire logging infrastructure:
- Environment variable configuration
- Smart performance-aware logging
- Metrics logging with formatting
- Security filtering
- JSON and text formatters
"""

import json
import logging
import logging.handlers
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from re import Pattern
from typing import Any, Optional


class LogLevel(Enum):
    """Standard log levels with environment variable mapping."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

    @classmethod
    def from_string(cls, level_str: str) -> "LogLevel":
        """Convert string to LogLevel enum."""
        try:
            return cls(level_str.upper())
        except ValueError:
            return cls.INFO  # Default fallback


class SmartLogLevel(Enum):
    """Smart log levels for performance-aware logging."""

    CRITICAL_ONLY = "critical"      # Only errors and critical performance issues
    PERFORMANCE = "performance"     # Include performance metrics
    ROUTING = "routing"            # Include routing decisions
    DEBUG = "debug"                # All details

    @classmethod
    def from_string(cls, level_str: str) -> "SmartLogLevel":
        """Convert string to SmartLogLevel enum."""
        try:
            return cls(level_str.lower())
        except ValueError:
            return cls.PERFORMANCE  # Default fallback


class LogFormat(Enum):
    """Supported log formats."""

    TEXT = "text"
    JSON = "json"

    @classmethod
    def from_string(cls, format_str: str) -> "LogFormat":
        """Convert string to LogFormat enum."""
        try:
            return cls(format_str.lower())
        except ValueError:
            return cls.TEXT  # Default fallback


class LogTarget(Enum):
    """Supported log targets."""

    CONSOLE = "console"
    FILE = "file"

    @classmethod
    def from_string_list(cls, targets_str: str) -> list["LogTarget"]:
        """Convert comma-separated string to list of LogTarget enums."""
        if not targets_str:
            return [cls.FILE, cls.CONSOLE]  # Default

        targets = []
        for target in targets_str.split(","):
            target = target.strip().lower()
            try:
                targets.append(cls(target))
            except ValueError:
                continue  # Skip invalid targets

        return targets or [cls.FILE, cls.CONSOLE]  # Fallback if all invalid


class SensitiveDataFilter(logging.Filter):
    """Enhanced filter to redact sensitive information from log messages."""

    # Pre-compiled patterns with optimized regex for better performance
    _SENSITIVE_PATTERNS: list[tuple[Pattern[str], str]] = [
        (
            re.compile(
                r'(API_KEY|TOKEN|SECRET|PASSWORD|APIKEY)["\s]*[:=]["\s]*[^\s"\'<>&]{8,}',
                re.IGNORECASE,
            ),
            r"\1=***REDACTED***",
        ),
        (
            re.compile(r"(Bearer\s+)[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE),
            r"\1***REDACTED***",
        ),
        (
            re.compile(r"(ghp_|github_pat_|gho_|ghu_)[A-Za-z0-9]{36,}", re.IGNORECASE),
            r"***REDACTED_GITHUB_TOKEN***",
        ),
        (re.compile(r"sk-[A-Za-z0-9]{48,}"), r"***REDACTED_OPENAI_KEY***"),
        (re.compile(r"(deepseek|groq|openrouter)_api_key[\":\s=]*[^\s\"'<>&]{8,}", re.IGNORECASE),
         r"***REDACTED_API_KEY***"),
        # Database URLs
        (re.compile(r"(postgresql|mysql|sqlite)://[^\s\"'<>&]+", re.IGNORECASE),
         r"***REDACTED_DATABASE_URL***"),
        # Generic secrets in environment variables
        (re.compile(r"([A-Z_]+_SECRET|[A-Z_]+_TOKEN)[\":\s=]*[^\s\"'<>&]{8,}", re.IGNORECASE),
         r"***REDACTED_SECRET***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter sensitive data from log record with optimized processing."""
        # Process message if present
        if hasattr(record, "msg") and record.msg:
            record.msg = self._redact_sensitive_data(str(record.msg))

        # Process arguments efficiently
        if hasattr(record, "args") and record.args:
            record.args = tuple(
                self._redact_sensitive_data(str(arg)) for arg in record.args
            )

        return True

    def _redact_sensitive_data(self, text: str) -> str:
        """Redact sensitive data from text with early exit optimization."""
        # Early exit for empty or very short strings
        if not text or len(text) < 8:
            return text

        # Apply patterns with short-circuit optimization
        for pattern, replacement in self._SENSITIVE_PATTERNS:
            if pattern.search(text):  # Only substitute if pattern found
                text = pattern.sub(replacement, text)

        return text


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ("name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "getMessage", "exc_info",
                          "exc_text", "stack_info"):
                log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class LoggingConfig:
    """Centralized logging configuration from environment variables."""

    def __init__(self) -> None:
        """Initialize logging configuration from environment."""
        self.log_level = LogLevel.from_string(os.getenv("LOG_LEVEL", "INFO"))
        self.smart_log_level = SmartLogLevel.from_string(os.getenv("SMART_LOG_LEVEL", "performance"))
        self.log_format = LogFormat.from_string(os.getenv("LOG_FORMAT", "text"))
        self.log_targets = LogTarget.from_string_list(os.getenv("LOG_TARGETS", "file,console"))

        # File logging configuration
        self.log_file_max_size = self._parse_file_size(os.getenv("LOG_FILE_MAX_SIZE", "10MB"))
        self.log_file_backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))

        # Performance configuration
        self.log_sampling_rate = float(os.getenv("LOG_SAMPLING_RATE", "1.0"))

        # Log directory
        self.log_dir = Path.home() / ".mas_sequential_thinking" / "logs"

        # Performance monitoring flags
        self.log_performance_issues = os.getenv("LOG_PERFORMANCE_ISSUES", "true").lower() == "true"
        self.log_response_quality = os.getenv("LOG_RESPONSE_QUALITY", "true").lower() == "true"

    def _parse_file_size(self, size_str: str) -> int:
        """Parse file size string like '10MB' to bytes."""
        size_str = size_str.upper().strip()

        # Extract number and unit
        if size_str.endswith("GB"):
            return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
        if size_str.endswith("MB"):
            return int(float(size_str[:-2]) * 1024 * 1024)
        if size_str.endswith("KB"):
            return int(float(size_str[:-2]) * 1024)
        if size_str.endswith("B"):
            return int(size_str[:-1])
        # Assume MB if no unit
        try:
            return int(float(size_str) * 1024 * 1024)
        except ValueError:
            return 10 * 1024 * 1024  # 10MB default

    def create_formatter(self) -> logging.Formatter:
        """Create appropriate formatter based on configuration."""
        if self.log_format == LogFormat.JSON:
            return JSONFormatter()
        return logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    def create_handlers(self) -> list[logging.Handler]:
        """Create logging handlers based on configuration."""
        handlers: list[logging.Handler] = []

        # Create singleton sensitive data filter for reuse
        sensitive_filter = SensitiveDataFilter()
        formatter = self.create_formatter()

        if LogTarget.FILE in self.log_targets:
            # Ensure log directory exists
            self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_dir / "mas_sequential_thinking.log",
                maxBytes=self.log_file_max_size,
                backupCount=self.log_file_backup_count,
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(sensitive_filter)
            handlers.append(file_handler)

        if LogTarget.CONSOLE in self.log_targets:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(formatter)
            console_handler.addFilter(sensitive_filter)
            handlers.append(console_handler)

        return handlers

    def get_logging_level(self) -> int:
        """Get Python logging level from LogLevel enum."""
        level_mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }
        return level_mapping[self.log_level]


# ===== SMART LOGGING FUNCTIONALITY =====

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

    def __init__(self, log_level: SmartLogLevel = SmartLogLevel.PERFORMANCE) -> None:
        self.log_level = log_level
        self.session_snapshots: list[ProcessingSnapshot] = []
        self.logger: Optional[logging.Logger] = None  # Will be initialized later

    def _ensure_logger(self) -> None:
        """Ensure logger is initialized."""
        if self.logger is None:
            self.logger = LoggerFactory.get_logger(__name__)

    def _get_logger(self) -> logging.Logger:
        """Get logger, ensuring it's initialized."""
        self._ensure_logger()
        assert self.logger is not None
        return self.logger

    def log_thought_start(self, thought_data) -> None:
        """Log thought processing start with minimal noise."""
        if self.log_level in [SmartLogLevel.ROUTING, SmartLogLevel.DEBUG]:
            self._ensure_logger()
            self._get_logger().info(f"Processing thought #{thought_data.thoughtNumber}: {thought_data.thought[:60]}...")

    def log_routing_decision(
        self,
        complexity_score: float,
        strategy_name: str,
        estimated_time: float,
        reasoning: str | None = None
    ) -> None:
        """Log routing decision with smart verbosity."""
        self._ensure_logger()
        # Always log if it's a potentially expensive decision
        if estimated_time > 120 or complexity_score > 30:
            self._get_logger().warning(
                f"🚨 Expensive routing: {strategy_name} (complexity: {complexity_score:.1f}, "
                f"est. time: {estimated_time:.0f}s)"
            )
            if reasoning and self.log_level == SmartLogLevel.DEBUG:
                self._get_logger().info(f"Routing reasoning: {reasoning}")

        elif self.log_level in [SmartLogLevel.ROUTING, SmartLogLevel.DEBUG]:
            self._get_logger().info(f"Route: {strategy_name} (complexity: {complexity_score:.1f})")

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
            self._get_logger().warning(
                f"⚠️ Thought #{snapshot.thought_id} completed with issues: {', '.join(issues)}"
            )
        elif self.log_level in [SmartLogLevel.PERFORMANCE, SmartLogLevel.ROUTING, SmartLogLevel.DEBUG]:
            self._get_logger().info(
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

        self._get_logger().info(f"📊 Session Summary: {len(self.session_snapshots)} thoughts, "
                       f"{total_time:.1f}s total, {avg_efficiency:.2f} avg efficiency")

        if slow_thoughts:
            self._get_logger().warning(f"🐌 {len(slow_thoughts)} slow thoughts detected")

        if expensive_thoughts:
            self._get_logger().warning(f"💰 {len(expensive_thoughts)} expensive thoughts detected")

    def log_response_quality(self, content: str, thought_number: int) -> None:
        """Log response quality issues."""
        content_length = len(content)

        # Check for academic over-complexity
        academic_indicators = content.count("$$") + content.count("\\(") + content.count("###")
        if academic_indicators > 5:
            self._get_logger().warning(f"🎓 Thought #{thought_number}: Overly academic response detected")

        # Check for reasonable length
        if content_length > 2000:
            self._get_logger().warning(f"📝 Thought #{thought_number}: Very long response ({content_length} chars)")
        elif self.log_level == SmartLogLevel.DEBUG:
            self._get_logger().debug(f"Response length: {content_length} chars")

    def force_debug_next(self) -> None:
        """Force debug level for next operations (for troubleshooting)."""
        self.log_level = SmartLogLevel.DEBUG


class PerformanceMonitor:
    """Monitor and alert on performance degradation."""

    def __init__(self) -> None:
        self.baseline_efficiency = 0.8
        self.baseline_time_per_thought = 30.0
        self.recent_snapshots: list[ProcessingSnapshot] = []
        self.max_recent_count = 10
        self.logger: Optional[logging.Logger] = None  # Will be initialized later

    def _ensure_logger(self) -> None:
        """Ensure logger is initialized."""
        if self.logger is None:
            self.logger = LoggerFactory.get_logger(__name__)

    def _get_logger(self) -> logging.Logger:
        """Get logger, ensuring it's initialized."""
        self._ensure_logger()
        assert self.logger is not None
        return self.logger

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
            self._get_logger().warning(f"📉 Performance degradation detected: efficiency dropped to {recent_efficiency:.2f}")

        if recent_avg_time > self.baseline_time_per_thought * 2:
            self._get_logger().warning(f"🐌 Processing time increased significantly: {recent_avg_time:.1f}s avg")

    def get_performance_recommendation(self) -> str | None:
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


# ===== METRICS LOGGING FUNCTIONALITY =====

@dataclass
class MetricsConfig:
    """Configuration for metrics logging."""
    separator_length: int = 60
    log_level: LogLevel = LogLevel.INFO
    include_separators: bool = True


class MetricsLogger:
    """Centralized metrics logging with consistent formatting."""

    def __init__(self, config: MetricsConfig | None = None) -> None:
        self.config = config or MetricsConfig()
        self.logger: Optional[logging.Logger] = None  # Will be initialized later

    def _ensure_logger(self) -> None:
        """Ensure logger is initialized."""
        if self.logger is None:
            self.logger = LoggerFactory.get_logger(__name__)

    def _get_logger(self) -> logging.Logger:
        """Get logger, ensuring it's initialized."""
        self._ensure_logger()
        assert self.logger is not None
        return self.logger

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
        thought_data,
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
        self.log_separator(60)

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
        efficiency_threshold = 60.0  # seconds
        if processing_time < efficiency_threshold:
            return 1.0
        return max(0.1, efficiency_threshold / processing_time)

    def _calculate_execution_consistency(self, success: bool) -> float:
        """Calculate execution consistency score."""
        return 1.0 if success else 0.5

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
            self._get_logger().debug(message)
        elif level == LogLevel.WARNING:
            self._get_logger().warning(message)
        elif level == LogLevel.ERROR:
            self._get_logger().error(message)
        elif level == LogLevel.CRITICAL:
            self._get_logger().critical(message)
        else:
            self._get_logger().info(message)


class PerformanceTracker:
    """Tracks and reports performance metrics over time."""

    def __init__(self) -> None:
        self.metrics_logger = MetricsLogger()
        self.processing_times: list[float] = []
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


# ===== FACTORY AND MAIN INTERFACE =====

class LoggerFactory:
    """Factory for creating configured loggers."""

    _config: LoggingConfig | None = None
    _root_logger_configured: bool = False

    @classmethod
    def configure(cls, config: LoggingConfig | None = None) -> None:
        """Configure the logger factory with global settings."""
        cls._config = config or LoggingConfig()
        cls._setup_root_logger()

    @classmethod
    def _setup_root_logger(cls) -> None:
        """Setup the root logger for the application."""
        if cls._root_logger_configured or not cls._config:
            return

        # Configure root logger for the application
        root_logger = logging.getLogger("mas_sequential_thinking")
        root_logger.setLevel(cls._config.get_logging_level())

        # Remove existing handlers to prevent duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add new handlers
        for handler in cls._config.create_handlers():
            root_logger.addHandler(handler)

        cls._root_logger_configured = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a configured logger for the given name."""
        if not cls._config:
            cls.configure()

        # Unified naming strategy: use mas_sequential_thinking prefix
        if name.startswith("src.mcp_server_mas_sequential_thinking"):
            # Remove src.mcp_server prefix for cleaner logs
            logger_name = name.replace("src.mcp_server_mas_sequential_thinking", "mas_sequential_thinking")
        elif name.startswith("mcp_server_mas_sequential_thinking"):
            # Replace mcp_server prefix
            logger_name = name.replace("mcp_server_mas_sequential_thinking", "mas_sequential_thinking")
        elif name in {"__main__", "sequential_thinking"}:
            # Main application logger
            logger_name = "mas_sequential_thinking"
        else:
            # For other modules, use application prefix
            logger_name = f"mas_sequential_thinking.{name}"

        logger = logging.getLogger(logger_name)

        # Set up inheritance hierarchy
        if logger_name != "mas_sequential_thinking":
            logger.parent = logging.getLogger("mas_sequential_thinking")

        return logger

    @classmethod
    def get_smart_log_level(cls) -> SmartLogLevel:
        """Get the configured smart log level."""
        if not cls._config:
            cls.configure()
        assert cls._config is not None
        return cls._config.smart_log_level

    @classmethod
    def should_log_performance_issues(cls) -> bool:
        """Check if performance issues should be logged."""
        if not cls._config:
            cls.configure()
        assert cls._config is not None
        return cls._config.log_performance_issues

    @classmethod
    def should_log_response_quality(cls) -> bool:
        """Check if response quality should be logged."""
        if not cls._config:
            cls.configure()
        assert cls._config is not None
        return cls._config.log_response_quality

    @classmethod
    def reload_config(cls) -> None:
        """Reload configuration from environment variables."""
        cls._config = LoggingConfig()
        cls._root_logger_configured = False
        cls._setup_root_logger()


# ===== GLOBAL INSTANCES AND FUNCTIONS =====

# Global instances
smart_logger = SmartLogger()
performance_monitor = PerformanceMonitor()


def setup_logging(config: LoggingConfig | None = None) -> logging.Logger:
    """Setup logging with unified configuration and return root logger."""
    LoggerFactory.configure(config)
    return logging.getLogger("mas_sequential_thinking")


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger - primary interface for all modules."""
    return LoggerFactory.get_logger(name)


def get_smart_log_level() -> SmartLogLevel:
    """Get the current smart log level."""
    return LoggerFactory.get_smart_log_level()


def reload_logging_config() -> None:
    """Reload logging configuration from environment variables."""
    LoggerFactory.reload_config()


def configure_smart_logging(level: SmartLogLevel) -> None:
    """Configure smart logging level."""
    smart_logger.log_level = level


def log_thought_processing(
    thought_data,
    complexity_score: float,
    strategy_name: str,
    processing_time: float,
    efficiency_score: float,
    content: str,
    token_estimate: tuple[int, int] = (0, 0),
    cost_estimate: float = 0.0,
    reasoning: str | None = None
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
        logger = get_logger(__name__)
        logger.info(f"💡 Performance tip: {recommendation}")


# Export all public interfaces
__all__ = [
    "JSONFormatter",
    "LogFormat",
    # Core configuration
    "LogLevel",
    "LogTarget",
    "LoggerFactory",
    "LoggingConfig",
    "MetricsConfig",
    # Metrics logging
    "MetricsLogger",
    "PerformanceMonitor",
    "PerformanceTracker",
    "ProcessingSnapshot",
    # Security
    "SensitiveDataFilter",
    "SmartLogLevel",
    # Smart logging
    "SmartLogger",
    "configure_smart_logging",
    "get_logger",
    "get_smart_log_level",
    "log_session_end",
    "log_thought_processing",
    "performance_monitor",
    "reload_logging_config",
    # Main functions
    "setup_logging",
    # Global instances
    "smart_logger"
]
