"""Streamlined logging configuration based on Python best practices.

Replaces complex 985-line implementation with focused, performance-optimized approach.
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path


def setup_logging(level: str | None = None) -> logging.Logger:
    """Setup streamlined logging with environment-based configuration.

    Args:
        level: Log level override. If None, uses LOG_LEVEL env var or defaults to INFO.

    Returns:
        Configured logger instance for the application.
    """
    # Determine log level from environment or parameter
    log_level = level or os.getenv("LOG_LEVEL", "INFO")

    try:
        numeric_level = getattr(logging, log_level.upper())
    except AttributeError:
        numeric_level = logging.INFO

    # Create logs directory
    log_dir = Path.home() / ".sequential_thinking" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Configure root logger for this application
    logger = logging.getLogger("sequential_thinking")
    logger.setLevel(numeric_level)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler for development/debugging
    if os.getenv("ENVIRONMENT") != "production":
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation for persistent logging
    log_file = log_dir / "sequential_thinking.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicates
    logger.propagate = False

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance with consistent configuration.

    Args:
        name: Logger name. If None, uses calling module's name.

    Returns:
        Logger instance.
    """
    if name is None:
        # Get caller's module name for better traceability
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "sequential_thinking")

    return logging.getLogger(name)


def log_performance_metric(
    logger: logging.Logger, operation: str, duration: float, **kwargs
) -> None:
    """Log performance metrics in consistent format.

    Uses lazy evaluation to avoid string formatting overhead.
    """
    if logger.isEnabledFor(logging.INFO):
        extras = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        logger.info(
            "Performance: %s completed in %.2fs%s",
            operation,
            duration,
            f" ({extras})" if extras else "",
        )


def log_routing_decision(
    logger: logging.Logger, strategy: str, complexity: float, reasoning: str = ""
) -> None:
    """Log AI routing decisions with consistent structure."""
    logger.info(
        "AI Routing: strategy=%s, complexity=%.1f%s",
        strategy,
        complexity,
        f", reason={reasoning}" if reasoning else "",
    )


def log_thought_processing(
    logger: logging.Logger,
    stage: str,
    thought_number: int,
    thought_length: int = 0,
    **context,
) -> None:
    """Log thought processing stages with structured data."""
    if logger.isEnabledFor(logging.INFO):
        ctx_str = ", ".join(f"{k}={v}" for k, v in context.items()) if context else ""
        logger.info(
            "Thought Processing: stage=%s, number=%d, length=%d%s",
            stage,
            thought_number,
            thought_length,
            f", {ctx_str}" if ctx_str else "",
        )


class LogTimer:
    """Context manager for timing operations with automatic logging."""

    def __init__(
        self, logger: logging.Logger, operation: str, level: int = logging.INFO
    ) -> None:
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None

    def __enter__(self):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Starting: %s", self.operation)

        import time

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time

        duration = time.time() - self.start_time

        if exc_type is None:
            if self.logger.isEnabledFor(self.level):
                self.logger.log(
                    self.level, "Completed: %s (%.2fs)", self.operation, duration
                )
        else:
            self.logger.error(
                "Failed: %s (%.2fs) - %s", self.operation, duration, exc_val
            )


# Legacy compatibility - maintain existing function names but with simplified implementation
def create_logger(name: str) -> logging.Logger:
    """Legacy compatibility function."""
    return get_logger(name)


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Legacy compatibility function."""
    return setup_logging(level)


class MetricsLogger:
    """Simple metrics logger for structured logging output."""

    def __init__(self, logger_name: str = "sequential_thinking") -> None:
        """Initialize metrics logger with specified logger name."""
        self.logger = logging.getLogger(logger_name)

    def log_metrics_block(self, title: str, metrics: dict) -> None:
        """Log a block of metrics with a title.

        Args:
            title: Block title to display
            metrics: Dictionary of metrics to log
        """
        if not self.logger.isEnabledFor(logging.INFO):
            return

        self.logger.info("%s", title)
        for key, value in metrics.items():
            self.logger.info("  %s: %s", key, value)

    def log_separator(self, length: int = 60) -> None:
        """Log a separator line.

        Args:
            length: Length of the separator line
        """
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info("-" * length)
