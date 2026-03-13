"""Unit tests for infrastructure/logging_config.py."""

import logging
import time

import pytest

from mcp_server_mas_sequential_thinking.infrastructure.logging_config import (
    LogTimer,
    MetricsLogger,
    configure_logging,
    create_logger,
    get_logger,
    log_performance_metric,
    log_routing_decision,
    log_thought_processing,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_returns_logger(self):
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)

    def test_with_explicit_level(self):
        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_with_warning_level(self):
        logger = setup_logging(level="WARNING")
        assert logger.level == logging.WARNING

    def test_invalid_level_defaults_to_info(self):
        logger = setup_logging(level="NOTVALID")
        assert logger.level == logging.INFO


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_named_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_with_none_name_uses_caller_module(self):
        logger = get_logger(None)
        assert isinstance(logger, logging.Logger)


class TestLegacyFunctions:
    """Tests for legacy compatibility functions."""

    def test_create_logger(self):
        logger = create_logger("my_logger")
        assert isinstance(logger, logging.Logger)

    def test_configure_logging(self):
        logger = configure_logging("INFO")
        assert isinstance(logger, logging.Logger)


class TestLogPerformanceMetric:
    """Tests for log_performance_metric function."""

    def test_logs_without_extras(self):
        logger = logging.getLogger("test_perf")
        logger.setLevel(logging.INFO)
        log_performance_metric(logger, "test_op", 1.23)

    def test_logs_with_extras(self):
        logger = logging.getLogger("test_perf")
        logger.setLevel(logging.INFO)
        log_performance_metric(logger, "test_op", 1.23, count=5, status="ok")

    def test_disabled_logger_skips(self):
        logger = logging.getLogger("disabled_logger")
        logger.setLevel(logging.CRITICAL)
        log_performance_metric(logger, "op", 1.0)  # Should not raise


class TestLogRoutingDecision:
    """Tests for log_routing_decision function."""

    def test_basic_call(self):
        logger = logging.getLogger("test_routing")
        log_routing_decision(logger, "full_sequence", 8.5)

    def test_with_reasoning(self):
        logger = logging.getLogger("test_routing")
        log_routing_decision(logger, "single_hat", 2.0, reasoning="simple thought")


class TestLogThoughtProcessing:
    """Tests for log_thought_processing function."""

    def test_basic_call(self):
        logger = logging.getLogger("test_thought")
        logger.setLevel(logging.INFO)
        log_thought_processing(logger, "start", 1, thought_length=100)

    def test_with_context(self):
        logger = logging.getLogger("test_thought")
        logger.setLevel(logging.INFO)
        log_thought_processing(logger, "end", 2, thought_length=200, strategy="full")

    def test_disabled_logger(self):
        logger = logging.getLogger("disabled_thought")
        logger.setLevel(logging.CRITICAL)
        log_thought_processing(logger, "start", 1)  # Should not raise


class TestLogTimer:
    """Tests for LogTimer context manager."""

    def test_successful_operation(self):
        logger = logging.getLogger("test_timer")
        logger.setLevel(logging.INFO)
        with LogTimer(logger, "test_operation"):
            time.sleep(0.001)

    def test_failed_operation(self):
        logger = logging.getLogger("test_timer_fail")
        logger.setLevel(logging.INFO)
        with pytest.raises(ValueError):
            with LogTimer(logger, "failing_op"):
                raise ValueError("test error")

    def test_debug_level_start_log(self):
        logger = logging.getLogger("test_timer_debug")
        logger.setLevel(logging.DEBUG)
        with LogTimer(logger, "debug_op", level=logging.DEBUG):
            pass

    def test_timer_without_start_time(self):
        """Covers the case where start_time is None."""
        logger = logging.getLogger("test_timer_no_start")
        timer = LogTimer(logger, "no_start")
        timer.start_time = None
        timer.__exit__(None, None, None)


class TestMetricsLogger:
    """Tests for MetricsLogger."""

    def test_log_metrics_block(self):
        ml = MetricsLogger("test_metrics")
        ml.logger.setLevel(logging.INFO)
        ml.log_metrics_block("My Metrics", {"key1": "val1", "key2": 42})

    def test_log_metrics_block_disabled(self):
        ml = MetricsLogger("disabled_metrics")
        ml.logger.setLevel(logging.CRITICAL)
        ml.log_metrics_block("Title", {"k": "v"})  # Should not log

    def test_log_separator(self):
        ml = MetricsLogger("test_sep")
        ml.logger.setLevel(logging.INFO)
        ml.log_separator(40)

    def test_log_separator_disabled(self):
        ml = MetricsLogger("disabled_sep")
        ml.logger.setLevel(logging.CRITICAL)
        ml.log_separator()  # Should not log

    def test_default_logger_name(self):
        ml = MetricsLogger()
        assert ml.logger.name == "sequential_thinking"
