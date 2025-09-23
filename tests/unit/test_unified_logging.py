"""
Comprehensive tests for the unified logging system.

Testing all logging functionality in a single module after complete refactoring.
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from mcp_server_mas_sequential_thinking.logging_config import (
    JSONFormatter,
    LogFormat,
    LoggerFactory,
    LoggingConfig,
    LogLevel,
    LogTarget,
    MetricsConfig,
    MetricsLogger,
    PerformanceMonitor,
    PerformanceTracker,
    ProcessingSnapshot,
    SensitiveDataFilter,
    SmartLogLevel,
    configure_smart_logging,
    get_logger,
    get_smart_log_level,
    log_session_end,
    log_thought_processing,
    performance_monitor,
    reload_logging_config,
    setup_logging,
    smart_logger,
)


class TestUnifiedLoggingSystem:
    """Test the complete unified logging system."""

    def test_setup_logging_integration(self):
        """Test complete logging setup integration."""
        logger = setup_logging()
        assert isinstance(logger, logging.Logger)
        assert logger.name == "sequential_thinking"

    def test_get_logger_hierarchy(self):
        """Test logger hierarchy creation."""
        module_logger = get_logger("test.module")
        assert isinstance(module_logger, logging.Logger)
        assert "sequential_thinking" in module_logger.name

    def test_smart_log_level_configuration(self):
        """Test smart log level configuration."""
        original_level = get_smart_log_level()

        configure_smart_logging(SmartLogLevel.DEBUG)
        assert smart_logger.log_level == SmartLogLevel.DEBUG

        # Reset
        configure_smart_logging(original_level)

    def test_performance_monitoring_integration(self):
        """Test performance monitoring system."""
        snapshot = ProcessingSnapshot(
            thought_id=1,
            complexity_score=5.0,
            strategy_name="test_strategy",
            processing_time=30.0,
            token_estimate=(100, 200),
            cost_estimate=0.005,
            efficiency_score=0.8
        )

        performance_monitor.record_snapshot(snapshot)
        assert len(performance_monitor.recent_snapshots) >= 1

    def test_metrics_logging_functionality(self):
        """Test metrics logging with formatting."""
        metrics_logger = MetricsLogger()

        # Test metrics block logging
        test_metrics = {
            "test_float": 3.14159,
            "test_bool": True,
            "test_string": "test_value",
            "test_list": [1, 2, 3, 4, 5]
        }

        # Should not raise an exception
        metrics_logger.log_metrics_block("TEST METRICS", test_metrics)

    def test_sensitive_data_filtering(self):
        """Test sensitive data filtering across the system."""
        filter_instance = SensitiveDataFilter()

        sensitive_texts = [
            "API_KEY=sk-1234567890abcdef",
            "Bearer abcd1234567890",
            "postgresql://user:password@localhost/db",
            "ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        ]

        for text in sensitive_texts:
            filtered = filter_instance._redact_sensitive_data(text)
            assert "***REDACTED" in filtered
            # Ensure no actual sensitive data remains
            assert not any(char.isalnum() and len(text) > 8 for char in filtered.split("***REDACTED")[0] if "=" in filtered)

    def test_json_formatter_output(self):
        """Test JSON formatter produces valid JSON."""
        formatter = JSONFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.funcName = "test_func"
        record.module = "test_module"

        result = formatter.format(record)

        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed["message"] == "Test message"
        assert parsed["level"] == "INFO"

    def test_environment_configuration_reload(self):
        """Test environment configuration reload."""
        # Test with different environment settings
        test_env = {
            "LOG_LEVEL": "ERROR",
            "LOG_FORMAT": "json",
            "LOG_TARGETS": "console",
            "SMART_LOG_LEVEL": "debug"
        }

        with patch.dict(os.environ, test_env, clear=False):
            reload_logging_config()

            config = LoggerFactory._config
            assert config.log_level == LogLevel.ERROR
            assert config.log_format == LogFormat.JSON
            assert config.log_targets == [LogTarget.CONSOLE]
            assert config.smart_log_level == SmartLogLevel.DEBUG

    def test_processing_snapshot_properties(self):
        """Test ProcessingSnapshot performance checks."""
        # Fast, cheap, efficient snapshot
        good_snapshot = ProcessingSnapshot(
            thought_id=1,
            complexity_score=5.0,
            strategy_name="fast",
            processing_time=30.0,
            token_estimate=(100, 200),
            cost_estimate=0.001,
            efficiency_score=0.9
        )

        assert not good_snapshot.is_slow
        assert not good_snapshot.is_expensive
        assert not good_snapshot.is_inefficient

        # Slow, expensive, inefficient snapshot
        bad_snapshot = ProcessingSnapshot(
            thought_id=2,
            complexity_score=20.0,
            strategy_name="slow",
            processing_time=120.0,
            token_estimate=(1000, 2000),
            cost_estimate=0.05,
            efficiency_score=0.3
        )

        assert bad_snapshot.is_slow
        assert bad_snapshot.is_expensive
        assert bad_snapshot.is_inefficient

    def test_smart_logger_adaptive_verbosity(self):
        """Test smart logger adapts verbosity based on issues."""
        # Create mock thought data
        class MockThoughtData:
            def __init__(self):
                self.thoughtNumber = 1
                self.thought = "Test thought content"

        thought_data = MockThoughtData()

        # Test normal processing
        normal_snapshot = ProcessingSnapshot(
            thought_id=1,
            complexity_score=5.0,
            strategy_name="normal",
            processing_time=20.0,
            token_estimate=(100, 200),
            cost_estimate=0.001,
            efficiency_score=0.9
        )

        # Should not raise exceptions
        smart_logger.log_thought_start(thought_data)
        smart_logger.log_routing_decision(5.0, "normal", 20.0)
        smart_logger.log_processing_complete(normal_snapshot)
        smart_logger.log_response_quality("Normal response", 1)

    def test_log_thought_processing_integration(self):
        """Test the main log_thought_processing function."""
        class MockThoughtData:
            def __init__(self):
                self.thoughtNumber = 1
                self.thought = "Integration test thought"

        thought_data = MockThoughtData()

        # Should not raise exceptions
        log_thought_processing(
            thought_data=thought_data,
            complexity_score=5.0,
            strategy_name="integration_test",
            processing_time=25.0,
            efficiency_score=0.85,
            content="Test response content",
            token_estimate=(150, 300),
            cost_estimate=0.002,
            reasoning="Test reasoning"
        )

    def test_session_logging_workflow(self):
        """Test complete session logging workflow."""
        # Process some thoughts
        class MockThoughtData:
            def __init__(self, num):
                self.thoughtNumber = num
                self.thought = f"Session test thought {num}"

        for i in range(3):
            thought_data = MockThoughtData(i + 1)
            log_thought_processing(
                thought_data=thought_data,
                complexity_score=float(i + 1) * 2,
                strategy_name=f"session_test_{i}",
                processing_time=20.0 + i * 5,
                efficiency_score=0.8 - i * 0.1,
                content=f"Session response {i + 1}",
                token_estimate=(100 + i * 50, 200 + i * 100),
                cost_estimate=0.001 * (i + 1)
            )

        # End session
        log_session_end()

        # Verify snapshots were recorded
        assert len(smart_logger.session_snapshots) >= 3

    def test_performance_tracker_summary(self):
        """Test performance tracker summary generation."""
        tracker = PerformanceTracker()

        # Record some processing times
        tracker.record_processing(25.0, True)
        tracker.record_processing(30.0, True)
        tracker.record_processing(45.0, False)

        summary = tracker.get_performance_summary()
        assert summary["status"] != "no_data"
        assert "average_processing_time" in summary
        assert "success_rate" in summary
        assert summary["total_attempts"] == 3

    def test_metrics_config_customization(self):
        """Test metrics configuration customization."""
        custom_config = MetricsConfig(
            separator_length=80,
            log_level=LogLevel.DEBUG,
            include_separators=False
        )

        metrics_logger = MetricsLogger(custom_config)
        assert metrics_logger.config.separator_length == 80
        assert metrics_logger.config.log_level == LogLevel.DEBUG
        assert not metrics_logger.config.include_separators

    def test_file_size_parsing(self):
        """Test file size parsing in configuration."""
        config = LoggingConfig()

        # Test various size formats
        assert config._parse_file_size("5MB") == 5 * 1024 * 1024
        assert config._parse_file_size("1GB") == 1024 * 1024 * 1024
        assert config._parse_file_size("500KB") == 500 * 1024
        assert config._parse_file_size("1024B") == 1024
        assert config._parse_file_size("10") == 10 * 1024 * 1024  # Default to MB
        assert config._parse_file_size("invalid") == 10 * 1024 * 1024  # Fallback

    def test_logging_targets_configuration(self):
        """Test logging targets configuration."""
        # Test various target combinations
        assert LogTarget.from_string_list("file") == [LogTarget.FILE]
        assert LogTarget.from_string_list("console") == [LogTarget.CONSOLE]
        assert LogTarget.from_string_list("file,console") == [LogTarget.FILE, LogTarget.CONSOLE]
        assert LogTarget.from_string_list("invalid") == [LogTarget.FILE, LogTarget.CONSOLE]  # Fallback
        assert LogTarget.from_string_list("") == [LogTarget.FILE, LogTarget.CONSOLE]  # Default

    def test_complete_logging_workflow_with_files(self):
        """Test complete logging workflow with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup custom config with file logging
            env_vars = {
                "LOG_TARGETS": "file",
                "LOG_FORMAT": "text"
            }

            with patch.dict(os.environ, env_vars, clear=False):
                config = LoggingConfig()
                config.log_dir = Path(temp_dir) / "test_logs"

                # Reset and configure
                LoggerFactory._config = None
                LoggerFactory._root_logger_configured = False
                LoggerFactory.configure(config)

                # Get logger and log something
                logger = get_logger("test.workflow")
                logger.info("Test log message for file output")

                # Force handlers to flush
                for handler in logging.getLogger("sequential_thinking").handlers:
                    handler.flush()

                # Check if log directory was created
                assert config.log_dir.exists()


class TestLoggingSystemEdgeCases:
    """Test edge cases and error conditions."""

    def test_logger_factory_without_configuration(self):
        """Test logger factory auto-configuration."""
        # Reset factory state
        LoggerFactory._config = None
        LoggerFactory._root_logger_configured = False

        # Should auto-configure
        logger = LoggerFactory.get_logger("auto.config.test")
        assert LoggerFactory._config is not None
        assert isinstance(logger, logging.Logger)

    def test_empty_performance_tracker(self):
        """Test performance tracker with no data."""
        empty_tracker = PerformanceTracker()
        summary = empty_tracker.get_performance_summary()
        assert summary == {"status": "no_data"}

    def test_performance_monitor_recommendations(self):
        """Test performance monitor recommendations."""
        monitor = PerformanceMonitor()

        # Add slow snapshots
        for i in range(6):
            slow_snapshot = ProcessingSnapshot(
                thought_id=i,
                complexity_score=10.0,
                strategy_name="slow",
                processing_time=120.0,  # Slow
                token_estimate=(500, 1000),
                cost_estimate=0.001,
                efficiency_score=0.5
            )
            monitor.record_snapshot(slow_snapshot)

        recommendation = monitor.get_performance_recommendation()
        assert recommendation is not None
        assert "complexity" in recommendation.lower()

    def test_smart_logger_force_debug(self):
        """Test smart logger debug forcing."""
        original_level = smart_logger.log_level

        smart_logger.force_debug_next()
        assert smart_logger.log_level == SmartLogLevel.DEBUG

        # Reset
        smart_logger.log_level = original_level

    def test_metrics_value_formatting_edge_cases(self):
        """Test metrics value formatting with edge cases."""
        logger = MetricsLogger()

        # Test various value types
        test_values = [
            (3.0, "3"),  # Integer-like float
            (3.14159, "3.142"),  # Regular float
            (True, "✅"),  # Boolean true
            (False, "❌"),  # Boolean false
            ([1, 2, 3], "1, 2, 3"),  # Short list
            (list(range(10)), "0, 1, 2... (10 total)"),  # Long list
            (None, "None"),  # None value
            ("string", "string"),  # String value
        ]

        for value, expected in test_values:
            formatted = logger._format_metric_value(value)
            if "total" in expected:
                assert "total" in formatted
            else:
                assert expected in formatted or formatted == expected

    def test_logging_configuration_defaults(self):
        """Test logging configuration defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = LoggingConfig()

            assert config.log_level == LogLevel.INFO
            assert config.smart_log_level == SmartLogLevel.PERFORMANCE
            assert config.log_format == LogFormat.TEXT
            assert LogTarget.FILE in config.log_targets
            assert LogTarget.CONSOLE in config.log_targets
            assert config.log_file_backup_count == 5
            assert config.log_sampling_rate == 1.0


# Integration test for the complete system
def test_complete_system_integration():
    """Test the complete logging system integration."""
    # Reset everything
    LoggerFactory._config = None
    LoggerFactory._root_logger_configured = False

    # Setup with environment configuration
    test_env = {
        "LOG_LEVEL": "INFO",
        "LOG_FORMAT": "text",
        "LOG_TARGETS": "console",
        "SMART_LOG_LEVEL": "performance"
    }

    with patch.dict(os.environ, test_env, clear=False):
        # Initialize system
        root_logger = setup_logging()
        test_logger = get_logger("integration.test")

        # Test all logging functions
        test_logger.debug("Debug message")
        test_logger.info("Info message")
        test_logger.warning("Warning message")
        test_logger.error("Error message")

        # Test smart logging
        smart_level = get_smart_log_level()
        assert smart_level == SmartLogLevel.PERFORMANCE

        # Test thought processing
        class MockThought:
            thoughtNumber = 999
            thought = "Integration test complete"

        log_thought_processing(
            thought_data=MockThought(),
            complexity_score=7.5,
            strategy_name="integration_complete",
            processing_time=35.0,
            efficiency_score=0.9,
            content="Integration test completed successfully",
            token_estimate=(200, 400),
            cost_estimate=0.003
        )

        # End session
        log_session_end()

        # Verify system state
        assert isinstance(root_logger, logging.Logger)
        assert isinstance(test_logger, logging.Logger)
        assert len(smart_logger.session_snapshots) > 0
