"""
TDD测试 - LoggingMixin重构组件

遵循TDD流程：RED → GREEN → REFACTOR
测试重构后的日志工具混入类功能
"""

import pytest
import logging
from unittest.mock import patch, MagicMock
from io import StringIO

from src.mcp_server_mas_sequential_thinking.server_core import LoggingMixin
from src.mcp_server_mas_sequential_thinking.constants import PerformanceMetrics


class TestLoggingMixin:
    """测试LoggingMixin的所有方法"""

    def setup_method(self):
        """为每个测试方法设置"""
        self.logging_mixin = LoggingMixin()

    def test_log_section_header_with_default_length(self):
        """测试带默认分隔符长度的区段标题记录"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            title = "🎯 TEST SECTION"

            self.logging_mixin._log_section_header(title)

            mock_logger.info.assert_called_once_with(title)

    def test_log_section_header_with_custom_length(self):
        """测试带自定义分隔符长度的区段标题记录"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            title = "🎯 CUSTOM SECTION"
            custom_length = 50

            self.logging_mixin._log_section_header(title, custom_length)

            mock_logger.info.assert_called_once_with(title)

    def test_log_metrics_block_with_mixed_types(self):
        """测试记录包含混合数据类型的指标块"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            title = "📊 PERFORMANCE METRICS"
            metrics = {
                "Efficiency Score": 0.85,  # float
                "Strategy": "multi_agent",  # string
                "Token Count": 1500,  # int
                "Custom Object": {"nested": "value"},  # other
            }

            self.logging_mixin._log_metrics_block(title, metrics)

            # 验证调用次数和参数
            expected_calls = [
                (title,),
                ("  Efficiency Score: 0.85",),
                ("  Strategy: multi_agent",),
                ("  Token Count: 1500",),
                ("  Custom Object: {'nested': 'value'}",),
            ]

            assert mock_logger.info.call_count == 5
            for i, (expected_args,) in enumerate(expected_calls):
                actual_call = mock_logger.info.call_args_list[i]
                assert actual_call[0][0] == expected_args

    def test_log_metrics_block_empty_metrics(self):
        """测试记录空指标块"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            title = "📊 EMPTY METRICS"
            metrics = {}

            self.logging_mixin._log_metrics_block(title, metrics)

            # 只应该记录标题
            mock_logger.info.assert_called_once_with(title)

    def test_log_separator_default_length(self):
        """测试默认长度分隔符记录"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            self.logging_mixin._log_separator()

            expected_separator = f"  {'=' * PerformanceMetrics.SEPARATOR_LENGTH}"
            mock_logger.info.assert_called_once_with(expected_separator)

    def test_log_separator_custom_length(self):
        """测试自定义长度分隔符记录"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            custom_length = 30

            self.logging_mixin._log_separator(custom_length)

            expected_separator = f"  {'=' * custom_length}"
            mock_logger.info.assert_called_once_with(expected_separator)

    def test_calculate_efficiency_score_fast_processing(self):
        """测试快速处理的效率评分计算"""
        fast_time = 0.5  # 小于阈值

        score = self.logging_mixin._calculate_efficiency_score(fast_time)

        assert score == PerformanceMetrics.PERFECT_EFFICIENCY_SCORE

    def test_calculate_efficiency_score_slow_processing(self):
        """测试慢速处理的效率评分计算"""
        slow_time = 120.0  # 大于EFFICIENCY_TIME_THRESHOLD (60.0)

        score = self.logging_mixin._calculate_efficiency_score(slow_time)

        expected_score = max(
            PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
            PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / slow_time,
        )
        assert score == expected_score

    def test_calculate_efficiency_score_boundary_case(self):
        """测试边界情况的效率评分计算"""
        boundary_time = PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD

        score = self.logging_mixin._calculate_efficiency_score(boundary_time)

        expected_score = max(
            PerformanceMetrics.MINIMUM_EFFICIENCY_SCORE,
            PerformanceMetrics.EFFICIENCY_TIME_THRESHOLD / boundary_time,
        )
        assert score == expected_score

    def test_calculate_execution_consistency_success(self):
        """测试成功执行的一致性评分计算"""
        score = self.logging_mixin._calculate_execution_consistency(True)

        assert score == PerformanceMetrics.PERFECT_EXECUTION_CONSISTENCY

    def test_calculate_execution_consistency_failure(self):
        """测试失败执行的一致性评分计算"""
        score = self.logging_mixin._calculate_execution_consistency(False)

        assert score == PerformanceMetrics.DEFAULT_EXECUTION_CONSISTENCY


class TestLoggingMixinIntegration:
    """测试LoggingMixin的集成用法"""

    def test_mixin_inheritance(self):
        """测试LoggingMixin可以被正确继承"""

        class TestClass(LoggingMixin):
            def test_method(self):
                return self._calculate_efficiency_score(1.0)

        test_instance = TestClass()
        score = test_instance.test_method()

        assert isinstance(score, float)
        assert score > 0

    def test_complete_logging_workflow(self):
        """测试完整的日志记录工作流程"""

        class MockProcessor(LoggingMixin):
            def process_with_logging(self):
                # 模拟完整的日志记录工作流程
                self._log_section_header("🎯 PROCESSING START")

                metrics = {
                    "Processing Time": 2.5,
                    "Strategy": "test_strategy",
                    "Success": True,
                }
                self._log_metrics_block("📊 METRICS", metrics)

                efficiency = self._calculate_efficiency_score(2.5)
                consistency = self._calculate_execution_consistency(True)

                self._log_separator()

                return efficiency, consistency

        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            processor = MockProcessor()
            efficiency, consistency = processor.process_with_logging()

            # 验证所有日志调用都执行了
            assert mock_logger.info.call_count >= 4  # 标题 + 3个指标 + 分隔符
            assert isinstance(efficiency, float)
            assert isinstance(consistency, float)


# RED阶段测试 - 这些测试应该失败，直到实现了对应功能
class TestLoggingMixinEdgeCases:
    """测试LoggingMixin的边界情况和错误处理"""

    def test_log_metrics_with_none_values(self):
        """测试处理包含None值的指标"""
        with patch(
            "src.mcp_server_mas_sequential_thinking.server_core.logger"
        ) as mock_logger:
            mixin = LoggingMixin()
            metrics = {"Valid Metric": 1.0, "None Metric": None}

            mixin._log_metrics_block("TEST METRICS", metrics)

            # 应该正确处理None值
            assert mock_logger.info.call_count == 3  # 标题 + 2个指标

    def test_efficiency_score_zero_processing_time(self):
        """测试零处理时间的效率评分"""
        mixin = LoggingMixin()

        # 零时间应该返回完美评分
        score = mixin._calculate_efficiency_score(0.0)
        assert score == PerformanceMetrics.PERFECT_EFFICIENCY_SCORE

    def test_efficiency_score_negative_processing_time(self):
        """测试负处理时间的效率评分（不应该发生但需要处理）"""
        mixin = LoggingMixin()

        # 负时间应该返回最小评分或完美评分（取决于实现）
        score = mixin._calculate_efficiency_score(-1.0)
        assert isinstance(score, float)
        assert score >= 0  # 评分不应该为负
