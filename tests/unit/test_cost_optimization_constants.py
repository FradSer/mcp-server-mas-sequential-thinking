"""
TDD测试 - CostOptimizationConstants重构组件

遵循TDD流程：RED → GREEN → REFACTOR
测试重构后的成本优化常量类功能
"""


from mcp_server_mas_sequential_thinking.constants import CostOptimizationConstants


class TestCostOptimizationConstants:
    """测试CostOptimizationConstants的所有常量"""

    def test_quality_scoring_weights_sum_to_one(self):
        """测试质量评分权重总和为1.0"""
        total_weight = (
            CostOptimizationConstants.QUALITY_WEIGHT
            + CostOptimizationConstants.COST_WEIGHT
            + CostOptimizationConstants.SPEED_WEIGHT
            + CostOptimizationConstants.RELIABILITY_WEIGHT
        )

        assert abs(total_weight - 1.0) < 0.0001  # 允许浮点数精度误差

    def test_quality_scoring_weights_values(self):
        """测试质量评分权重的具体值"""
        assert CostOptimizationConstants.QUALITY_WEIGHT == 0.4
        assert CostOptimizationConstants.COST_WEIGHT == 0.3
        assert CostOptimizationConstants.SPEED_WEIGHT == 0.2
        assert CostOptimizationConstants.RELIABILITY_WEIGHT == 0.1

    def test_quality_scoring_weights_are_positive(self):
        """测试所有权重都是正数"""
        weights = [
            CostOptimizationConstants.QUALITY_WEIGHT,
            CostOptimizationConstants.COST_WEIGHT,
            CostOptimizationConstants.SPEED_WEIGHT,
            CostOptimizationConstants.RELIABILITY_WEIGHT,
        ]

        for weight in weights:
            assert weight > 0
            assert isinstance(weight, float)

    def test_cost_calculation_factors_values(self):
        """测试成本计算因子的具体值"""
        assert CostOptimizationConstants.COST_NORMALIZATION_FACTOR == 0.0003
        assert CostOptimizationConstants.COST_EPSILON == 0.0001
        assert CostOptimizationConstants.DEFAULT_COST_ESTIMATE == 0.0002
        assert CostOptimizationConstants.SPEED_NORMALIZATION_BASE == 10
        assert CostOptimizationConstants.SPEED_THRESHOLD == 1

    def test_cost_calculation_factors_are_positive(self):
        """测试成本计算因子都是正数"""
        factors = [
            CostOptimizationConstants.COST_NORMALIZATION_FACTOR,
            CostOptimizationConstants.COST_EPSILON,
            CostOptimizationConstants.DEFAULT_COST_ESTIMATE,
            CostOptimizationConstants.SPEED_NORMALIZATION_BASE,
            CostOptimizationConstants.SPEED_THRESHOLD,
        ]

        for factor in factors:
            assert factor > 0

    def test_quality_scoring_bounds(self):
        """测试质量评分边界值"""
        assert CostOptimizationConstants.MIN_QUALITY_SCORE == 0.0
        assert CostOptimizationConstants.MAX_QUALITY_SCORE == 1.0
        assert (
            CostOptimizationConstants.MIN_QUALITY_SCORE
            < CostOptimizationConstants.MAX_QUALITY_SCORE
        )

    def test_budget_utilization_thresholds_order(self):
        """测试预算使用阈值的顺序关系"""
        moderate = CostOptimizationConstants.MODERATE_BUDGET_UTILIZATION
        high = CostOptimizationConstants.HIGH_BUDGET_UTILIZATION
        critical = CostOptimizationConstants.CRITICAL_BUDGET_UTILIZATION

        assert moderate < high < critical
        assert 0 < moderate < 1
        assert 0 < high < 1
        assert 0 < critical < 1

    def test_budget_utilization_threshold_values(self):
        """测试预算使用阈值的具体值"""
        assert CostOptimizationConstants.MODERATE_BUDGET_UTILIZATION == 0.7
        assert CostOptimizationConstants.HIGH_BUDGET_UTILIZATION == 0.8
        assert CostOptimizationConstants.CRITICAL_BUDGET_UTILIZATION == 0.9

    def test_complexity_bonuses_are_positive(self):
        """测试复杂度奖励都是正数"""
        assert CostOptimizationConstants.QUALITY_COMPLEXITY_BONUS > 0
        assert CostOptimizationConstants.COST_COMPLEXITY_BONUS > 0

        assert CostOptimizationConstants.QUALITY_COMPLEXITY_BONUS == 0.2
        assert CostOptimizationConstants.COST_COMPLEXITY_BONUS == 0.0001

    def test_provider_optimization_constants(self):
        """测试提供商优化常量"""
        assert CostOptimizationConstants.HIGH_USAGE_PENALTY == 2.0
        assert CostOptimizationConstants.MODERATE_USAGE_PENALTY == 0.5
        assert CostOptimizationConstants.QUALITY_UPDATE_WEIGHT == 0.1
        assert CostOptimizationConstants.OLD_QUALITY_WEIGHT == 0.9

        # 验证权重关系
        assert (
            CostOptimizationConstants.QUALITY_UPDATE_WEIGHT
            + CostOptimizationConstants.OLD_QUALITY_WEIGHT
        ) == 1.0

    def test_usage_analysis_thresholds(self):
        """测试使用分析阈值"""
        assert CostOptimizationConstants.MIN_DATA_THRESHOLD == 10
        assert CostOptimizationConstants.HIGH_MULTI_AGENT_RATIO == 0.7
        assert CostOptimizationConstants.HIGH_SINGLE_AGENT_RATIO == 0.8
        assert CostOptimizationConstants.MINIMUM_COST_DIFFERENCE == 0.01

        # 验证比率在有效范围内
        assert 0 < CostOptimizationConstants.HIGH_MULTI_AGENT_RATIO < 1
        assert 0 < CostOptimizationConstants.HIGH_SINGLE_AGENT_RATIO < 1

    def test_groq_provider_configuration(self):
        """测试Groq提供商配置"""
        assert CostOptimizationConstants.GROQ_RATE_LIMIT == 14400
        assert CostOptimizationConstants.GROQ_CONTEXT_LENGTH == 32768
        assert CostOptimizationConstants.GROQ_QUALITY_SCORE == 0.75
        assert CostOptimizationConstants.GROQ_RESPONSE_TIME == 0.8

        # 验证质量分数在有效范围内
        assert 0 <= CostOptimizationConstants.GROQ_QUALITY_SCORE <= 1

    def test_deepseek_provider_configuration(self):
        """测试Deepseek提供商配置"""
        assert CostOptimizationConstants.DEEPSEEK_QUALITY_SCORE == 0.85
        assert CostOptimizationConstants.DEEPSEEK_CONTEXT_LENGTH == 128000

        assert 0 <= CostOptimizationConstants.DEEPSEEK_QUALITY_SCORE <= 1

    def test_provider_context_lengths_are_positive(self):
        """测试所有提供商的上下文长度都是正数"""
        context_lengths = [
            CostOptimizationConstants.GROQ_CONTEXT_LENGTH,
            CostOptimizationConstants.DEEPSEEK_CONTEXT_LENGTH,
            CostOptimizationConstants.GITHUB_CONTEXT_LENGTH,
            CostOptimizationConstants.OPENROUTER_CONTEXT_LENGTH,
            CostOptimizationConstants.OLLAMA_CONTEXT_LENGTH,
        ]

        for length in context_lengths:
            assert length > 0
            assert isinstance(length, int)

    def test_provider_response_times_are_positive(self):
        """测试所有提供商的响应时间都是正数"""
        response_times = [
            CostOptimizationConstants.GROQ_RESPONSE_TIME,
            CostOptimizationConstants.OPENROUTER_RESPONSE_TIME,
            CostOptimizationConstants.OLLAMA_RESPONSE_TIME,
        ]

        for time in response_times:
            assert time > 0
            assert isinstance(time, (int, float))


class TestCostOptimizationConstantsIntegration:
    """测试CostOptimizationConstants的集成使用场景"""

    def test_quality_score_calculation_bounds(self):
        """测试质量评分计算的边界值"""
        min_score = CostOptimizationConstants.MIN_QUALITY_SCORE
        max_score = CostOptimizationConstants.MAX_QUALITY_SCORE

        # 模拟质量评分计算
        sample_scores = [0.0, 0.5, 1.0, 0.85, 0.75]

        for score in sample_scores:
            assert min_score <= score <= max_score

    def test_weight_normalization_example(self):
        """测试权重归一化示例"""
        # 模拟使用权重计算总分
        quality_component = 0.8 * CostOptimizationConstants.QUALITY_WEIGHT
        cost_component = 0.9 * CostOptimizationConstants.COST_WEIGHT
        speed_component = 0.7 * CostOptimizationConstants.SPEED_WEIGHT
        reliability_component = 0.95 * CostOptimizationConstants.RELIABILITY_WEIGHT

        total_score = (
            quality_component + cost_component + speed_component + reliability_component
        )

        # 总分应该在合理范围内
        assert 0 <= total_score <= 1

    def test_budget_utilization_classification(self):
        """测试预算使用分类"""
        moderate = CostOptimizationConstants.MODERATE_BUDGET_UTILIZATION
        high = CostOptimizationConstants.HIGH_BUDGET_UTILIZATION
        critical = CostOptimizationConstants.CRITICAL_BUDGET_UTILIZATION

        # 测试分类逻辑
        test_values = [0.6, 0.75, 0.85, 0.95]

        for value in test_values:
            if value < moderate:
                category = "low"
            elif value < high:
                category = "moderate"
            elif value < critical:
                category = "high"
            else:
                category = "critical"

            assert category in ["low", "moderate", "high", "critical"]

    def test_provider_comparison_metrics(self):
        """测试提供商比较指标的一致性"""
        # 所有提供商都应该有质量分数在0-1范围内
        quality_scores = [
            CostOptimizationConstants.GROQ_QUALITY_SCORE,
            CostOptimizationConstants.DEEPSEEK_QUALITY_SCORE,
            CostOptimizationConstants.OLLAMA_QUALITY_SCORE,
        ]

        for score in quality_scores:
            assert 0 <= score <= 1

        # 上下文长度应该合理
        context_lengths = [
            CostOptimizationConstants.GROQ_CONTEXT_LENGTH,
            CostOptimizationConstants.DEEPSEEK_CONTEXT_LENGTH,
            CostOptimizationConstants.GITHUB_CONTEXT_LENGTH,
            CostOptimizationConstants.OPENROUTER_CONTEXT_LENGTH,
            CostOptimizationConstants.OLLAMA_CONTEXT_LENGTH,
        ]

        for length in context_lengths:
            assert 1000 <= length <= 300000  # 合理的上下文长度范围


class TestCostOptimizationConstantsEdgeCases:
    """测试CostOptimizationConstants的边界情况"""

    def test_epsilon_prevents_division_by_zero(self):
        """测试epsilon常量防止除零错误"""
        epsilon = CostOptimizationConstants.COST_EPSILON

        # 模拟除法操作
        test_divisor = 0.0
        safe_divisor = test_divisor + epsilon

        # 应该能安全执行除法
        result = 1.0 / safe_divisor
        assert result > 0
        assert float("inf") != result

    def test_normalization_factor_scaling(self):
        """测试归一化因子的缩放效果"""
        factor = CostOptimizationConstants.COST_NORMALIZATION_FACTOR

        # 测试不同数量级的值
        test_values = [0.001, 0.01, 0.1, 1.0]

        for value in test_values:
            normalized = value / factor
            assert normalized > 0

    def test_threshold_boundaries(self):
        """测试阈值边界行为"""
        moderate = CostOptimizationConstants.MODERATE_BUDGET_UTILIZATION
        high = CostOptimizationConstants.HIGH_BUDGET_UTILIZATION

        # 测试边界值
        boundary_values = [
            moderate - 0.001,
            moderate,
            moderate + 0.001,
            high - 0.001,
            high,
            high + 0.001,
        ]

        for value in boundary_values:
            # 所有边界值都应该在0-1范围内
            assert 0 <= value <= 1

    def test_constant_immutability_concept(self):
        """测试常量的概念不变性（通过验证类型和值）"""
        # 验证关键常量的类型和值没有改变
        assert isinstance(CostOptimizationConstants.QUALITY_WEIGHT, float)
        assert isinstance(CostOptimizationConstants.MIN_DATA_THRESHOLD, int)
        assert isinstance(CostOptimizationConstants.GROQ_CONTEXT_LENGTH, int)

        # 验证一些关键关系保持不变
        assert (
            CostOptimizationConstants.MIN_QUALITY_SCORE
            < CostOptimizationConstants.MAX_QUALITY_SCORE
        )
        assert (
            CostOptimizationConstants.COST_EPSILON
            < CostOptimizationConstants.COST_NORMALIZATION_FACTOR
        )

    def test_provider_specific_value_ranges(self):
        """测试提供商特定值的合理范围"""
        # Groq 速度应该很快
        assert CostOptimizationConstants.GROQ_RESPONSE_TIME < 1.0

        # OpenRouter 可能较慢但仍合理
        assert CostOptimizationConstants.OPENROUTER_RESPONSE_TIME < 10.0

        # Ollama 本地运行可能较慢
        assert CostOptimizationConstants.OLLAMA_RESPONSE_TIME < 20.0

        # Deepseek 有大上下文
        assert CostOptimizationConstants.DEEPSEEK_CONTEXT_LENGTH >= 100000
