"""Quality assurance and evaluation system using Agno Evals."""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any

from agno.agent import Agent
from agno.eval.accuracy import AccuracyEval, AccuracyResult
from agno.models.openai import OpenAIChat

from .models import ThoughtData
from .intelligent_coordinator import CoordinationPlan
from .route_execution import ExecutionMode
from .modernized_config import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class SystemPerformanceMetrics:
    """Comprehensive system performance metrics."""

    coordination_accuracy: float
    execution_consistency: float
    response_quality: float
    efficiency_score: float
    processing_time: float
    cost_effectiveness: float

    overall_score: float
    improvement_suggestions: List[str]


class CoordinationAccuracyEvaluator:
    """Evaluates coordination decision quality using Agno AccuracyEval."""

    def __init__(self):
        # Use lightweight model for evaluation
        config = get_model_config()
        self.eval_model = config.create_agent_model()

    def create_coordination_evaluation(
        self,
        thought: ThoughtData,
        plan: CoordinationPlan,
        response: str
    ) -> AccuracyEval:
        """Create coordination accuracy evaluation."""

        # Detect if content is primarily Chinese to use appropriate evaluation approach
        chinese_chars = len([c for c in thought.thought if '\u4e00' <= c <= '\u9fff'])
        is_chinese_content = chinese_chars > len(thought.thought) * 0.3

        if is_chinese_content:
            # Use Chinese-specific evaluation approach
            input_text = f"""
分析以下中文思维处理的协调决策是否合理：

原始思维: "{thought.thought}"
思维上下文: 第{thought.thought_number}个思维，共{thought.total_thoughts}个，下一步需要：{thought.next_needed}

协调决策:
- 策略: {plan.strategy.value}
- 复杂度: {plan.complexity_level.value} (评分: {plan.complexity_score:.1f}/100)
- 执行模式: {plan.execution_mode.value}
- 专家组合: {plan.specialist_roles}
- 协调方式: {plan.coordination_strategy}
- 置信度: {plan.confidence:.2f}

实际响应: "{response[:200]}..."

请评估这个协调决策的准确性和合理性，特别考虑中文哲学思维的特点。
"""

            expected_output = f"""
基于中文思维的复杂度和文化内涵，协调决策评估：

1. 复杂度评估{'' if plan.complexity_score >= 30 else '可能'}准确
2. 策略选择{'' if plan.strategy.value != 'single_agent' or plan.complexity_score < 30 else '可能'}合理
3. 专家组合{'' if len(plan.specialist_roles) >= 2 or plan.complexity_score < 30 else '可能'}适当
4. 执行模式与策略一致
5. 文化适应性良好

总体评估：协调决策{'合理' if plan.confidence > 0.5 else '需要改进'}
"""

            agent_instructions = [
                "你是中文思维处理协调决策的质量评估专家。",
                "评估协调决策是否合理，包括复杂度评估、策略选择、专家组合等。",
                "特别关注中文思维的语义复杂度、哲学深度和文化内涵。",
                "基于思维内容的实际复杂度，判断协调决策是否准确。",
                "考虑中文表达的含蓄性和多层含义。",
                "提供具体的改进建议。",
                "用简洁清晰的中文回答。"
            ]

            additional_guidelines = "评估应考虑中文思维的语义复杂度和哲学深度，协调决策应与实际复杂度匹配。特别关注中文表达的文化背景和隐含意义。"

        else:
            # Use English evaluation approach for non-Chinese content
            input_text = f"""
Analyze whether the coordination decision for this thought processing is appropriate:

ORIGINAL THOUGHT: "{thought.thought}"
THOUGHT CONTEXT: #{thought.thought_number}/{thought.total_thoughts}, next_needed={thought.next_needed}

COORDINATION DECISION:
- Strategy: {plan.strategy.value}
- Complexity: {plan.complexity_level.value} (score: {plan.complexity_score:.1f}/100)
- Execution Mode: {plan.execution_mode.value}
- Specialists: {plan.specialist_roles}
- Coordination: {plan.coordination_strategy}
- Confidence: {plan.confidence:.2f}

ACTUAL RESPONSE: "{response[:200]}..."

Evaluate the accuracy and appropriateness of this coordination decision.
"""

            expected_output = f"""
Based on the thought's complexity and content, coordination decision assessment:

1. Complexity assessment {'is' if plan.complexity_score >= 30 else 'may be'} accurate
2. Strategy selection {'is' if plan.strategy.value != 'single_agent' or plan.complexity_score < 30 else 'may be'} appropriate
3. Specialist combination {'is' if len(plan.specialist_roles) >= 2 or plan.complexity_score < 30 else 'may be'} suitable
4. Execution mode matches strategy
5. Overall coordination approach is effective

Overall assessment: Coordination decision is {'reasonable' if plan.confidence > 0.5 else 'needs improvement'}
"""

            agent_instructions = [
                "You are an expert evaluator of coordination decisions for thought processing.",
                "Assess whether coordination decisions are reasonable, including complexity assessment, strategy selection, and specialist combinations.",
                "Focus on content complexity analysis and strategic appropriateness.",
                "Consider the actual complexity of the thought when judging coordination accuracy.",
                "Provide specific, actionable improvement suggestions.",
                "Answer clearly and concisely."
            ]

            additional_guidelines = "Evaluation should consider the semantic complexity and philosophical depth of thoughts. Coordination decisions should match actual complexity levels."

        # Create evaluation agent (language-appropriate evaluator)
        eval_agent = Agent(
            name="CoordinationEvaluator",
            role="协调决策质量评估专家" if is_chinese_content else "Coordination Decision Quality Assessor",
            model=self.eval_model,
            instructions=agent_instructions,
            markdown=False
        )

        return AccuracyEval(
            name="Coordination Accuracy Evaluation",
            model=self.eval_model,
            agent=eval_agent,
            input=input_text,
            expected_output=expected_output,
            additional_guidelines=additional_guidelines,
            num_iterations=1  # Single iteration for performance
        )


class ResponseQualityEvaluator:
    """Evaluates response quality using Agno AccuracyEval."""

    def __init__(self):
        config = get_model_config()
        self.eval_model = config.create_agent_model()

    def create_quality_evaluation(
        self,
        thought: ThoughtData,
        response: str
    ) -> AccuracyEval:
        """Create response quality evaluation."""

        # Detect if content is primarily Chinese to use appropriate evaluation approach
        chinese_chars = len([c for c in thought.thought if '\u4e00' <= c <= '\u9fff'])
        is_chinese_content = chinese_chars > len(thought.thought) * 0.3

        if is_chinese_content:
            # Chinese-specific evaluation approach
            input_text = f"""
评估以下中文思维处理响应的质量：

原始思维: "{thought.thought}"
处理响应: "{response}"

评估标准：
1. 是否充分回应了原始思维的内容？
2. 是否提供了有价值的分析或指导？
3. 是否包含推进后续思维的指导(Guidance)？
4. 内容组织是否清晰有序？
5. 是否符合Sequential Thinking的要求？
6. 是否体现了中文思维的深度和文化内涵？
7. 是否考虑了中文表达的含蓄性和多层含义？
"""

            expected_output = f"""
中文思维响应质量评估：

1. 内容相关性：{'高' if len(response) > 100 else '中等'}
2. 分析深度：基于中文哲学思维的深度分析
3. 指导价值：{'包含明确的下一步指导' if 'guidance' in response.lower() or '指导' in response else '缺少明确指导'}
4. 结构清晰：{'良好' if any(marker in response for marker in ['#', '*', '-', '1.', '2.']) else '需要改进'}
5. 思维推进：为后续思维提供基础
6. 文化适应性：{'体现中文思维特色' if any(term in response for term in ['哲学', '内心', '感悟', '修养', '境界']) else '可以更好体现文化内涵'}
7. 语言表达：{'生动有意境' if any(marker in response for marker in ['比喻', '隐喻', '诗意']) else '表达可以更有文化特色'}

总体质量：{'优秀' if len(response) > 200 and ('guidance' in response.lower() or '指导' in response) else '良好'}
"""

            agent_instructions = [
                "你是中文思维处理响应质量的评估专家。",
                "评估响应是否有效推进Sequential Thinking流程。",
                "关注内容质量、指导价值和思维链条的连续性。",
                "特别关注中文哲学思维的分析深度和文化内涵。",
                "考虑中文表达的含蓄性、意境和文化背景。",
                "评估是否体现了中文思维的特色和深度。",
                "提供具体改进建议。"
            ]

            additional_guidelines = "重点评估响应对Sequential Thinking流程的推进作用，确保为下一个思维提供有效指导。特别关注中文思维的文化特色和哲学深度。"

        else:
            # English evaluation approach
            input_text = f"""
评估以下思维处理响应的质量：

原始思维: "{thought.thought}"
处理响应: "{response}"

评估标准：
1. 是否充分回应了原始思维的内容？
2. 是否提供了有价值的分析或指导？
3. 是否包含推进后续思维的指导(Guidance)？
4. 内容组织是否清晰有序？
5. 是否符合Sequential Thinking的要求？
"""

            expected_output = f"""
响应质量评估：

1. 内容相关性：{'高' if len(response) > 100 else '中等'}
2. 分析深度：基于哲学思维的深度分析
3. 指导价值：{'包含明确的下一步指导' if 'guidance' in response.lower() or '指导' in response else '缺少明确指导'}
4. 结构清晰：{'良好' if any(marker in response for marker in ['#', '*', '-', '1.']) else '需要改进'}
5. 思维推进：为后续思维提供基础

总体质量：{'优秀' if len(response) > 200 and ('guidance' in response.lower() or '指导' in response) else '良好'}
"""

            agent_instructions = [
                "你是思维处理响应质量的评估专家。",
                "评估响应是否有效推进Sequential Thinking流程。",
                "关注内容质量、指导价值和思维链条的连续性。",
                "特别关注中文哲学思维的分析深度。",
                "提供具体改进建议。"
            ]

            additional_guidelines = "重点评估响应对Sequential Thinking流程的推进作用，确保为下一个思维提供有效指导。"

        eval_agent = Agent(
            name="ResponseQualityEvaluator",
            role="响应质量评估专家" if is_chinese_content else "Response Quality Evaluator",
            model=self.eval_model,
            instructions=agent_instructions,
            markdown=False
        )

        return AccuracyEval(
            name="Response Quality Evaluation",
            model=self.eval_model,
            agent=eval_agent,
            input=input_text,
            expected_output=expected_output,
            additional_guidelines=additional_guidelines,
            num_iterations=1
        )


class QualityAssuranceManager:
    """Manages comprehensive quality assurance using Agno Evals."""

    def __init__(self):
        self.coordination_evaluator = CoordinationAccuracyEvaluator()
        self.quality_evaluator = ResponseQualityEvaluator()
        self.performance_history: List[SystemPerformanceMetrics] = []

    async def evaluate_full_pipeline(
        self,
        thought: ThoughtData,
        plan: CoordinationPlan,
        execution_log: Dict[str, Any],
        response: str,
        processing_time: float
    ) -> SystemPerformanceMetrics:
        """Evaluate the complete thought processing pipeline using Agno Evals."""

        logger.info("🔍 Running Agno-powered quality evaluation...")

        # Initialize scores
        coordination_accuracy = 0.5
        response_quality = 0.5
        execution_consistency = 1.0

        # Run coordination accuracy evaluation
        try:
            coord_eval = self.coordination_evaluator.create_coordination_evaluation(
                thought, plan, response
            )

            logger.info("  📊 Evaluating coordination accuracy...")
            coord_result: Optional[AccuracyResult] = coord_eval.run(print_results=False)

            if coord_result and coord_result.avg_score is not None:
                coordination_accuracy = coord_result.avg_score / 10.0  # Convert 0-10 to 0-1
                logger.info(f"  ✅ Coordination accuracy: {coordination_accuracy:.2f}")
            else:
                logger.warning("  ⚠️  Coordination evaluation failed, using fallback")

        except Exception as e:
            logger.warning(f"Coordination evaluation error: {e}")

        # Run response quality evaluation
        try:
            quality_eval = self.quality_evaluator.create_quality_evaluation(
                thought, response
            )

            logger.info("  📊 Evaluating response quality...")
            quality_result: Optional[AccuracyResult] = quality_eval.run(print_results=False)

            if quality_result and quality_result.avg_score is not None:
                response_quality = quality_result.avg_score / 10.0  # Convert 0-10 to 0-1
                logger.info(f"  ✅ Response quality: {response_quality:.2f}")
            else:
                logger.warning("  ⚠️  Quality evaluation failed, using fallback")

        except Exception as e:
            logger.warning(f"Quality evaluation error: {e}")

        # Evaluate execution consistency (rule-based)
        execution_consistency = self._evaluate_execution_consistency(plan, execution_log)
        logger.info(f"  ✅ Execution consistency: {execution_consistency:.2f}")

        # Calculate efficiency
        efficiency_score = self._calculate_efficiency(plan, processing_time)
        logger.info(f"  ✅ Efficiency: {efficiency_score:.2f}")

        # Calculate overall score
        overall_score = (
            coordination_accuracy * 0.3 +
            execution_consistency * 0.2 +
            response_quality * 0.3 +
            efficiency_score * 0.2
        )

        # Generate improvement suggestions
        suggestions = []
        if coordination_accuracy < 0.7:
            suggestions.append("优化协调决策算法")
        if execution_consistency < 0.9:
            suggestions.append("改进执行一致性验证")
        if response_quality < 0.7:
            suggestions.append("提升响应质量和指导价值")
        if efficiency_score < 0.7:
            suggestions.append("优化处理效率")

        metrics = SystemPerformanceMetrics(
            coordination_accuracy=coordination_accuracy,
            execution_consistency=execution_consistency,
            response_quality=response_quality,
            efficiency_score=efficiency_score,
            processing_time=processing_time,
            cost_effectiveness=1.0,  # Default cost effectiveness for Agno Evals version
            overall_score=overall_score,
            improvement_suggestions=suggestions
        )

        # Store for historical analysis
        self.performance_history.append(metrics)

        logger.info(f"🎯 Overall system quality: {overall_score:.2f}")
        if overall_score < 0.7:
            logger.warning(f"⚠️  Quality below threshold, suggestions: {suggestions[:2]}")

        return metrics

    def _evaluate_execution_consistency(self, plan: CoordinationPlan, execution_log: Dict[str, Any]) -> float:
        """Evaluate execution consistency (rule-based)."""
        score = 1.0

        # Check execution mode consistency
        planned_mode = plan.execution_mode.value
        actual_mode = execution_log.get("execution_mode", "unknown")
        if planned_mode != actual_mode:
            score -= 0.3

        # Check specialist count
        planned_specialists = len(plan.specialist_roles)
        actual_specialists = execution_log.get("actual_specialists", 0)
        if abs(planned_specialists - actual_specialists) > 1:
            score -= 0.2

        # Check timeout compliance
        planned_timeout = plan.timeout_seconds
        actual_time = execution_log.get("processing_time", 0)
        if actual_time > planned_timeout * 1.2:
            score -= 0.1

        return max(0.0, score)

    def _calculate_efficiency(self, plan: CoordinationPlan, processing_time: float) -> float:
        """Calculate processing efficiency."""
        expected_time = plan.timeout_seconds * 0.6  # Expect to use ~60% of timeout
        if processing_time <= expected_time:
            return 1.0
        elif processing_time <= plan.timeout_seconds:
            return 1.0 - (processing_time - expected_time) / (plan.timeout_seconds - expected_time) * 0.5
        else:
            return 0.0

    def get_performance_trends(self) -> Dict[str, float]:
        """Get performance trends over recent evaluations."""
        if len(self.performance_history) < 2:
            return {}

        recent = self.performance_history[-5:]  # Last 5 evaluations

        return {
            "avg_coordination_accuracy": sum(m.coordination_accuracy for m in recent) / len(recent),
            "avg_execution_consistency": sum(m.execution_consistency for m in recent) / len(recent),
            "avg_response_quality": sum(m.response_quality for m in recent) / len(recent),
            "avg_overall_score": sum(m.overall_score for m in recent) / len(recent),
            "trend_direction": "improving" if recent[-1].overall_score > recent[0].overall_score else "declining"
        }

    def analyze_coordination_effectiveness(self) -> Dict[str, Any]:
        """Analyze coordination effectiveness patterns and provide optimization insights."""
        if len(self.performance_history) < 3:
            return {"status": "insufficient_data", "message": "需要至少3次评估数据来分析协调效果"}

        # Strategy performance analysis
        strategy_performance = {}
        complexity_performance = {}

        for metrics in self.performance_history:
            # Extract strategy from suggestions (simplified approach)
            strategy = "unknown"
            if any("单一" in s for s in metrics.improvement_suggestions):
                strategy = "single_agent"
            elif any("混合" in s for s in metrics.improvement_suggestions):
                strategy = "hybrid"
            elif any("多代理" in s for s in metrics.improvement_suggestions):
                strategy = "multi_agent"

            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(metrics.overall_score)

        # Identify best performing strategies
        best_strategies = []
        for strategy, scores in strategy_performance.items():
            if len(scores) >= 2:
                avg_score = sum(scores) / len(scores)
                best_strategies.append((strategy, avg_score, len(scores)))

        best_strategies.sort(key=lambda x: x[1], reverse=True)

        # Performance trend analysis
        recent_10 = self.performance_history[-10:]
        coordination_trend = self._calculate_trend([m.coordination_accuracy for m in recent_10])
        quality_trend = self._calculate_trend([m.response_quality for m in recent_10])
        efficiency_trend = self._calculate_trend([m.efficiency_score for m in recent_10])

        # Common issues identification
        common_issues = []
        low_coordination_count = sum(1 for m in recent_10 if m.coordination_accuracy < 0.7)
        low_quality_count = sum(1 for m in recent_10 if m.response_quality < 0.7)
        low_efficiency_count = sum(1 for m in recent_10 if m.efficiency_score < 0.7)

        if low_coordination_count >= len(recent_10) * 0.5:
            common_issues.append("协调决策准确性持续偏低")
        if low_quality_count >= len(recent_10) * 0.5:
            common_issues.append("响应质量持续不足")
        if low_efficiency_count >= len(recent_10) * 0.5:
            common_issues.append("处理效率需要改进")

        # Optimization recommendations
        recommendations = []
        if coordination_trend < -0.1:
            recommendations.append("协调算法需要优化 - 考虑调整复杂度评估模型")
        if quality_trend < -0.1:
            recommendations.append("响应质量下降 - 建议增强中文思维评估标准")
        if efficiency_trend < -0.1:
            recommendations.append("效率下降 - 考虑优化超时设置和重试策略")

        if best_strategies:
            best_strategy, best_score, count = best_strategies[0]
            recommendations.append(f"推荐更多使用 {best_strategy} 策略 (平均分数: {best_score:.2f})")

        return {
            "status": "analysis_complete",
            "total_evaluations": len(self.performance_history),
            "recent_average_score": sum(m.overall_score for m in recent_10) / len(recent_10),
            "strategy_performance": {
                strategy: {
                    "average_score": sum(scores) / len(scores),
                    "evaluation_count": len(scores),
                    "success_rate": sum(1 for s in scores if s >= 0.7) / len(scores)
                }
                for strategy, scores in strategy_performance.items() if len(scores) >= 2
            },
            "performance_trends": {
                "coordination_accuracy": coordination_trend,
                "response_quality": quality_trend,
                "efficiency": efficiency_trend
            },
            "common_issues": common_issues,
            "optimization_recommendations": recommendations,
            "top_performing_strategies": [
                {"strategy": s[0], "score": s[1], "count": s[2]}
                for s in best_strategies[:3]
            ]
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction for a series of values (-1 to 1)."""
        if len(values) < 3:
            return 0.0

        # Simple linear trend calculation
        n = len(values)
        x_avg = (n - 1) / 2
        y_avg = sum(values) / n

        numerator = sum((i - x_avg) * (values[i] - y_avg) for i in range(n))
        denominator = sum((i - x_avg) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        slope = numerator / denominator
        # Normalize slope to [-1, 1] range
        return max(-1.0, min(1.0, slope * 2))

    def generate_coordination_report(self) -> str:
        """Generate a comprehensive coordination effectiveness report in Chinese."""
        analysis = self.analyze_coordination_effectiveness()

        if analysis["status"] == "insufficient_data":
            return f"📊 协调效果报告\n\n{analysis['message']}"

        report = f"""📊 协调效果分析报告

🎯 总体表现:
- 总评估次数: {analysis['total_evaluations']}
- 近期平均分数: {analysis['recent_average_score']:.2f}

📈 性能趋势:
- 协调准确性: {'📈 改善' if analysis['performance_trends']['coordination_accuracy'] > 0.1 else '📉 下降' if analysis['performance_trends']['coordination_accuracy'] < -0.1 else '➡️ 稳定'}
- 响应质量: {'📈 改善' if analysis['performance_trends']['response_quality'] > 0.1 else '📉 下降' if analysis['performance_trends']['response_quality'] < -0.1 else '➡️ 稳定'}
- 处理效率: {'📈 改善' if analysis['performance_trends']['efficiency'] > 0.1 else '📉 下降' if analysis['performance_trends']['efficiency'] < -0.1 else '➡️ 稳定'}

🏆 策略表现排名:"""

        for i, strategy_info in enumerate(analysis.get('top_performing_strategies', []), 1):
            report += f"\n{i}. {strategy_info['strategy']}: {strategy_info['score']:.2f} ({strategy_info['count']}次)"

        if analysis.get('common_issues'):
            report += f"\n\n⚠️ 常见问题:"
            for issue in analysis['common_issues']:
                report += f"\n- {issue}"

        if analysis.get('optimization_recommendations'):
            report += f"\n\n💡 优化建议:"
            for rec in analysis['optimization_recommendations']:
                report += f"\n- {rec}"

        report += f"\n\n📝 报告生成时间: {len(self.performance_history)}次评估后"

        return report


def create_quality_assurance_manager() -> QualityAssuranceManager:
    """Create quality assurance manager using Agno Evals."""
    return QualityAssuranceManager()