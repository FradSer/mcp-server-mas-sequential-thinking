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
from .types import ExecutionMode
from .modernized_config import get_model_config

logger = logging.getLogger(__name__)


@dataclass
class SystemPerformanceMetrics:
    """Simplified system performance metrics."""

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
            input_text = f"""
分析以下中文思维处理的协调决策是否合理：

原始思维: "{thought.thought}"
思维上下文: 第{thought.thought_number}个思维，共{thought.total_thoughts}个

协调决策:
- 策略: {plan.strategy.value}
- 复杂度: {plan.complexity_level.value} (评分: {plan.complexity_score:.1f}/100)
- 执行模式: {plan.execution_mode.value}
- 专家组合: {plan.specialist_roles}

实际响应: "{response[:200]}..."

请评估这个协调决策的准确性和合理性。
"""

            expected_output = f"""
基于思维复杂度评估，协调决策{'合理' if plan.confidence > 0.5 else '需要改进'}：
1. 复杂度评估准确
2. 策略选择合理
3. 专家组合适当
4. 执行模式一致
"""

            agent_instructions = [
                "你是中文思维处理协调决策的质量评估专家。",
                "评估协调决策是否合理，包括复杂度评估、策略选择、专家组合等。",
                "特别关注中文思维的语义复杂度和文化内涵。",
                "用简洁清晰的中文回答。"
            ]

        else:
            input_text = f"""
Analyze whether the coordination decision for this thought processing is appropriate:

ORIGINAL THOUGHT: "{thought.thought}"
THOUGHT CONTEXT: #{thought.thought_number}/{thought.total_thoughts}

COORDINATION DECISION:
- Strategy: {plan.strategy.value}
- Complexity: {plan.complexity_level.value} (score: {plan.complexity_score:.1f}/100)
- Execution Mode: {plan.execution_mode.value}
- Specialists: {plan.specialist_roles}

ACTUAL RESPONSE: "{response[:200]}..."

Evaluate the accuracy and appropriateness of this coordination decision.
"""

            expected_output = f"""
Based on thought complexity, coordination decision is {'reasonable' if plan.confidence > 0.5 else 'needs improvement'}:
1. Complexity assessment is accurate
2. Strategy selection is appropriate
3. Specialist combination is suitable
4. Execution mode matches strategy
"""

            agent_instructions = [
                "You are an expert evaluator of coordination decisions for thought processing.",
                "Assess whether coordination decisions are reasonable, including complexity assessment, strategy selection, and specialist combinations.",
                "Focus on content complexity analysis and strategic appropriateness.",
                "Answer clearly and concisely."
            ]

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
            num_iterations=1
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

        # Detect if content is primarily Chinese
        chinese_chars = len([c for c in thought.thought if '\u4e00' <= c <= '\u9fff'])
        is_chinese_content = chinese_chars > len(thought.thought) * 0.3

        if is_chinese_content:
            input_text = f"""
评估以下中文思维处理响应的质量：

原始思维: "{thought.thought}"
处理响应: "{response}"

评估标准：
1. 是否充分回应了原始思维的内容？
2. 是否提供了有价值的分析或指导？
3. 是否包含推进后续思维的指导？
4. 内容组织是否清晰有序？
"""

            expected_output = f"""
中文思维响应质量评估：
1. 内容相关性：{'高' if len(response) > 100 else '中等'}
2. 分析深度：基于中文哲学思维的深度分析
3. 指导价值：{'包含明确指导' if 'guidance' in response.lower() or '指导' in response else '缺少指导'}
4. 结构清晰：{'良好' if any(marker in response for marker in ['#', '*', '-', '1.', '2.']) else '需要改进'}

总体质量：{'优秀' if len(response) > 200 else '良好'}
"""

            agent_instructions = [
                "你是中文思维处理响应质量的评估专家。",
                "评估响应是否有效推进Sequential Thinking流程。",
                "关注内容质量、指导价值和思维链条的连续性。",
                "特别关注中文哲学思维的分析深度和文化内涵。",
                "提供具体改进建议。"
            ]

        else:
            input_text = f"""
Evaluate the quality of this thought processing response:

Original Thought: "{thought.thought}"
Processing Response: "{response}"

Evaluation criteria:
1. Does it adequately address the original thought content?
2. Does it provide valuable analysis or guidance?
3. Does it include guidance for subsequent thinking?
4. Is the content clearly organized?
"""

            expected_output = f"""
Response quality assessment:
1. Content relevance: {'high' if len(response) > 100 else 'medium'}
2. Analysis depth: Deep philosophical analysis
3. Guidance value: {'Contains clear guidance' if 'guidance' in response.lower() else 'Lacks guidance'}
4. Structure clarity: {'Good' if any(marker in response for marker in ['#', '*', '-', '1.']) else 'Needs improvement'}

Overall quality: {'Excellent' if len(response) > 200 else 'Good'}
"""

            agent_instructions = [
                "You are a thought processing response quality evaluator.",
                "Assess whether responses effectively advance the Sequential Thinking process.",
                "Focus on content quality, guidance value, and thinking chain continuity.",
                "Provide specific improvement suggestions."
            ]

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
            num_iterations=1
        )


class QualityAssuranceManager:
    """Simplified quality assurance manager using Agno Evals."""

    def __init__(self):
        self.coordination_evaluator = CoordinationAccuracyEvaluator()
        self.quality_evaluator = ResponseQualityEvaluator()

    async def evaluate_full_pipeline(
        self,
        thought: ThoughtData,
        plan: CoordinationPlan,
        execution_log: Dict[str, Any],
        response: str,
        processing_time: float
    ) -> SystemPerformanceMetrics:
        """Evaluate the complete thought processing pipeline using Agno Evals."""

        logger.info("🔍 Running quality evaluation...")

        # Initialize scores
        coordination_accuracy = 0.5
        response_quality = 0.5
        execution_consistency = 1.0

        # Run coordination accuracy evaluation
        try:
            coord_eval = self.coordination_evaluator.create_coordination_evaluation(
                thought, plan, response
            )
            coord_result: Optional[AccuracyResult] = coord_eval.run(print_results=False)

            if coord_result and coord_result.avg_score is not None:
                coordination_accuracy = coord_result.avg_score / 10.0  # Convert 0-10 to 0-1
                logger.info(f"  ✅ Coordination accuracy: {coordination_accuracy:.2f}")
        except Exception as e:
            logger.warning(f"Coordination evaluation error: {e}")

        # Run response quality evaluation
        try:
            quality_eval = self.quality_evaluator.create_quality_evaluation(
                thought, response
            )
            quality_result: Optional[AccuracyResult] = quality_eval.run(print_results=False)

            if quality_result and quality_result.avg_score is not None:
                response_quality = quality_result.avg_score / 10.0  # Convert 0-10 to 0-1
                logger.info(f"  ✅ Response quality: {response_quality:.2f}")
        except Exception as e:
            logger.warning(f"Quality evaluation error: {e}")

        # Evaluate execution consistency (rule-based)
        execution_consistency = self._evaluate_execution_consistency(plan, execution_log)

        # Calculate efficiency
        efficiency_score = self._calculate_efficiency(plan, processing_time)

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
            cost_effectiveness=1.0,
            overall_score=overall_score,
            improvement_suggestions=suggestions
        )

        logger.info(f"🎯 Overall system quality: {overall_score:.2f}")
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

        return max(0.0, score)

    def _calculate_efficiency(self, plan: CoordinationPlan, processing_time: float) -> float:
        """Calculate processing efficiency based on complexity and actual time."""
        complexity_multipliers = {
            "simple": 30.0,
            "moderate": 90.0,
            "complex": 180.0,
            "highly_complex": 300.0
        }

        expected_time = complexity_multipliers.get(
            plan.complexity_level.value, 120.0
        )

        if processing_time <= expected_time:
            return 1.0
        elif processing_time <= expected_time * 2:
            return 1.0 - (processing_time - expected_time) / expected_time * 0.5
        else:
            return 0.5


def create_quality_assurance_manager() -> QualityAssuranceManager:
    """Create quality assurance manager using Agno Evals."""
    return QualityAssuranceManager()