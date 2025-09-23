"""Six Thinking Hats Sequential Processor.

实现基于 De Bono 六帽思维的顺序处理器，集成到 Agno Workflow 系统中。
支持单帽到六帽的智能序列处理，解决"综合+评审"分离问题。
"""

# Lazy import to break circular dependency
import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

from agno.workflow.types import StepOutput

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config
from mcp_server_mas_sequential_thinking.routing.six_hats_router import (
    RoutingDecision,
    SixHatsIntelligentRouter,
)

from .six_hats_core import HatColor, HatComplexity, SixHatsAgentFactory

# logger already defined above


@dataclass
class SixHatsProcessingResult:
    """六帽处理结果."""

    content: str
    strategy_used: str
    hat_sequence: list[str]
    processing_time: float
    complexity_score: float
    cost_reduction: float
    individual_results: dict[str, str]  # 每个帽子的结果
    step_name: str


class SixHatsSequentialProcessor:
    """六帽顺序思维处理器."""

    def __init__(self) -> None:
        self.model_config = get_model_config()
        self.hat_factory = SixHatsAgentFactory()
        self.router = SixHatsIntelligentRouter()

    async def process_thought_with_six_hats(
        self, thought_data: "ThoughtData", context_prompt: str = ""
    ) -> SixHatsProcessingResult:
        """使用六帽思维处理思想."""
        start_time = time.time()

        logger.info("🎩 SIX HATS PROCESSING START:")
        logger.info(f"  📝 Input: {thought_data.thought[:100]}...")
        logger.info(f"  📋 Context: {len(context_prompt)} chars")

        try:
            # 步骤1: 智能路由决策
            routing_decision = await self.router.route_thought(thought_data)

            logger.info(f"  🎯 Strategy: {routing_decision.strategy.name}")
            logger.info(
                f"  🎨 Hat Sequence: {[hat.value for hat in routing_decision.strategy.hat_sequence]}"
            )

            # 步骤2: 根据复杂度执行相应处理
            if routing_decision.strategy.complexity == HatComplexity.SINGLE:
                result = await self._process_single_hat(
                    thought_data, context_prompt, routing_decision
                )
            elif routing_decision.strategy.complexity == HatComplexity.DOUBLE:
                result = await self._process_double_hat_sequence(
                    thought_data, context_prompt, routing_decision
                )
            elif routing_decision.strategy.complexity == HatComplexity.TRIPLE:
                result = await self._process_triple_hat_sequence(
                    thought_data, context_prompt, routing_decision
                )
            else:  # FULL
                result = await self._process_full_hat_sequence(
                    thought_data, context_prompt, routing_decision
                )

            processing_time = time.time() - start_time

            # 创建最终结果
            final_result = SixHatsProcessingResult(
                content=result["final_content"],
                strategy_used=routing_decision.strategy.name,
                hat_sequence=[
                    hat.value for hat in routing_decision.strategy.hat_sequence
                ],
                processing_time=processing_time,
                complexity_score=routing_decision.complexity_metrics.complexity_score,
                cost_reduction=routing_decision.estimated_cost_reduction,
                individual_results=result.get("individual_results", {}),
                step_name="six_hats_processing",
            )

            logger.info("✅ SIX HATS PROCESSING COMPLETED:")
            logger.info(f"  ⏱️  Time: {processing_time:.3f}s")
            logger.info(
                f"  💰 Cost Reduction: {routing_decision.estimated_cost_reduction:.1f}%"
            )
            logger.info(f"  📊 Output Length: {len(final_result.content)} chars")

            return final_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(
                f"❌ Six Hats processing failed after {processing_time:.3f}s: {e}"
            )

            return SixHatsProcessingResult(
                content=f"Six Hats processing failed: {e!s}",
                strategy_used="error_fallback",
                hat_sequence=[],
                processing_time=processing_time,
                complexity_score=0.0,
                cost_reduction=0.0,
                individual_results={},
                step_name="error_handling",
            )

    async def _process_single_hat(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """处理单帽模式."""
        hat_color = decision.strategy.hat_sequence[0]
        logger.info(f"  🎩 SINGLE HAT MODE: {hat_color.value}")

        # 为蓝帽子使用增强模型，其他帽子使用标准模型
        if hat_color == HatColor.BLUE:
            model = self.model_config.create_enhanced_model()
            logger.info("    🚀 Using enhanced model for Blue Hat synthesis")
        else:
            model = self.model_config.create_standard_model()
            logger.info("    📎 Using standard model for individual hat processing")

        agent = self.hat_factory.create_hat_agent(hat_color, model, context, {})

        # 执行处理
        result = await agent.arun(input=thought_data.thought)

        # 提取内容
        content = self._extract_content(result)

        return {
            "final_content": content,
            "individual_results": {hat_color.value: content},
        }

    async def _process_double_hat_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """处理双帽序列."""
        hat1, hat2 = decision.strategy.hat_sequence
        logger.info(f"  🎩 DOUBLE HAT SEQUENCE: {hat1.value} → {hat2.value}")

        individual_results = {}

        # 第一顶帽子
        if hat1 == HatColor.BLUE:
            model1 = self.model_config.create_enhanced_model()
            logger.info(f"    🚀 Using enhanced model for {hat1.value} hat")
        else:
            model1 = self.model_config.create_standard_model()
            logger.info(f"    📎 Using standard model for {hat1.value} hat")

        agent1 = self.hat_factory.create_hat_agent(hat1, model1, context, {})
        result1 = await agent1.arun(input=thought_data.thought)
        content1 = self._extract_content(result1)
        individual_results[hat1.value] = content1

        logger.info(f"    ✅ {hat1.value} hat completed")

        # 第二顶帽子（基于第一顶帽子的结果）
        if hat2 == HatColor.BLUE:
            model2 = self.model_config.create_enhanced_model()
            logger.info(f"    🚀 Using enhanced model for {hat2.value} hat synthesis")
        else:
            model2 = self.model_config.create_standard_model()
            logger.info(f"    📎 Using standard model for {hat2.value} hat")

        previous_results = {hat1.value: content1}
        agent2 = self.hat_factory.create_hat_agent(
            hat2, model2, context, previous_results
        )

        # 构建第二帽子的输入
        hat2_input = f"""
Original thought: {thought_data.thought}

{hat1.value.title()} hat perspective: {content1}

Now apply {hat2.value} hat thinking to this.
"""

        result2 = await agent2.arun(input=hat2_input)
        content2 = self._extract_content(result2)
        individual_results[hat2.value] = content2

        logger.info(f"    ✅ {hat2.value} hat completed")

        # 如果最后一个是蓝帽，直接使用其结果作为最终回答
        if hat2 == HatColor.BLUE:
            final_content = content2
        else:
            # 否则，组合结果
            final_content = self._combine_dual_hat_results(
                hat1, content1, hat2, content2, thought_data.thought
            )

        return {
            "final_content": final_content,
            "individual_results": individual_results,
        }

    async def _process_triple_hat_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """处理三帽序列（核心模式）."""
        hat_sequence = decision.strategy.hat_sequence
        logger.info(
            f"  🎩 TRIPLE HAT SEQUENCE: {' → '.join(hat.value for hat in hat_sequence)}"
        )

        individual_results = {}
        previous_results = {}

        # 顺序执行三个帽子
        for i, hat_color in enumerate(hat_sequence):
            logger.info(f"    🎩 Processing {hat_color.value} hat ({i + 1}/3)")

            # 为蓝帽子使用增强模型，其他帽子使用标准模型
            if hat_color == HatColor.BLUE:
                model = self.model_config.create_enhanced_model()
                logger.info(f"      🚀 Using enhanced model for {hat_color.value} hat synthesis")
            else:
                model = self.model_config.create_standard_model()
                logger.info(f"      📎 Using standard model for {hat_color.value} hat")

            agent = self.hat_factory.create_hat_agent(
                hat_color, model, context, previous_results
            )

            # 构建输入
            if i == 0:
                # 第一个帽子：原始输入
                hat_input = thought_data.thought
            else:
                # 后续帽子：包含之前的结果
                hat_input = self._build_sequential_input(
                    thought_data.thought, previous_results, hat_color
                )

            result = await agent.arun(input=hat_input)
            content = self._extract_content(result)
            individual_results[hat_color.value] = content
            previous_results[hat_color.value] = content

            logger.info(f"      ✅ {hat_color.value} hat completed")

        # 如果最后一个是蓝帽，它的结果就是最终结果
        if hat_sequence[-1] == HatColor.BLUE:
            final_content = individual_results[HatColor.BLUE.value]
        else:
            # 否则，创建简单的综合
            final_content = self._synthesize_triple_results(
                individual_results, hat_sequence, thought_data.thought
            )

        return {
            "final_content": final_content,
            "individual_results": individual_results,
        }

    async def _process_full_hat_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """处理完整六帽序列."""
        hat_sequence = decision.strategy.hat_sequence
        logger.info(
            f"  🎩 FULL HAT SEQUENCE: {' → '.join(hat.value for hat in hat_sequence)}"
        )

        individual_results = {}
        previous_results = {}

        # 顺序执行所有帽子
        for i, hat_color in enumerate(hat_sequence):
            logger.info(
                f"    🎩 Processing {hat_color.value} hat ({i + 1}/{len(hat_sequence)})"
            )

            # 为蓝帽子使用增强模型，其他帽子使用标准模型
            if hat_color == HatColor.BLUE:
                model = self.model_config.create_enhanced_model()
                logger.info(f"      🚀 Using enhanced model for {hat_color.value} hat synthesis")
            else:
                model = self.model_config.create_standard_model()
                logger.info(f"      📎 Using standard model for {hat_color.value} hat")

            agent = self.hat_factory.create_hat_agent(
                hat_color, model, context, previous_results
            )

            # 构建输入
            if i == 0 or hat_color == HatColor.BLUE:
                # 第一个帽子和蓝帽：特殊处理
                if hat_color == HatColor.BLUE and i > 0:
                    # 蓝帽整合所有前面的结果
                    hat_input = self._build_blue_hat_integration_input(
                        thought_data.thought, previous_results
                    )
                else:
                    hat_input = thought_data.thought
            else:
                # 其他帽子：顺序处理
                hat_input = self._build_sequential_input(
                    thought_data.thought, previous_results, hat_color
                )

            result = await agent.arun(input=hat_input)
            content = self._extract_content(result)
            individual_results[hat_color.value] = content
            previous_results[hat_color.value] = content

            logger.info(f"      ✅ {hat_color.value} hat completed")

        # 最终蓝帽的结果就是最终结果
        final_blue_result = None
        for hat_color in reversed(hat_sequence):
            if hat_color == HatColor.BLUE:
                final_blue_result = individual_results[hat_color.value]
                break

        final_content = final_blue_result or self._synthesize_full_results(
            individual_results, hat_sequence, thought_data.thought
        )

        return {
            "final_content": final_content,
            "individual_results": individual_results,
        }

    def _extract_content(self, result: Any) -> str:
        """提取agent运行结果的内容."""
        if hasattr(result, "content"):
            content = result.content
            if isinstance(content, str):
                return content.strip()
            return str(content).strip()
        if isinstance(result, str):
            return result.strip()
        return str(result).strip()

    def _build_sequential_input(
        self,
        original_thought: str,
        previous_results: dict[str, str],
        current_hat: HatColor,
    ) -> str:
        """构建顺序处理的输入."""
        input_parts = [f"Original thought: {original_thought}", ""]

        if previous_results:
            input_parts.append("Previous hat perspectives:")
            for hat_name, content in previous_results.items():
                input_parts.append(
                    f"  {hat_name.title()} hat: {content[:200]}{'...' if len(content) > 200 else ''}"
                )
            input_parts.append("")

        input_parts.append(
            f"Now apply {current_hat.value} hat thinking to this situation."
        )

        return "\n".join(input_parts)

    def _build_blue_hat_integration_input(
        self, original_thought: str, all_results: dict[str, str]
    ) -> str:
        """构建蓝帽整合输入."""
        input_parts = [
            f"Original question: {original_thought}",
            "",
            "Collected perspectives from different thinking modes:",
        ]

        for hat_name, content in all_results.items():
            if hat_name != "blue":  # 避免包含之前的蓝帽结果
                input_parts.append(f"• {hat_name.title()} perspective: {content}")

        input_parts.extend(
            [
                "",
                "TASK: As the Blue Hat (metacognitive thinking), synthesize all perspectives into ONE comprehensive, unified answer.",
                "REQUIREMENTS:",
                "1. Provide a single, coherent response directly addressing the original question",
                "2. Integrate insights from all perspectives naturally",
                "3. Do NOT list or separate different 'hat' perspectives in your response",
                "4. Do NOT use section headers like 'White Hat says' or 'Red Hat perspective'",
                "5. Write as a unified voice providing the final answer",
                "6. This will be the ONLY response the user sees - make it complete and standalone",
            ]
        )

        return "\n".join(input_parts)

    def _combine_dual_hat_results(
        self,
        hat1: HatColor,
        content1: str,
        hat2: HatColor,
        content2: str,
        original_thought: str,
    ) -> str:
        """组合双帽结果."""
        # 如果第二个是蓝帽，直接返回其结果（应该已经是综合的）
        if hat2 == HatColor.BLUE:
            return content2

        # 否则创建简单的综合回答
        if hat1 == HatColor.WHITE and hat2 == HatColor.RED:
            return f"Regarding '{original_thought}': The factual analysis shows {content1.lower()} while emotionally, {content2.lower()} These perspectives combine to suggest a balanced approach that considers both objective realities and human responses."
        elif hat1 == HatColor.BLACK and hat2 == HatColor.YELLOW:
            return f"Considering '{original_thought}': While there are important concerns to address - {content1.lower().strip('.')} - there are also significant opportunities - {content2.lower()} Taking both into account leads to a measured approach that manages risks while pursuing benefits."
        else:
            # 通用综合
            return f"Analyzing '{original_thought}': Drawing from different thinking approaches, we can see that {content1.lower().strip('.')} Additionally, {content2.lower()} Integrating these insights provides a comprehensive understanding of the situation."

    def _synthesize_triple_results(
        self,
        results: dict[str, str],
        hat_sequence: list[HatColor],
        original_thought: str,
    ) -> str:
        """综合三帽结果."""
        # 尝试创建一个真正综合的回答，而不是列出各个帽子
        content_pieces = []
        for hat_color in hat_sequence:
            hat_name = hat_color.value
            content = results.get(hat_name, "")
            if content:
                # 提取核心见解，而不是原始帽子输出
                clean_content = content.strip().rstrip('.!')
                content_pieces.append(clean_content)

        if len(content_pieces) >= 3:
            # 三个或更多视角的综合
            return f"""Considering the question '{original_thought}', a comprehensive analysis reveals several key insights.

{content_pieces[0].lower()}, which provides the foundational understanding. Building on this, {content_pieces[1].lower()}, adding important nuance to our perspective. Furthermore, {content_pieces[2].lower() if len(content_pieces) > 2 else ''}

Integrating these multiple thinking approaches, the answer emerges as a balanced synthesis that acknowledges the complexity while providing clear direction."""
        elif len(content_pieces) == 2:
            return f"Addressing '{original_thought}': {content_pieces[0].lower()}, and additionally, {content_pieces[1].lower()} Together, these insights provide a well-rounded understanding."
        elif len(content_pieces) == 1:
            return f"Regarding '{original_thought}': {content_pieces[0]}"
        else:
            return f"After thorough consideration of '{original_thought}', a comprehensive analysis across multiple thinking perspectives suggests this requires further exploration."

    def _synthesize_full_results(
        self,
        results: dict[str, str],
        hat_sequence: list[HatColor],
        original_thought: str,
    ) -> str:
        """综合完整六帽结果."""
        # 如果有蓝帽结果，优先使用
        blue_result = results.get("blue")
        if blue_result:
            return blue_result

        # 否则创建综合
        return self._synthesize_triple_results(results, hat_sequence, original_thought)

    def _get_hat_contribution(self, hat_color: HatColor) -> str:
        """获取帽子的贡献描述."""
        contributions = {
            HatColor.WHITE: "factual information and objective data",
            HatColor.RED: "emotional insights and intuitive responses",
            HatColor.BLACK: "critical analysis and risk assessment",
            HatColor.YELLOW: "positive possibilities and value identification",
            HatColor.GREEN: "creative alternatives and innovative solutions",
            HatColor.BLUE: "process management and integrated thinking",
        }
        return contributions.get(hat_color, "specialized thinking")


# 创建全局处理器实例
_six_hats_processor = SixHatsSequentialProcessor()


# 便利函数
async def process_with_six_hats(
    thought_data: "ThoughtData", context: str = ""
) -> SixHatsProcessingResult:
    """使用六帽思维处理思想的便利函数."""
    return await _six_hats_processor.process_thought_with_six_hats(
        thought_data, context
    )


def create_six_hats_step_output(result: SixHatsProcessingResult) -> StepOutput:
    """将六帽处理结果转换为 Agno StepOutput."""
    return StepOutput(
        content=result.content,
        success=True,
        step_name=result.step_name,
    )
