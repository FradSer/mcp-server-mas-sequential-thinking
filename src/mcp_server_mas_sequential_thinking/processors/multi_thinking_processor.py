"""Multi-Thinking Sequential Processor.

Implements a parallel processor based on multi-directional thinking methodology,
integrated with the Agno Workflow system. Supports intelligent parallel processing
from single direction to full multi-direction analysis.
"""

# Lazy import to break circular dependency
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agno.workflow.types import StepOutput

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData

logger = logging.getLogger(__name__)
from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config
from mcp_server_mas_sequential_thinking.routing.multi_thinking_router import (
    MultiThinkingIntelligentRouter,
    RoutingDecision,
)

from .multi_thinking_core import (
    MultiThinkingAgentFactory,
    ProcessingDepth,
    ThinkingDirection,
)

# logger already defined above


@dataclass
class MultiThinkingProcessingResult:
    """Multi-thinking processing result."""

    content: str
    strategy_used: str
    thinking_sequence: list[str]
    processing_time: float
    complexity_score: float
    cost_reduction: float
    individual_results: dict[str, str]  # Results from each thinking direction
    step_name: str


class MultiThinkingSequentialProcessor:
    """Multi-thinking parallel processor."""

    def __init__(self) -> None:
        self.model_config = get_model_config()
        self.thinking_factory = MultiThinkingAgentFactory()
        self.router = MultiThinkingIntelligentRouter()

    async def process_with_multi_thinking(
        self, thought_data: "ThoughtData", context_prompt: str = ""
    ) -> MultiThinkingProcessingResult:
        """Process thoughts using multi-thinking methodology with parallel execution."""
        start_time = time.time()

        logger.info("Multi-thinking processing started")
        if logger.isEnabledFor(logging.INFO):
            logger.info("Input preview: %s", thought_data.thought[:100])
            logger.info("Context length: %d chars", len(context_prompt))

        try:
            # Step 1: Intelligent routing decision
            routing_decision = await self.router.route_thought(thought_data)

            logger.info("Selected strategy: %s", routing_decision.strategy.name)
            if logger.isEnabledFor(logging.INFO):
                sequence = [
                    direction.value
                    for direction in routing_decision.strategy.thinking_sequence
                ]
                logger.info("Thinking sequence: %s", sequence)

            # Step 2: Execute processing based on complexity
            if routing_decision.strategy.complexity == ProcessingDepth.SINGLE:
                result = await self._process_single_direction(
                    thought_data, context_prompt, routing_decision
                )
            elif routing_decision.strategy.complexity == ProcessingDepth.DOUBLE:
                result = await self._process_double_direction_sequence(
                    thought_data, context_prompt, routing_decision
                )
            elif routing_decision.strategy.complexity == ProcessingDepth.TRIPLE:
                result = await self._process_triple_direction_sequence(
                    thought_data, context_prompt, routing_decision
                )
            else:  # FULL
                result = await self._process_full_direction_sequence(
                    thought_data, context_prompt, routing_decision
                )

            processing_time = time.time() - start_time

            # Create final result
            final_result = MultiThinkingProcessingResult(
                content=result["final_content"],
                strategy_used=routing_decision.strategy.name,
                thinking_sequence=[
                    direction.value
                    for direction in routing_decision.strategy.thinking_sequence
                ],
                processing_time=processing_time,
                complexity_score=routing_decision.complexity_metrics.complexity_score,
                cost_reduction=routing_decision.estimated_cost_reduction,
                individual_results=result.get("individual_results", {}),
                step_name="multi_thinking_processing",
            )

            logger.info(
                "Multi-thinking processing completed - Time: %.3fs, Cost reduction: %.1f%%, Output: %d chars",
                processing_time,
                routing_decision.estimated_cost_reduction,
                len(final_result.content),
            )

            return final_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(
                f"Multi-thinking processing failed after {processing_time:.3f}s: {e}"
            )

            return MultiThinkingProcessingResult(
                content=f"Multi-thinking processing failed: {e!s}",
                strategy_used="error_fallback",
                thinking_sequence=[],
                processing_time=processing_time,
                complexity_score=0.0,
                cost_reduction=0.0,
                individual_results={},
                step_name="error_handling",
            )

    async def _process_single_direction(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """Process single thinking direction mode."""
        thinking_direction = decision.strategy.thinking_sequence[0]
        logger.info(f"  SINGLE THINKING MODE: {thinking_direction.value}")

        # Use enhanced model for synthesis thinking, standard model for other directions
        if thinking_direction == ThinkingDirection.SYNTHESIS:
            model = self.model_config.create_enhanced_model()
            logger.info("    Using enhanced model for synthesis thinking")
        else:
            model = self.model_config.create_standard_model()
            logger.info("    Using standard model for focused thinking")

        agent = self.thinking_factory.create_thinking_agent(
            thinking_direction, model, context, {}
        )

        # Execute processing
        result = await agent.arun(input=thought_data.thought)

        # Extract content
        content = self._extract_content(result)

        return {
            "final_content": content,
            "individual_results": {thinking_direction.value: content},
        }

    async def _process_double_direction_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """Process dual thinking direction sequence with parallel execution."""
        direction1, direction2 = decision.strategy.thinking_sequence
        logger.info(
            f"  DUAL THINKING SEQUENCE: {direction1.value} + {direction2.value} (parallel)"
        )

        individual_results = {}

        # Check if synthesis agent is involved
        has_synthesis = any(d == ThinkingDirection.SYNTHESIS for d in [direction1, direction2])

        if has_synthesis:
            # If synthesis is involved, run non-synthesis agents in parallel, then synthesis
            non_synthesis_directions = [d for d in [direction1, direction2] if d != ThinkingDirection.SYNTHESIS]
            synthesis_direction = ThinkingDirection.SYNTHESIS

            # Run non-synthesis agents in parallel
            import asyncio
            tasks = []
            for direction in non_synthesis_directions:
                model = self.model_config.create_standard_model()
                logger.info(f"    Using standard model for {direction.value} thinking (parallel)")

                agent = self.thinking_factory.create_thinking_agent(
                    direction, model, context, {}
                )
                task = agent.arun(input=thought_data.thought)
                tasks.append((direction, task))

            # Execute parallel tasks
            logger.info(f"    Executing {len(tasks)} thinking agents in parallel")
            parallel_results = await asyncio.gather(*[task for _, task in tasks])

            # Process parallel results
            for (direction, _), result in zip(tasks, parallel_results, strict=False):
                content = self._extract_content(result)
                individual_results[direction.value] = content
                logger.info(f"    {direction.value} thinking completed (parallel)")

            # Run synthesis agent with all parallel results
            model = self.model_config.create_enhanced_model()
            logger.info(f"    Using enhanced model for {synthesis_direction.value} synthesis")

            synthesis_agent = self.thinking_factory.create_thinking_agent(
                synthesis_direction, model, context, individual_results
            )

            # Build synthesis input
            synthesis_input = self._build_synthesis_integration_input(
                thought_data.thought, individual_results
            )

            synthesis_result = await synthesis_agent.arun(input=synthesis_input)
            synthesis_content = self._extract_content(synthesis_result)
            individual_results[synthesis_direction.value] = synthesis_content

            logger.info(f"    {synthesis_direction.value} thinking completed")

            final_content = synthesis_content
        else:
            # No synthesis agent - run both agents in parallel
            import asyncio
            tasks = []

            for direction in [direction1, direction2]:
                model = self.model_config.create_standard_model()
                logger.info(f"    Using standard model for {direction.value} thinking (parallel)")

                agent = self.thinking_factory.create_thinking_agent(
                    direction, model, context, {}
                )
                task = agent.arun(input=thought_data.thought)
                tasks.append((direction, task))

            # Execute parallel tasks
            logger.info("    Executing 2 thinking agents in parallel")
            parallel_results = await asyncio.gather(*[task for _, task in tasks])

            # Process parallel results
            for (direction, _), result in zip(tasks, parallel_results, strict=False):
                content = self._extract_content(result)
                individual_results[direction.value] = content
                logger.info(f"    {direction.value} thinking completed (parallel)")

            # Combine results programmatically
            final_content = self._combine_dual_thinking_results(
                direction1, individual_results[direction1.value],
                direction2, individual_results[direction2.value],
                thought_data.thought
            )

        return {
            "final_content": final_content,
            "individual_results": individual_results,
        }

    async def _process_triple_direction_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """Process triple thinking direction sequence with parallel execution."""
        thinking_sequence = decision.strategy.thinking_sequence
        logger.info(
            f"  TRIPLE THINKING SEQUENCE: {' + '.join(direction.value for direction in thinking_sequence)} (parallel)"
        )

        individual_results = {}

        # Triple strategy currently uses FACTUAL + CREATIVE + CRITICAL - all run in parallel
        import asyncio
        tasks = []

        for thinking_direction in thinking_sequence:
            logger.info(f"    Preparing {thinking_direction.value} thinking for parallel execution")

            # All agents use standard model (no synthesis in triple strategy)
            model = self.model_config.create_standard_model()
            logger.info(f"      Using standard model for {thinking_direction.value} thinking")

            agent = self.thinking_factory.create_thinking_agent(
                thinking_direction, model, context, {}
            )

            # All agents receive original input directly (parallel processing)
            task = agent.arun(input=thought_data.thought)
            tasks.append((thinking_direction, task))

        # Execute all thinking directions in parallel
        logger.info(f"    Executing {len(tasks)} thinking agents in parallel")
        parallel_results = await asyncio.gather(*[task for _, task in tasks])

        # Process parallel results
        for (thinking_direction, _), result in zip(tasks, parallel_results, strict=False):
            content = self._extract_content(result)
            individual_results[thinking_direction.value] = content
            logger.info(f"      {thinking_direction.value} thinking completed (parallel)")

        # Create programmatic synthesis (no synthesis agent in triple strategy)
        final_content = self._synthesize_triple_thinking_results(
            individual_results, thinking_sequence, thought_data.thought
        )

        return {
            "final_content": final_content,
            "individual_results": individual_results,
        }

    async def _process_full_direction_sequence(
        self, thought_data: "ThoughtData", context: str, decision: RoutingDecision
    ) -> dict[str, Any]:
        """Process full multi-thinking direction sequence with parallel execution."""
        thinking_sequence = decision.strategy.thinking_sequence
        logger.info(
            "  FULL THINKING SEQUENCE: Initial orchestration -> Parallel processing -> Final synthesis"
        )

        individual_results = {}

        # Step 1: Initial SYNTHESIS for orchestration (if first agent is SYNTHESIS)
        if thinking_sequence[0] == ThinkingDirection.SYNTHESIS:
            logger.info("    Step 1: Initial synthesis orchestration")

            initial_synthesis_model = self.model_config.create_enhanced_model()
            logger.info("      Using enhanced model for initial orchestration")

            initial_synthesis_agent = self.thinking_factory.create_thinking_agent(
                ThinkingDirection.SYNTHESIS, initial_synthesis_model, context, {}
            )

            initial_result = await initial_synthesis_agent.arun(input=thought_data.thought)
            initial_content = self._extract_content(initial_result)
            individual_results["synthesis_initial"] = initial_content

            logger.info("      Initial orchestration completed")

        # Step 2: Identify non-synthesis agents for parallel execution
        non_synthesis_agents = [
            direction for direction in thinking_sequence
            if direction != ThinkingDirection.SYNTHESIS
        ]

        if non_synthesis_agents:
            logger.info(f"    Step 2: Parallel execution of {len(non_synthesis_agents)} thinking agents")

            import asyncio
            tasks = []

            for thinking_direction in non_synthesis_agents:
                logger.info(f"      Preparing {thinking_direction.value} thinking for parallel execution")

                model = self.model_config.create_standard_model()
                logger.info(f"        Using standard model for {thinking_direction.value} thinking")

                agent = self.thinking_factory.create_thinking_agent(
                    thinking_direction, model, context, {}
                )

                # All non-synthesis agents receive original input (parallel processing)
                task = agent.arun(input=thought_data.thought)
                tasks.append((thinking_direction, task))

            # Execute all non-synthesis agents in parallel
            logger.info(f"      Executing {len(tasks)} thinking agents in parallel")
            parallel_results = await asyncio.gather(*[task for _, task in tasks])

            # Process parallel results
            for (thinking_direction, _), result in zip(tasks, parallel_results, strict=False):
                content = self._extract_content(result)
                individual_results[thinking_direction.value] = content
                logger.info(f"        {thinking_direction.value} thinking completed (parallel)")

        # Step 3: Final SYNTHESIS for integration (if last agent is SYNTHESIS)
        final_synthesis_agents = [
            i for i, direction in enumerate(thinking_sequence)
            if direction == ThinkingDirection.SYNTHESIS and i > 0
        ]

        if final_synthesis_agents:
            logger.info("    Step 3: Final synthesis integration")

            final_synthesis_model = self.model_config.create_enhanced_model()
            logger.info("      Using enhanced model for final synthesis")

            # Remove initial synthesis from results for final integration
            integration_results = {
                k: v for k, v in individual_results.items()
                if not k.startswith("synthesis_")
            }

            final_synthesis_agent = self.thinking_factory.create_thinking_agent(
                ThinkingDirection.SYNTHESIS, final_synthesis_model, context, integration_results
            )

            # Build synthesis integration input
            synthesis_input = self._build_synthesis_integration_input(
                thought_data.thought, integration_results
            )

            final_result = await final_synthesis_agent.arun(input=synthesis_input)
            final_content = self._extract_content(final_result)
            individual_results["synthesis_final"] = final_content

            logger.info("      Final synthesis integration completed")

            # Use final synthesis result as the answer
            final_answer = final_content
        else:
            # No final synthesis - create programmatic synthesis
            final_answer = self._synthesize_full_thinking_results(
                individual_results, thinking_sequence, thought_data.thought
            )

        return {
            "final_content": final_answer,
            "individual_results": individual_results,
        }

    def _extract_content(self, result: Any) -> str:
        """Extract content from agent execution result."""
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
        current_direction: ThinkingDirection,
    ) -> str:
        """Build input for sequential processing."""
        input_parts = [f"Original thought: {original_thought}", ""]

        if previous_results:
            input_parts.append("Previous analysis perspectives:")
            for direction_name, content in previous_results.items():
                # Use generic descriptions instead of specific direction names
                perspective_name = self._get_generic_perspective_name(direction_name)
                input_parts.append(
                    f"  {perspective_name}: {content[:200]}{'...' if len(content) > 200 else ''}"
                )
            input_parts.append("")

        # Use generic instruction instead of direction-specific instruction
        thinking_style = self._get_thinking_style_instruction(current_direction)
        input_parts.append(f"Now analyze this from a {thinking_style} perspective.")

        return "\n".join(input_parts)

    def _build_synthesis_integration_input(
        self, original_thought: str, all_results: dict[str, str]
    ) -> str:
        """Build synthesis integration input."""
        input_parts = [
            f"Original question: {original_thought}",
            "",
            "Collected insights from comprehensive analysis:",
        ]

        for direction_name, content in all_results.items():
            if (
                direction_name != "synthesis"
            ):  # Avoid including previous synthesis results
                # Completely hide direction concepts, use generic analysis types
                perspective_name = self._get_generic_perspective_name(direction_name)
                input_parts.append(f"• {perspective_name}: {content}")

        input_parts.extend(
            [
                "",
                "TASK: Synthesize all analysis insights into ONE comprehensive, unified answer.",
                "REQUIREMENTS:",
                "1. Provide a single, coherent response directly addressing the original question",
                "2. Integrate all insights naturally without mentioning different analysis types",
                "3. Do NOT list or separate different analysis perspectives in your response",
                "4. Do NOT use section headers or reference any specific analysis methods",
                "5. Do NOT mention 'direction', 'perspective', 'analysis type', or similar terms",
                "6. Write as a unified voice providing the final answer",
                "7. This will be the ONLY response the user sees - make it complete and standalone",
                "8. Your response should read as if it came from a single, integrated thought process",
            ]
        )

        return "\n".join(input_parts)

    def _combine_dual_thinking_results(
        self,
        direction1: ThinkingDirection,
        content1: str,
        direction2: ThinkingDirection,
        content2: str,
        original_thought: str,
    ) -> str:
        """Combine dual thinking direction results."""
        # If the second is synthesis thinking, return its result directly (should already be synthesized)
        if direction2 == ThinkingDirection.SYNTHESIS:
            return content2

        # Otherwise create synthesized answer without mentioning analysis methods
        if (
            direction1 == ThinkingDirection.FACTUAL
            and direction2 == ThinkingDirection.EMOTIONAL
        ):
            return f"Regarding '{original_thought}': A comprehensive analysis reveals both objective realities and human emotional responses. {content1.lower()} while also recognizing that {content2.lower()} These complementary insights suggest a balanced approach that considers both factual evidence and human experience."
        if (
            direction1 == ThinkingDirection.CRITICAL
            and direction2 == ThinkingDirection.OPTIMISTIC
        ):
            return f"Considering '{original_thought}': A thorough evaluation identifies both important concerns and significant opportunities. {content1.lower().strip('.')} while also recognizing promising aspects: {content2.lower()} A measured approach would address the concerns while pursuing the benefits."
        # Generic synthesis, completely hiding analysis structure
        return f"Analyzing '{original_thought}': A comprehensive evaluation reveals multiple important insights. {content1.lower().strip('.')} Additionally, {content2.lower()} Integrating these findings provides a well-rounded understanding that addresses the question from multiple angles."

    def _synthesize_triple_thinking_results(
        self,
        results: dict[str, str],
        thinking_sequence: list[ThinkingDirection],
        original_thought: str,
    ) -> str:
        """Synthesize triple thinking direction results."""
        # Create truly synthesized answer, hiding all analysis structure
        content_pieces = []
        for thinking_direction in thinking_sequence:
            direction_name = thinking_direction.value
            content = results.get(direction_name, "")
            if content:
                # Extract core insights, completely hiding sources
                clean_content = content.strip().rstrip(".!")
                content_pieces.append(clean_content)

        if len(content_pieces) >= 3:
            # Synthesis of three or more perspectives, completely unified
            return f"""Considering the question '{original_thought}', a comprehensive analysis reveals several crucial insights.

{content_pieces[0].lower()}, which establishes the foundation for understanding. This leads to recognizing that {content_pieces[1].lower()}, adding essential depth to our comprehension. Furthermore, {content_pieces[2].lower() if len(content_pieces) > 2 else ""}

Drawing these insights together, the answer emerges as a unified understanding that acknowledges the full complexity while providing clear guidance."""
        if len(content_pieces) == 2:
            return f"Addressing '{original_thought}': A thorough evaluation shows that {content_pieces[0].lower()}, and importantly, {content_pieces[1].lower()} Together, these insights form a comprehensive understanding."
        if len(content_pieces) == 1:
            return f"Regarding '{original_thought}': {content_pieces[0]}"
        return f"After comprehensive consideration of '{original_thought}', the analysis suggests this question merits deeper exploration to provide a complete answer."

    def _synthesize_full_thinking_results(
        self,
        results: dict[str, str],
        thinking_sequence: list[ThinkingDirection],
        original_thought: str,
    ) -> str:
        """Synthesize full multi-thinking results."""
        # If there's a synthesis result, use it preferentially
        synthesis_result = results.get("synthesis")
        if synthesis_result:
            return synthesis_result

        # Otherwise create synthesis
        return self._synthesize_triple_thinking_results(
            results, thinking_sequence, original_thought
        )

    def _get_thinking_contribution(self, thinking_direction: ThinkingDirection) -> str:
        """Get thinking direction contribution description."""
        contributions = {
            ThinkingDirection.FACTUAL: "factual information and objective data",
            ThinkingDirection.EMOTIONAL: "emotional insights and intuitive responses",
            ThinkingDirection.CRITICAL: "critical analysis and risk assessment",
            ThinkingDirection.OPTIMISTIC: "positive possibilities and value identification",
            ThinkingDirection.CREATIVE: "creative alternatives and innovative solutions",
            ThinkingDirection.SYNTHESIS: "process management and integrated thinking",
        }
        return contributions.get(thinking_direction, "specialized thinking")

    def _get_generic_perspective_name(self, direction_name: str) -> str:
        """Get generic analysis type name for thinking direction, hiding direction concepts."""
        name_mapping = {
            "factual": "Factual analysis",
            "emotional": "Emotional considerations",
            "critical": "Risk assessment",
            "optimistic": "Opportunity analysis",
            "creative": "Creative exploration",
            "synthesis": "Strategic synthesis",
        }
        return name_mapping.get(direction_name.lower(), "Analysis")

    def _get_thinking_style_instruction(
        self, thinking_direction: ThinkingDirection
    ) -> str:
        """Get thinking style instruction, avoiding mention of direction concepts."""
        style_mapping = {
            ThinkingDirection.FACTUAL: "factual and objective",
            ThinkingDirection.EMOTIONAL: "emotional and intuitive",
            ThinkingDirection.CRITICAL: "critical and cautious",
            ThinkingDirection.OPTIMISTIC: "positive and optimistic",
            ThinkingDirection.CREATIVE: "creative and innovative",
            ThinkingDirection.SYNTHESIS: "strategic and integrative",
        }
        return style_mapping.get(thinking_direction, "analytical")


# Create global processor instance
_multi_thinking_processor = MultiThinkingSequentialProcessor()


# Convenience function
async def process_with_multi_thinking(
    thought_data: "ThoughtData", context: str = ""
) -> MultiThinkingProcessingResult:
    """Convenience function for processing thoughts using multi-thinking directions."""
    return await _multi_thinking_processor.process_with_multi_thinking(
        thought_data, context
    )


def create_multi_thinking_step_output(
    result: MultiThinkingProcessingResult,
) -> StepOutput:
    """Convert multi-thinking processing result to Agno StepOutput."""
    return StepOutput(
        content=result.content,
        success=True,
        step_name=result.step_name,
    )
