"""Multi-Thinking Sequential Processor.

Implements a parallel processor based on multi-directional thinking methodology,
integrated with the Agno Workflow system. Supports intelligent parallel processing
from single direction to full multi-direction analysis.
"""

from __future__ import annotations

# Lazy import to break circular dependency
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agno.workflow.types import StepOutput

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

if TYPE_CHECKING:
    from mcp_server_mas_sequential_thinking.core.models import ThoughtData
    from mcp_server_mas_sequential_thinking.routing.complexity_types import (
        ComplexityMetrics,
    )
    from mcp_server_mas_sequential_thinking.routing.workflow_state import (
        MultiThinkingState,
    )

logger = logging.getLogger(__name__)

# Message History Configuration (Agno 2.2.12+ optimization)
# Defines optimal context window size for each thinking direction to reduce token usage
MESSAGE_HISTORY_CONFIG = {
    ThinkingDirection.FACTUAL: 5,  # Recent context for data gathering
    ThinkingDirection.EMOTIONAL: 0,  # Fresh perspective without historical bias
    ThinkingDirection.CRITICAL: 3,  # Focused risk analysis with minimal context
    ThinkingDirection.OPTIMISTIC: 3,  # Focused opportunity analysis
    ThinkingDirection.CREATIVE: 8,  # Broader context for creative connections
    ThinkingDirection.SYNTHESIS: 10,  # Maximum context for comprehensive integration
}


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

    def _build_event_handler(
        self,
        state: "MultiThinkingState" | None,
        agent_name: str,
    ):
        if state is None:
            return None

        def handle_event(event: Any) -> None:
            event_name = event.__class__.__name__
            if event_name == "ModelRequestCompleted":
                self._record_token_usage_from_payload(state, agent_name, event)

        return handle_event

    async def _execute_agent(
        self,
        agent: Any,
        agent_name: str,
        input_text: str,
        history_limit: int,
        state: "MultiThinkingState" | None,
    ) -> str:
        if state is not None:
            state.mark_agent_started(agent_name)

        start_time = time.time()
        try:
            event_handler = self._build_event_handler(state, agent_name)
            if event_handler is None:
                result = await agent.arun(
                    input=input_text, num_history_messages=history_limit
                )
            else:
                result = await agent.arun(
                    input=input_text,
                    num_history_messages=history_limit,
                    on_event=event_handler,
                )
            content = self._extract_content(result)
            if state is not None:
                state.mark_agent_completed(
                    agent_name, content, time.time() - start_time
                )
            return content
        except Exception as exc:
            if state is not None:
                state.mark_agent_failed(agent_name, str(exc))
            raise

    @staticmethod
    def _record_token_usage_from_payload(
        state: "MultiThinkingState" | None,
        agent_name: str,
        payload: Any,
    ) -> None:
        if state is None:
            return

        input_tokens = None
        output_tokens = None

        metrics = None
        for attr in ("metrics", "usage", "token_usage"):
            metrics = getattr(payload, attr, None)
            if metrics is not None:
                break

        if metrics is not None:
            input_tokens = getattr(metrics, "input_tokens", None)
            output_tokens = getattr(metrics, "output_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(metrics, "prompt_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(metrics, "completion_tokens", None)

        if input_tokens is None:
            input_tokens = getattr(payload, "input_tokens", None)
            if input_tokens is None:
                input_tokens = getattr(payload, "prompt_tokens", None)

        if output_tokens is None:
            output_tokens = getattr(payload, "output_tokens", None)
            if output_tokens is None:
                output_tokens = getattr(payload, "completion_tokens", None)

        if input_tokens is None and output_tokens is None:
            return

        state.record_token_usage(
            agent_name, int(input_tokens or 0), int(output_tokens or 0)
        )

    async def process_with_multi_thinking(
        self,
        thought_data: "ThoughtData",
        context_prompt: str = "",
        state: "MultiThinkingState" | None = None,
        forced_strategy_name: str | None = None,
        complexity_metrics: ComplexityMetrics | None = None,
    ) -> MultiThinkingProcessingResult:
        """Process thoughts using multi-thinking methodology with parallel execution."""
        start_time = time.time()

        logger.info("Multi-thinking processing started")
        if logger.isEnabledFor(logging.INFO):
            logger.info("Input preview: %s", thought_data.thought[:100])
            logger.info("Context length: %d chars", len(context_prompt))

        if forced_strategy_name is not None and forced_strategy_name != "full_exploration":
            return self._build_error_result(
                "Unsupported strategy. Only 'full_exploration' is allowed.",
                start_time,
            )

        try:
            # Step 1: Intelligent routing decision
            if forced_strategy_name:
                routing_decision = await self.router.route_thought_with_strategy(
                    thought_data,
                    forced_strategy_name,
                    complexity_metrics=complexity_metrics,
                )
            else:
                routing_decision = await self.router.route_thought(
                    thought_data, complexity_metrics=complexity_metrics
                )

            logger.info("Selected strategy: %s", routing_decision.strategy.name)
            if logger.isEnabledFor(logging.INFO):
                sequence = [
                    direction.value
                    for direction in routing_decision.strategy.thinking_sequence
                ]
                logger.info("Thinking sequence: %s", sequence)

            # Step 2: Always execute full-sequence processing.
            if routing_decision.strategy.complexity != ProcessingDepth.FULL:
                return self._build_error_result(
                    "Routing returned unsupported complexity mode. FULL is required.",
                    start_time,
                )
            result = await self._process_full_direction_sequence(
                thought_data, context_prompt, routing_decision, state
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
            error_message = (
                f"Multi-thinking processing failed due to unexpected error: {e!s}"
            )
            logger.exception(error_message)
            return self._build_error_result(error_message, start_time)

    def _build_error_result(
        self, error_message: str, start_time: float
    ) -> MultiThinkingProcessingResult:
        """Build a standardized error result for failed processing runs."""
        processing_time = time.time() - start_time
        logger.error("%s (%.3fs)", error_message, processing_time)
        return MultiThinkingProcessingResult(
            content=f"Multi-thinking processing failed: {error_message}",
            strategy_used="error_fallback",
            thinking_sequence=[],
            processing_time=processing_time,
            complexity_score=0.0,
            cost_reduction=0.0,
            individual_results={},
            step_name="error_handling",
        )

    async def _process_full_direction_sequence(
        self,
        thought_data: "ThoughtData",
        context: str,
        decision: RoutingDecision,
        state: "MultiThinkingState" | None,
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

            initial_content = await self._execute_agent(
                initial_synthesis_agent,
                "synthesis_initial",
                thought_data.thought,
                MESSAGE_HISTORY_CONFIG.get(ThinkingDirection.SYNTHESIS, 10),
                state,
            )
            individual_results["synthesis_initial"] = initial_content

            logger.info("      Initial orchestration completed")

        # Step 2: Identify non-synthesis agents for parallel execution
        non_synthesis_agents = [
            direction
            for direction in thinking_sequence
            if direction != ThinkingDirection.SYNTHESIS
        ]

        if non_synthesis_agents:
            logger.info(
                f"    Step 2: Parallel execution of {len(non_synthesis_agents)} thinking agents"
            )

            import asyncio

            tasks: list[tuple[ThinkingDirection, Any]] = []

            for thinking_direction in non_synthesis_agents:
                logger.info(
                    f"      Preparing {thinking_direction.value} thinking for parallel execution"
                )

                model = self.model_config.create_standard_model()
                logger.info(
                    f"        Using standard model for {thinking_direction.value} thinking"
                )

                agent = self.thinking_factory.create_thinking_agent(
                    thinking_direction, model, context, {}
                )

                # All non-synthesis agents receive original input (parallel processing)
                task = self._execute_agent(
                    agent,
                    thinking_direction.value,
                    thought_data.thought,
                    MESSAGE_HISTORY_CONFIG.get(thinking_direction, 5),
                    state,
                )
                tasks.append((thinking_direction, task))

            # Execute all non-synthesis agents in parallel
            logger.info(f"      Executing {len(tasks)} thinking agents in parallel")
            parallel_results = await asyncio.gather(*[task for _, task in tasks])

            # Process parallel results
            for (completed_direction, _), content in zip(
                tasks, parallel_results, strict=False
            ):
                individual_results[completed_direction.value] = content
                logger.info(
                    "        %s thinking completed (parallel)",
                    completed_direction.value,
                )

        # Step 3: Final SYNTHESIS for integration (if last agent is SYNTHESIS)
        final_synthesis_agents = [
            i
            for i, direction in enumerate(thinking_sequence)
            if direction == ThinkingDirection.SYNTHESIS and i > 0
        ]

        if final_synthesis_agents:
            logger.info("    Step 3: Final synthesis integration")

            final_synthesis_model = self.model_config.create_enhanced_model()
            logger.info("      Using enhanced model for final synthesis")

            # Remove initial synthesis from results for final integration
            integration_results = {
                k: v
                for k, v in individual_results.items()
                if not k.startswith("synthesis_")
            }

            final_synthesis_agent = self.thinking_factory.create_thinking_agent(
                ThinkingDirection.SYNTHESIS,
                final_synthesis_model,
                context,
                integration_results,
            )

            # Build synthesis integration input
            synthesis_input = self._build_synthesis_integration_input(
                thought_data.thought, integration_results
            )

            final_content = await self._execute_agent(
                final_synthesis_agent,
                "synthesis_final",
                synthesis_input,
                MESSAGE_HISTORY_CONFIG.get(ThinkingDirection.SYNTHESIS, 10),
                state,
            )
            individual_results["synthesis_final"] = final_content

            logger.info("      Final synthesis integration completed")

            # Use final synthesis result as the answer
            final_answer = final_content
        else:
            # No final synthesis - create programmatic synthesis
            final_answer = self._synthesize_full_thinking_results(
                individual_results, list(thinking_sequence), thought_data.thought
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

    def _synthesize_full_thinking_results(
        self,
        results: dict[str, str],
        thinking_sequence: list[ThinkingDirection],
        original_thought: str,
    ) -> str:
        """Synthesize full multi-thinking results."""
        _ = thinking_sequence
        synthesis_result = results.get("synthesis")
        if synthesis_result:
            return synthesis_result

        if not results:
            return (
                f"After full multi-thinking analysis of '{original_thought}', "
                "no stable synthesis output was produced."
            )

        perspectives = []
        for direction_name, content in results.items():
            if not content:
                continue
            perspective = self._get_generic_perspective_name(direction_name)
            perspectives.append(f"{perspective}: {content}")

        if not perspectives:
            return (
                f"After full multi-thinking analysis of '{original_thought}', "
                "all perspective outputs were empty."
            )

        return (
            f"Integrated response for '{original_thought}':\n\n"
            + "\n".join(f"- {item}" for item in perspectives)
        )

    def _get_generic_perspective_name(self, direction_name: str) -> str:
        """Get generic analysis type name for thinking direction."""
        name_mapping = {
            "factual": "Factual analysis",
            "emotional": "Emotional considerations",
            "critical": "Risk assessment",
            "optimistic": "Opportunity analysis",
            "creative": "Creative exploration",
            "synthesis": "Strategic synthesis",
        }
        return name_mapping.get(direction_name.lower(), "Analysis")


# Create global processor instance
_multi_thinking_processor = MultiThinkingSequentialProcessor()


# Convenience function
async def process_with_multi_thinking(
    thought_data: "ThoughtData",
    context: str = "",
    state: "MultiThinkingState" | None = None,
    forced_strategy_name: str | None = None,
    complexity_metrics: ComplexityMetrics | None = None,
) -> MultiThinkingProcessingResult:
    """Convenience function for processing thoughts using multi-thinking directions."""
    return await _multi_thinking_processor.process_with_multi_thinking(
        thought_data,
        context,
        state=state,
        forced_strategy_name=forced_strategy_name,
        complexity_metrics=complexity_metrics,
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
