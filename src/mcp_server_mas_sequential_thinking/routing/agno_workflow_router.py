"""Multi-Thinking Workflow Router - Complete Rewrite.

纯净的多向思维工作流实现, 基于Agno v2.0框架。
完全移除旧的复杂度路由, 专注于多向思维方法论。
"""

import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

from mcp_server_mas_sequential_thinking.config.modernized_config import get_model_config

# Import at module level - moved from function to avoid PLC0415
from mcp_server_mas_sequential_thinking.core.models import ThoughtData
from mcp_server_mas_sequential_thinking.processors.multi_thinking_processor import (
    MultiThinkingSequentialProcessor,
    create_multi_thinking_step_output,
)

logger = logging.getLogger(__name__)


@dataclass
class MultiThinkingWorkflowResult:
    """Result from Multi-Thinking workflow execution."""

    content: str
    strategy_used: str
    processing_time: float
    complexity_score: float
    step_name: str
    thinking_sequence: list[str]
    cost_reduction: float


class MultiThinkingWorkflowRouter:
    """Pure Multi-Thinking workflow router using Agno v2.0."""

    def __init__(self) -> None:
        """Initialize Multi-Thinking workflow router."""
        self.model_config = get_model_config()

        # Initialize Multi-Thinking processor
        self.multi_thinking_processor = MultiThinkingSequentialProcessor()

        # Create Multi-Thinking processing step
        self.multi_thinking_step = self._create_multi_thinking_step()

        # Create router that always selects Multi-Thinking
        self.router = Router(
            name="multi_thinking_router",
            selector=self._multi_thinking_selector,
            choices=[self.multi_thinking_step],
        )

        # Create main workflow
        self.workflow = Workflow(
            name="multi_thinking_workflow",
            steps=[self.router],
        )

        logger.info("Multi-Thinking Workflow Router initialized")

    def _multi_thinking_selector(self, step_input: StepInput) -> list[Step]:
        """Selector that always returns Multi-Thinking processing."""
        try:
            logger.info("🎩 MULTI-THINKING WORKFLOW ROUTING:")

            # Extract thought content for logging
            if isinstance(step_input.input, dict):
                thought_content = step_input.input.get("thought", "")
                thought_number = step_input.input.get("thought_number", 1)
                total_thoughts = step_input.input.get("total_thoughts", 1)
            else:
                thought_content = str(step_input.input)
                thought_number = 1
                total_thoughts = 1

            logger.info(
                "  📝 Input: %s%s",
                thought_content[:100],
                "..." if len(thought_content) > 100 else "",
            )
            logger.info("  🔢 Progress: %s/%s", thought_number, total_thoughts)
            logger.info("  ✅ Multi-Thinking selected - exclusive thinking methodology")

            return [self.multi_thinking_step]

        except Exception as e:
            logger.exception("Error in Multi-Thinking selector: %s", e)
            logger.warning("Continuing with Multi-Thinking processing despite error")
            return [self.multi_thinking_step]

    def _create_multi_thinking_step(self) -> Step:
        """Create Six Thinking Hats processing step."""

        async def multi_thinking_executor(
            step_input: StepInput, session_state: dict[str, Any]
        ) -> StepOutput:
            """Execute Multi-Thinking thinking process."""
            try:
                logger.info("🎩 MULTI-THINKING STEP EXECUTION:")

                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get(
                        "thought", str(step_input.input)
                    )
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                    context = step_input.input.get("context", "")
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1
                    context = ""

                # ThoughtData is now imported at module level

                # Extract context preservation fields from input if available
                if isinstance(step_input.input, dict):
                    is_revision = step_input.input.get("isRevision", False)
                    branch_from_thought = step_input.input.get("branchFromThought")
                    branch_id = step_input.input.get("branchId")
                    needs_more_thoughts = step_input.input.get(
                        "needsMoreThoughts", False
                    )
                    next_thought_needed = step_input.input.get(
                        "nextThoughtNeeded", True
                    )
                else:
                    is_revision = False
                    branch_from_thought = None
                    branch_id = None
                    needs_more_thoughts = False
                    next_thought_needed = True

                thought_data = ThoughtData(
                    thought=thought_content,
                    thoughtNumber=thought_number,
                    totalThoughts=total_thoughts,
                    nextThoughtNeeded=next_thought_needed,
                    isRevision=is_revision,
                    branchFromThought=branch_from_thought,
                    branchId=branch_id,
                    needsMoreThoughts=needs_more_thoughts,
                )

                logger.info("  📝 Input: %s...", thought_content[:100])
                logger.info("  🔢 Thought: %s/%s", thought_number, total_thoughts)

                # Process with Multi-Thinking
                result = (
                    await self.multi_thinking_processor.process_with_multi_thinking(
                        thought_data, context
                    )
                )

                # Store metadata in session_state
                session_state["current_strategy"] = result.strategy_used
                session_state["current_complexity_score"] = result.complexity_score
                session_state["thinking_sequence"] = result.thinking_sequence
                session_state["cost_reduction"] = result.cost_reduction

                logger.info("  ✅ Multi-Thinking completed: %s", result.strategy_used)
                logger.info("  📊 Complexity: %.1f", result.complexity_score)
                logger.info("  💰 Cost Reduction: %.1f%%", result.cost_reduction)

                return create_multi_thinking_step_output(result)

            except Exception as e:
                logger.exception("  ❌ Multi-Thinking execution failed")
                return StepOutput(
                    content=f"Multi-Thinking processing failed: {e!s}",
                    success=False,
                    error=str(e),
                    step_name="multi_thinking_error",
                )

        return Step(
            name="multi_thinking_processing",
            executor=multi_thinking_executor,
            description="Six Thinking Hats sequential processing",
        )

    async def process_thought_workflow(
        self, thought_data: "ThoughtData", context_prompt: str
    ) -> MultiThinkingWorkflowResult:
        """Process thought using Multi-Thinking workflow."""
        start_time = time.time()

        try:
            logger.info("🚀 MULTI-THINKING WORKFLOW INITIALIZATION:")
            logger.info(
                "  📝 Thought: %s%s",
                thought_data.thought[:100],
                "..." if len(thought_data.thought) > 100 else "",
            )
            logger.info(
                "  🔢 Thought Number: %s/%s",
                thought_data.thoughtNumber,
                thought_data.totalThoughts,
            )
            logger.info("  📋 Context Length: %d chars", len(context_prompt))
            logger.info(
                "  ⏰ Start Time: %s",
                time.strftime("%H:%M:%S", time.localtime(start_time)),
            )

            # Prepare workflow input for Multi-Thinking
            workflow_input = {
                "thought": thought_data.thought,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
                "context": context_prompt,
            }

            logger.info("📦 WORKFLOW INPUT PREPARATION:")
            logger.info("  📊 Input Keys: %s", list(workflow_input.keys()))
            logger.info("  📏 Input Size: %d chars", len(str(workflow_input)))

            # Initialize session_state for metadata tracking
            session_state: dict[str, Any] = {
                "start_time": start_time,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
            }

            logger.info("🎯 SESSION STATE SETUP:")
            logger.info("  🔑 State Keys: %s", list(session_state.keys()))
            logger.info("  📈 Metadata: %s", session_state)

            logger.info(
                "▶️  EXECUTING Multi-Thinking workflow for thought #%s",
                thought_data.thoughtNumber,
            )

            # Execute Multi-Thinking workflow
            logger.info("🔄 WORKFLOW EXECUTION START...")
            result = await self.workflow.arun(
                input=workflow_input, session_state=session_state
            )
            logger.info("✅ WORKFLOW EXECUTION COMPLETED")

            processing_time = time.time() - start_time

            # Extract clean content from result
            content = self._extract_clean_content(result)

            logger.info("📋 CONTENT VALIDATION:")
            logger.info(
                "  ✅ Content extracted successfully: %d characters", len(content)
            )
            logger.info(
                "  📝 Content preview: %s%s",
                content[:150],
                "..." if len(content) > 150 else "",
            )

            # Get metadata from session_state
            complexity_score = session_state.get("current_complexity_score", 0.0)
            strategy_used = session_state.get("current_strategy", "multi_thinking")
            thinking_sequence = session_state.get("thinking_sequence", [])
            cost_reduction = session_state.get("cost_reduction", 0.0)

            logger.info("📊 WORKFLOW RESULT COMPILATION:")
            logger.info("  🎯 Strategy used: %s", strategy_used)
            logger.info("  🧠 Thinking sequence: %s", " → ".join(thinking_sequence))
            logger.info("  📈 Complexity score: %.1f", complexity_score)
            logger.info("  💰 Cost reduction: %.1f%%", cost_reduction)
            logger.info("  ⏱️  Processing time: %.3fs", processing_time)

            workflow_result = MultiThinkingWorkflowResult(
                content=content,
                strategy_used=strategy_used,
                processing_time=processing_time,
                complexity_score=complexity_score,
                step_name="multi_thinking_execution",
                thinking_sequence=thinking_sequence,
                cost_reduction=cost_reduction,
            )

            logger.info("🎉 MULTI-THINKING WORKFLOW COMPLETION:")
            logger.info(
                "  ✅ Completed: strategy=%s, time=%.3fs, score=%.1f, reduction=%.1f%%",
                strategy_used,
                processing_time,
                complexity_score,
                cost_reduction,
            )

            return workflow_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.exception(
                "Multi-Thinking workflow execution failed after %.3fs", processing_time
            )

            return MultiThinkingWorkflowResult(
                content=f"Error processing thought with Multi-Thinking: {e!s}",
                strategy_used="error_fallback",
                processing_time=processing_time,
                complexity_score=0.0,
                step_name="error_handling",
                thinking_sequence=[],
                cost_reduction=0.0,
            )

    def _extract_clean_content(self, result: Any) -> str:
        """Extract clean content from workflow result."""

        def extract_recursive(obj: Any, depth: int = 0) -> str:
            """Recursively extract clean content from nested objects."""
            if depth > 10:  # Prevent infinite recursion
                return str(obj)

            # Handle dictionary with common content keys
            if isinstance(obj, dict):
                for key in [
                    "result",
                    "content",
                    "message",
                    "text",
                    "response",
                    "output",
                    "answer",
                ]:
                    if key in obj:
                        return extract_recursive(obj[key], depth + 1)
                # Fallback to any string content
                for value in obj.values():
                    if isinstance(value, str) and len(value.strip()) > 10:
                        return value.strip()
                return str(obj)

            # Handle objects with content attributes
            if hasattr(obj, "content"):
                content = obj.content
                if isinstance(content, str):
                    return content.strip()
                return extract_recursive(content, depth + 1)

            # Handle other output objects
            if hasattr(obj, "output"):
                return extract_recursive(obj.output, depth + 1)

            # Handle list/tuple - extract first meaningful content
            if isinstance(obj, (list, tuple)) and obj:
                for item in obj:
                    result = extract_recursive(item, depth + 1)
                    if isinstance(result, str) and len(result.strip()) > 10:
                        return result.strip()

            # If it's already a string, clean it up
            if isinstance(obj, str):
                content = obj.strip()

                # Remove object representations
                if any(
                    content.startswith(pattern)
                    for pattern in [
                        "RunOutput(",
                        "TeamRunOutput(",
                        "StepOutput(",
                        "WorkflowResult(",
                        "{'result':",
                        '{"result":',
                        "{'content':",
                        '{"content":',
                    ]
                ):
                    # Try to extract content using regex (re imported at module level)

                    patterns = [
                        (r"content='([^']*)'", 1),
                        (r'content="([^"]*)"', 1),
                        (r"'result':\s*'([^']*)'", 1),
                        (r'"result":\s*"([^"]*)"', 1),
                        (r"'([^']{20,})'", 1),
                        (r'"([^"]{20,})"', 1),
                    ]

                    for pattern, group in patterns:
                        match = re.search(pattern, content)
                        if match:
                            extracted = match.group(group).strip()
                            if len(extracted) > 10:
                                return extracted

                    # Clean up object syntax
                    cleaned = re.sub(r'[{}()"\']', " ", content)
                    cleaned = re.sub(
                        r"\b(RunOutput|TeamRunOutput|StepOutput|content|result|success|error)\b",
                        " ",
                        cleaned,
                    )
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()

                    if len(cleaned) > 20:
                        return cleaned

                return content

            # Fallback
            result = str(obj).strip()
            if len(result) > 20 and not any(
                result.startswith(pattern)
                for pattern in ["RunOutput(", "TeamRunOutput(", "StepOutput(", "<"]
            ):
                return result

            return "Multi-Thinking processing completed successfully"

        return extract_recursive(result)


# For backward compatibility with the old AgnoWorkflowRouter name
AgnoWorkflowRouter = MultiThinkingWorkflowRouter
WorkflowResult = MultiThinkingWorkflowResult
