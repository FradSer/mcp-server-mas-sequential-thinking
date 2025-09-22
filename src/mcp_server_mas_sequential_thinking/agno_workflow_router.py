"""Six Thinking Hats Workflow Router - Complete Rewrite

纯净的Six Thinking Hats工作流实现，基于Agno v2.0框架。
完全移除旧的复杂度路由，专注于六帽思维方法论。
"""

import time
from dataclasses import dataclass

from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.types import StepInput, StepOutput
from agno.workflow.workflow import Workflow

from .logging_config import get_logger
from .models import ThoughtData
from .modernized_config import get_model_config

# Import Six Hats support
from .six_hats_processor import (
    SixHatsSequentialProcessor,
    create_six_hats_step_output,
)

logger = get_logger(__name__)


@dataclass
class SixHatsWorkflowResult:
    """Result from Six Hats workflow execution."""

    content: str
    strategy_used: str
    processing_time: float
    complexity_score: float
    step_name: str
    hat_sequence: list[str]
    cost_reduction: float


class SixHatsWorkflowRouter:
    """Pure Six Thinking Hats workflow router using Agno v2.0."""

    def __init__(self):
        """Initialize Six Hats workflow router."""
        self.model_config = get_model_config()

        # Initialize Six Hats processor
        self.six_hats_processor = SixHatsSequentialProcessor()

        # Create Six Hats processing step
        self.six_hats_step = self._create_six_hats_step()

        # Create router that always selects Six Hats
        self.router = Router(
            name="six_hats_router",
            selector=self._six_hats_selector,
            choices=[self.six_hats_step],
        )

        # Create main workflow
        self.workflow = Workflow(
            name="six_hats_thinking_workflow",
            steps=[self.router],
        )

        logger.info("Six Hats Workflow Router initialized")

    def _six_hats_selector(self, step_input: StepInput) -> list[Step]:
        """Selector that always returns Six Hats processing."""
        try:
            logger.info("🎩 SIX HATS WORKFLOW ROUTING:")

            # Extract thought content for logging
            if isinstance(step_input.input, dict):
                thought_content = step_input.input.get("thought", "")
                thought_number = step_input.input.get("thought_number", 1)
                total_thoughts = step_input.input.get("total_thoughts", 1)
            else:
                thought_content = str(step_input.input)
                thought_number = 1
                total_thoughts = 1

            logger.info(f"  📝 Input: {thought_content[:100]}{'...' if len(thought_content) > 100 else ''}")
            logger.info(f"  🔢 Progress: {thought_number}/{total_thoughts}")
            logger.info("  ✅ Six Hats selected - exclusive thinking methodology")

            return [self.six_hats_step]

        except Exception as e:
            logger.error(f"Error in Six Hats selector: {e}")
            logger.warning("Continuing with Six Hats processing despite error")
            return [self.six_hats_step]

    def _create_six_hats_step(self) -> Step:
        """Create Six Thinking Hats processing step."""

        async def six_hats_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Execute Six Hats thinking process."""
            try:
                logger.info("🎩 SIX HATS STEP EXECUTION:")

                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get("thought", str(step_input.input))
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                    context = step_input.input.get("context", "")
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1
                    context = ""

                # Create ThoughtData
                thought_data = ThoughtData(
                    thought=thought_content,
                    thoughtNumber=thought_number,
                    totalThoughts=total_thoughts,
                    nextThoughtNeeded=True,
                    isRevision=False,
                    branchFromThought=None,
                    branchId=None,
                    needsMoreThoughts=False,
                )

                logger.info(f"  📝 Input: {thought_content[:100]}...")
                logger.info(f"  🔢 Thought: {thought_number}/{total_thoughts}")

                # Process with Six Hats
                result = await self.six_hats_processor.process_thought_with_six_hats(
                    thought_data, context
                )

                # Store metadata in session_state
                session_state["current_strategy"] = result.strategy_used
                session_state["current_complexity_score"] = result.complexity_score
                session_state["six_hats_sequence"] = result.hat_sequence
                session_state["cost_reduction"] = result.cost_reduction

                logger.info(f"  ✅ Six Hats completed: {result.strategy_used}")
                logger.info(f"  📊 Complexity: {result.complexity_score:.1f}")
                logger.info(f"  💰 Cost Reduction: {result.cost_reduction:.1f}%")

                return create_six_hats_step_output(result)

            except Exception as e:
                logger.error(f"  ❌ Six Hats execution failed: {e}")
                return StepOutput(
                    content=f"Six Hats processing failed: {e!s}",
                    success=False,
                    error=str(e),
                    step_name="six_hats_error"
                )

        return Step(
            name="six_hats_processing",
            executor=six_hats_executor,
            description="Six Thinking Hats sequential processing",
        )

    async def process_thought_workflow(
        self, thought_data: ThoughtData, context_prompt: str
    ) -> SixHatsWorkflowResult:
        """Process thought using Six Hats workflow."""
        start_time = time.time()

        try:
            logger.info("🚀 SIX HATS WORKFLOW INITIALIZATION:")
            logger.info(f"  📝 Thought: {thought_data.thought[:100]}{'...' if len(thought_data.thought) > 100 else ''}")
            logger.info(f"  🔢 Thought Number: {thought_data.thoughtNumber}/{thought_data.totalThoughts}")
            logger.info(f"  📋 Context Length: {len(context_prompt)} chars")
            logger.info(f"  ⏰ Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

            # Prepare workflow input for Six Hats
            workflow_input = {
                "thought": thought_data.thought,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
                "context": context_prompt,
            }

            logger.info("📦 WORKFLOW INPUT PREPARATION:")
            logger.info(f"  📊 Input Keys: {list(workflow_input.keys())}")
            logger.info(f"  📏 Input Size: {len(str(workflow_input))} chars")

            # Initialize session_state for metadata tracking
            session_state = {
                "start_time": start_time,
                "thought_number": thought_data.thoughtNumber,
                "total_thoughts": thought_data.totalThoughts,
            }

            logger.info("🎯 SESSION STATE SETUP:")
            logger.info(f"  🔑 State Keys: {list(session_state.keys())}")
            logger.info(f"  📈 Metadata: {session_state}")

            logger.info(f"▶️  EXECUTING Six Hats workflow for thought #{thought_data.thoughtNumber}")

            # Execute Six Hats workflow
            logger.info("🔄 WORKFLOW EXECUTION START...")
            result = await self.workflow.arun(
                input=workflow_input, session_state=session_state
            )
            logger.info("✅ WORKFLOW EXECUTION COMPLETED")

            processing_time = time.time() - start_time

            # Extract clean content from result
            content = self._extract_clean_content(result)

            logger.info("📋 CONTENT VALIDATION:")
            logger.info(f"  ✅ Content extracted successfully: {len(content)} characters")
            logger.info(f"  📝 Content preview: {content[:150]}{'...' if len(content) > 150 else ''}")

            # Get metadata from session_state
            complexity_score = session_state.get("current_complexity_score", 0.0)
            strategy_used = session_state.get("current_strategy", "six_hats")
            hat_sequence = session_state.get("six_hats_sequence", [])
            cost_reduction = session_state.get("cost_reduction", 0.0)

            logger.info("📊 WORKFLOW RESULT COMPILATION:")
            logger.info(f"  🎯 Strategy used: {strategy_used}")
            logger.info(f"  🎩 Hat sequence: {' → '.join(hat_sequence)}")
            logger.info(f"  📈 Complexity score: {complexity_score:.1f}")
            logger.info(f"  💰 Cost reduction: {cost_reduction:.1f}%")
            logger.info(f"  ⏱️  Processing time: {processing_time:.3f}s")

            workflow_result = SixHatsWorkflowResult(
                content=content,
                strategy_used=strategy_used,
                processing_time=processing_time,
                complexity_score=complexity_score,
                step_name="six_hats_execution",
                hat_sequence=hat_sequence,
                cost_reduction=cost_reduction,
            )

            logger.info("🎉 SIX HATS WORKFLOW COMPLETION:")
            logger.info(
                f"  ✅ Completed: strategy={strategy_used}, "
                f"time={processing_time:.3f}s, score={complexity_score:.1f}, "
                f"reduction={cost_reduction:.1f}%"
            )

            return workflow_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Six Hats workflow execution failed after {processing_time:.3f}s: {e}")

            return SixHatsWorkflowResult(
                content=f"Error processing thought with Six Hats: {e!s}",
                strategy_used="error_fallback",
                processing_time=processing_time,
                complexity_score=0.0,
                step_name="error_handling",
                hat_sequence=[],
                cost_reduction=0.0,
            )

    def _extract_clean_content(self, result) -> str:
        """Extract clean content from workflow result."""
        def extract_recursive(obj, depth=0):
            """Recursively extract clean content from nested objects."""
            if depth > 10:  # Prevent infinite recursion
                return str(obj)

            # Handle dictionary with common content keys
            if isinstance(obj, dict):
                for key in ["result", "content", "message", "text", "response", "output", "answer"]:
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
                if any(content.startswith(pattern) for pattern in [
                    "RunOutput(", "TeamRunOutput(", "StepOutput(", "WorkflowResult(",
                    "{'result':", '{"result":', "{'content':", '{"content":',
                ]):
                    # Try to extract content using regex
                    import re
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
                    cleaned = re.sub(r"\b(RunOutput|TeamRunOutput|StepOutput|content|result|success|error)\b", " ", cleaned)
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()

                    if len(cleaned) > 20:
                        return cleaned

                return content

            # Fallback
            result = str(obj).strip()
            if len(result) > 20 and not any(result.startswith(pattern) for pattern in [
                "RunOutput(", "TeamRunOutput(", "StepOutput(", "<"
            ]):
                return result

            return "Six Hats processing completed successfully"

        return extract_recursive(result)


# For backward compatibility with the old AgnoWorkflowRouter name
AgnoWorkflowRouter = SixHatsWorkflowRouter
WorkflowResult = SixHatsWorkflowResult
