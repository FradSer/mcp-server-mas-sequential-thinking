"""
Agno Workflow-based routing for adaptive sequential thinking.

This module implements Agno-standard Workflow orchestration with Router pattern
for intelligent multi-agent coordination without FastAPI dependencies.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import logging
import time
import hashlib

from agno.workflow.workflow import Workflow
from agno.workflow.router import Router
from agno.workflow.step import Step
from agno.workflow.parallel import Parallel
from agno.workflow.condition import Condition
from agno.workflow.types import StepInput, StepOutput
from agno.agent import Agent
from agno.team import Team
from agno.tools.reasoning import ReasoningTools

from .models import ThoughtData
from .adaptive_routing import (
    ComplexityAnalyzer,
    BasicComplexityAnalyzer,
    ComplexityLevel,
)
from .modernized_config import get_model_config
from .processing_constants import (
    ComplexityThresholds, QualityThresholds, RetryConfiguration,
    SixHatsConfiguration, get_complexity_level_name, is_content_sufficient_quality,
    count_complex_keywords, is_suitable_for_six_hats, ERROR_INDICATORS
)
from .base_executor import (
    BaseExecutor, SingleAgentExecutor, HybridTeamExecutor, MultiAgentExecutor
)

# Import Six Hats support
try:
    from .six_hats_processor import (
        SixHatsSequentialProcessor, process_with_six_hats, create_six_hats_step_output
    )
    SIX_HATS_AVAILABLE = True
except ImportError:
    SIX_HATS_AVAILABLE = False
    logger.warning("Six Hats functionality not available in workflow router")

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory for creating standardized agents with reduced duplication."""

    @staticmethod
    def create_agent(
        name: str, role: str, model: Any, instructions: List[str]
    ) -> Agent:
        """Create a standardized agent with common configuration."""
        return Agent(
            name=name,
            role=role,
            model=model,
            tools=[ReasoningTools],
            instructions=instructions,
        )

    @classmethod
    def create_planner(cls, model: Any, complexity_level: str = "basic") -> Agent:
        """Create a planner agent with complexity-appropriate instructions."""
        instructions = {
            "basic": [
                "Analyze the thought and create a strategic approach.",
                "Break down complex ideas into manageable components.",
                "Provide clear planning guidance.",
            ],
            "advanced": [
                "Create comprehensive strategic approaches.",
                "Plan multi-step solutions and methodologies.",
                "Provide strategic oversight and direction.",
            ],
        }
        return cls.create_agent(
            "Planner", "Strategic Planner", model, instructions[complexity_level]
        )

    @classmethod
    def create_analyzer(cls, model, complexity_level: str = "basic") -> Agent:
        """Create an analyzer agent with complexity-appropriate instructions."""
        instructions = {
            "basic": [
                "Perform deep analysis of the thought content.",
                "Identify patterns, connections, and implications.",
                "Provide analytical insights.",
            ],
            "advanced": [
                "Perform comprehensive analysis of complex thoughts.",
                "Identify deep patterns and relationships.",
                "Provide sophisticated analytical perspectives.",
            ],
        }
        return cls.create_agent(
            "Analyzer", "Core Analyst", model, instructions[complexity_level]
        )

    @classmethod
    def create_researcher(cls, model) -> Agent:
        """Create a researcher agent."""
        return cls.create_agent(
            "Researcher",
            "Information Gatherer",
            model,
            [
                "Gather relevant information and context.",
                "Verify facts and explore related concepts.",
                "Provide research-backed insights.",
            ],
        )

    @classmethod
    def create_critic(cls, model) -> Agent:
        """Create a critic agent."""
        return cls.create_agent(
            "Critic",
            "Quality Controller",
            model,
            [
                "Critically evaluate ideas and proposals.",
                "Identify potential weaknesses and improvements.",
                "Ensure high-quality outputs.",
            ],
        )

    @classmethod
    def create_synthesizer(cls, model) -> Agent:
        """Create a synthesizer agent."""
        return cls.create_agent(
            "Synthesizer",
            "Response Coordinator",
            model,
            [
                "Synthesize insights from all team members.",
                "Create coherent and comprehensive responses.",
                "Provide actionable guidance for next steps.",
            ],
        )


class StepExecutorMixin:
    """Mixin providing common step execution patterns."""

    @staticmethod
    def _create_step_output(
        content: str,
        strategy: str,
        success: bool = True,
        session_state: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        specialists: Optional[List[str]] = None,
    ) -> StepOutput:
        """Create standardized StepOutput with common fields."""
        # Create enriched content with metadata
        enriched_content = {
            "result": content,
            "strategy": strategy,
            "complexity": session_state.get("current_complexity_score", 0)
            if session_state
            else 0,
        }
        if specialists:
            enriched_content["specialists"] = specialists

        return StepOutput(
            content=enriched_content,
            success=success,
            error=error,
            step_name=f"{strategy}_processing",
        )

    @staticmethod
    def _update_session_state(
        session_state: Optional[Dict[str, Any]], strategy: str, completed_key: str
    ) -> None:
        """Update session state with completion tracking."""
        if session_state is not None:
            session_state[completed_key] = True
            # Note: current_strategy is set directly in executors for consistency

    @staticmethod
    def _handle_execution_error(error: Exception, strategy: str) -> StepOutput:
        """Handle execution errors with standardized error response."""
        formatted_strategy = strategy.replace("_", " ").capitalize()
        error_msg = f"{formatted_strategy} processing failed: {str(error)}"
        return StepOutput(content=error_msg, success=False, error=str(error))


@dataclass
class WorkflowResult:
    """Result from Agno workflow execution with metadata."""

    content: str
    strategy_used: str
    processing_time: float
    complexity_score: float
    step_name: str


# Removed ThoughtInput - using Agno standard StepInput instead


class AgnoWorkflowRouter(StepExecutorMixin):
    """
    Agno Workflow-based router with intelligent complexity routing.

    Uses Agno Workflow + Router pattern for declarative multi-agent coordination
    without FastAPI dependencies. Implements the recommended Agno architecture.
    """

    def __init__(self, complexity_analyzer: Optional[ComplexityAnalyzer] = None):
        self.complexity_analyzer = complexity_analyzer or BasicComplexityAnalyzer()
        self.model_config = get_model_config()

        # Initialize Six Hats processor if available
        self.six_hats_processor = SixHatsSequentialProcessor() if SIX_HATS_AVAILABLE else None

        # Create processing steps - include Six Hats if available
        processing_choices = []

        if SIX_HATS_AVAILABLE:
            # Six Hats step (highest priority for appropriate problems)
            self.six_hats_step = self._create_six_hats_step()
            processing_choices.append(self.six_hats_step)

        # Original steps as fallback
        self.single_agent_step = self._create_single_agent_step()
        self.hybrid_team_step = self._create_hybrid_team_step()
        self.full_team_step = self._create_full_team_step()

        processing_choices.extend([
            self.single_agent_step,
            self.hybrid_team_step,
            self.full_team_step,
        ])

        # Create complexity-based router with custom selector
        self.complexity_router = Router(
            name="adaptive_complexity_router",
            selector=self._enhanced_complexity_selector,
            choices=processing_choices,
        )

        # Create main workflow - single step to avoid condition override
        self.workflow = Workflow(
            name="adaptive_sequential_thinking_workflow",
            steps=[
                self.complexity_router  # Quality improvement handled within router
            ],
        )

        if SIX_HATS_AVAILABLE:
            logger.info("AgnoWorkflowRouter initialized with Six Hats + Original strategies")
        else:
            logger.info("AgnoWorkflowRouter initialized with Original strategies only")

    def _quality_evaluator(self, step_input: StepInput) -> bool:
        """Evaluate if additional quality improvement is needed."""
        try:
            previous_content = step_input.previous_step_content or ""

            # Use centralized quality assessment
            if not is_content_sufficient_quality(previous_content):
                logger.info("Quality issues detected: content_quality_insufficient")
                return True

            # Check for insufficient depth in longer content
            if self._has_insufficient_depth(previous_content):
                logger.info("Quality issues detected: insufficient_depth")
                return True

            logger.info("Quality check passed - no improvement needed")
            return False

        except Exception as e:
            logger.error(f"Error in quality evaluator: {e}")
            return False

    def _has_insufficient_depth(self, content: str) -> bool:
        """Check if content lacks analytical depth."""
        if len(content.strip()) <= QualityThresholds.INSUFFICIENT_DEPTH_THRESHOLD:
            return True

        depth_indicators = ["analyze", "consider", "factors", "implications"]
        depth_count = sum(
            1 for indicator in depth_indicators
            if indicator.lower() in content.lower()
        )
        return depth_count < 2

    def _create_quality_improvement_step(self) -> Step:
        """Create quality improvement step for when initial output needs enhancement."""
        model = self.model_config.create_team_model()

        quality_improver = Agent(
            name="QualityImprover",
            role="Output Enhancement Specialist",
            model=model,
            tools=[ReasoningTools],
            instructions=[
                "You are responsible for improving the quality of thought analysis.",
                "Review the previous output and enhance it with more depth and clarity.",
                "Address any gaps, errors, or insufficient detail.",
                "Provide a comprehensive, well-structured response.",
            ],
        )

        async def quality_improvement_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Quality improvement executor."""
            try:
                previous_content = step_input.previous_step_content or ""
                quality_issues = session_state.get("quality_issues", [])

                # Construct improvement prompt based on identified issues
                improvement_prompt = f"""
                Previous analysis: {previous_content}

                Quality issues identified: {", ".join(quality_issues)}

                Please provide an improved, more comprehensive analysis that addresses these issues.
                """

                result = await quality_improver.arun(
                    input=improvement_prompt, session_state=session_state
                )

                # Track improvement in session_state
                session_state["quality_improved"] = True
                session_state["final_quality_score"] = (
                    session_state.get("quality_score", 0) + 0.3
                )

                return StepOutput(
                    content=result,
                    success=True,
                )

            except Exception as e:
                return StepOutput(
                    content=f"Quality improvement failed: {str(e)}",
                    success=False,
                    error=str(e),
                )

        return Step(
            name="quality_improvement",
            executor=quality_improvement_executor,
            description="Enhance output quality when needed",
        )

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
                    content=f"Six Hats processing failed: {str(e)}",
                    success=False,
                    error=str(e),
                    step_name="six_hats_error"
                )

        return Step(
            name="six_hats_processing",
            executor=six_hats_executor,
            description="Six Thinking Hats sequential processing with intelligent routing",
        )

    def _enhanced_complexity_selector(self, step_input: StepInput) -> List[Step]:
        """Enhanced selector that considers Six Hats processing first."""
        try:
            logger.info("🧭 ENHANCED WORKFLOW ROUTING:")

            # Extract thought content from StepInput
            if isinstance(step_input.input, dict):
                thought_content = step_input.input.get("thought", "")
                thought_number = step_input.input.get("thought_number", 1)
                total_thoughts = step_input.input.get("total_thoughts", 1)
            else:
                thought_content = str(step_input.input)
                thought_number = 1
                total_thoughts = 1

            logger.info(f"  📝 Input: {thought_content[:100]}{'...' if len(thought_content) > 100 else ''}")

            # If Six Hats is available, try to use it first
            if SIX_HATS_AVAILABLE and self.six_hats_processor:
                logger.info("  🎩 Evaluating Six Hats suitability...")

                # Quick suitability check
                is_suitable_for_six_hats = self._is_suitable_for_six_hats(thought_content)

                if is_suitable_for_six_hats:
                    logger.info("  ✅ Six Hats selected - optimal for this thought type")
                    return [self.six_hats_step]

                logger.info("  ❌ Six Hats not suitable - falling back to original strategies")

            # Fall back to original complexity-based routing
            return self._original_complexity_selector(step_input)

        except Exception as e:
            logger.error(f"Error in enhanced selector: {e}")
            # Ultimate fallback
            return [self.single_agent_step]

    def _is_suitable_for_six_hats(self, thought_content: str) -> bool:
        """Quick heuristic to determine if Six Hats would be beneficial."""
        suitable = is_suitable_for_six_hats(thought_content)

        # Log details for debugging
        text_lower = thought_content.lower()
        indicator_count = sum(1 for indicator in ['creative', 'decide', 'evaluate', 'meaning', 'problem', '创新', '选择', '评估', '哲学', '问题'] if indicator in text_lower)
        has_questions = '?' in thought_content or '？' in thought_content
        is_complex_length = len(thought_content) > SixHatsConfiguration.MIN_COMPLEX_LENGTH

        logger.info(f"    📊 Suitability: indicators={indicator_count}, questions={has_questions}, complex_length={is_complex_length} → {suitable}")

        return suitable

    def _original_complexity_selector(self, step_input: StepInput) -> List[Step]:
        """Original complexity-based selector (fallback when Six Hats not suitable)."""
        try:
            logger.info("🧭 WORKFLOW ROUTING ANALYSIS:")

            # Extract thought content from StepInput
            if isinstance(step_input.input, dict):
                thought_content = step_input.input.get("thought", "")
                thought_number = step_input.input.get("thought_number", 1)
                total_thoughts = step_input.input.get("total_thoughts", 1)
            else:
                thought_content = str(step_input.input)
                thought_number = 1  # Default fallback
                total_thoughts = 1  # Default fallback

            logger.info(f"  📝 Input: {thought_content[:100]}{'...' if len(thought_content) > 100 else ''}")
            logger.info(f"  🔢 Progress: {thought_number}/{total_thoughts}")

            # Perform complexity analysis (no caching in selector)
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

            logger.info("  🔍 Analyzing complexity...")
            complexity_metrics = self.complexity_analyzer.analyze(thought_data)
            complexity_score = complexity_metrics.complexity_score
            complexity_level = self._determine_complexity_level(complexity_score)

            logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")
            logger.info(f"  📈 Complexity Level: {complexity_level.value}")

            # Determine strategy - simplified to 3 core strategies
            if complexity_level == ComplexityLevel.SIMPLE:
                strategy = "single_agent"
                selected_step = self.single_agent_step
                logger.info("  🤖 Route: Single Agent (Efficient processing)")
            elif complexity_level == ComplexityLevel.MODERATE:
                strategy = "hybrid"
                selected_step = self.hybrid_team_step
                logger.info("  🤝 Route: Hybrid Team (Balanced processing)")
            else:  # COMPLEX or HIGHLY_COMPLEX -> use multi_agent
                strategy = "multi_agent"
                selected_step = self.full_team_step
                logger.info("  👥 Route: Multi Agent (Comprehensive processing)")

            logger.info("🎯 ROUTING DECISION:")
            logger.info(
                f"  ✅ Selected: {strategy} (score={complexity_score:.1f}, "
                f"thought={thought_number}/{total_thoughts})"
            )

            return [selected_step]

        except Exception as e:
            logger.error(f"Error in complexity selector: {e}")
            logger.warning("Retrying with simplified complexity analysis")

            # RETRY MECHANISM: Use simplified complexity analysis instead of direct fallback
            try:
                # Extract basic content for simplified analysis
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get("thought", "")
                else:
                    thought_content = str(step_input.input)

                # Simplified complexity assessment using centralized logic
                content_length = len(thought_content)
                keyword_count = count_complex_keywords(thought_content)

                # Use retry configuration constants for thresholds
                if (content_length < RetryConfiguration.SIMPLE_CONTENT_LENGTH_THRESHOLD and
                    keyword_count <= RetryConfiguration.MAX_KEYWORD_COUNT_SIMPLE):
                    strategy = "single_agent"
                    selected_step = self.single_agent_step
                    logger.info("  🤖 Retry Route: Single Agent (simplified analysis)")
                elif (content_length < RetryConfiguration.MODERATE_CONTENT_LENGTH_THRESHOLD and
                      keyword_count <= RetryConfiguration.MAX_KEYWORD_COUNT_MODERATE):
                    strategy = "hybrid"
                    selected_step = self.hybrid_team_step
                    logger.info("  🤝 Retry Route: Hybrid Team (simplified analysis)")
                else:
                    strategy = "multi_agent"
                    selected_step = self.full_team_step
                    logger.info("  👥 Retry Route: Multi Agent (simplified analysis)")

                logger.info(f"  ✅ Retry successful: {strategy} (length={content_length}, keywords={keyword_count})")
                return [selected_step]

            except Exception as retry_error:
                logger.error(f"Retry also failed: {retry_error}")
                logger.warning("Final fallback to single agent")
                return [self.single_agent_step]

    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score using centralized thresholds."""
        match score:
            case s if s < ComplexityThresholds.SIMPLE_MAX:
                return ComplexityLevel.SIMPLE
            case s if s < ComplexityThresholds.MODERATE_MAX:
                return ComplexityLevel.MODERATE
            case s if s < ComplexityThresholds.COMPLEX_MAX:
                return ComplexityLevel.COMPLEX
            case _:
                return ComplexityLevel.HIGHLY_COMPLEX

    async def process_thought_workflow(
        self, thought_data: ThoughtData, context_prompt: str
    ) -> WorkflowResult:
        """
        Process thought using Agno workflow orchestration.

        This is the main entry point that follows Agno best practices.
        """
        start_time = time.time()

        try:
            logger.info("🚀 AGNO WORKFLOW INITIALIZATION:")
            logger.info(f"  📝 Thought: {thought_data.thought[:100]}{'...' if len(thought_data.thought) > 100 else ''}")
            logger.info(f"  🔢 Thought Number: {thought_data.thoughtNumber}/{thought_data.totalThoughts}")
            logger.info(f"  📋 Context Length: {len(context_prompt)} chars")
            logger.info(f"  ⏰ Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

            # Prepare workflow input as dictionary (Agno standard)
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

            logger.info(
                f"▶️  EXECUTING Agno workflow for thought #{thought_data.thoughtNumber}"
            )

            # Execute Agno workflow with session_state
            logger.info("🔄 WORKFLOW EXECUTION START...")
            result = await self.workflow.arun(
                input=workflow_input, session_state=session_state
            )
            logger.info("✅ WORKFLOW EXECUTION COMPLETED")

            processing_time = time.time() - start_time

            # Extract result content - handle nested structures and object representations
            def extract_clean_content(obj, depth=0):
                """Recursively extract clean content from nested objects."""
                # Prevent infinite recursion
                if depth > 10:
                    return str(obj)

                # Handle dictionary with 'result' key (common wrapper)
                if isinstance(obj, dict):
                    if "result" in obj:
                        return extract_clean_content(obj["result"], depth + 1)
                    elif "content" in obj:
                        return extract_clean_content(obj["content"], depth + 1)
                    else:
                        # Try to find any meaningful string content in the dict
                        for key, value in obj.items():
                            if isinstance(value, str) and len(value.strip()) > 10:
                                # Skip technical keys, prefer content-like keys
                                if key.lower() in [
                                    "message",
                                    "text",
                                    "response",
                                    "output",
                                    "answer",
                                ]:
                                    return value.strip()
                        # Fallback to any string content
                        for value in obj.values():
                            if isinstance(value, str) and len(value.strip()) > 10:
                                return value.strip()
                        return str(obj)

                # Handle RunOutput or TeamRunOutput objects
                if hasattr(obj, "content"):
                    content = obj.content
                    if isinstance(content, str):
                        return content.strip()
                    else:
                        return extract_clean_content(content, depth + 1)

                # Handle other output objects
                if hasattr(obj, "output"):
                    return extract_clean_content(obj.output, depth + 1)

                # Handle list/tuple - extract first meaningful content
                if isinstance(obj, (list, tuple)) and obj:
                    for item in obj:
                        result = extract_clean_content(item, depth + 1)
                        if isinstance(result, str) and len(result.strip()) > 10:
                            return result.strip()

                # If it's already a string, clean it up
                if isinstance(obj, str):
                    content = obj.strip()

                    # Remove object representations - more comprehensive patterns
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
                        # Try multiple extraction patterns
                        patterns = [
                            (r"content='([^']*)'", 1),  # Single quotes
                            (r'content="([^"]*)"', 1),  # Double quotes
                            (r"content=([^,)]*)", 1),  # No quotes
                            (
                                r"'result':\s*'([^']*)'",
                                1,
                            ),  # Result in dict with single quotes
                            (
                                r'"result":\s*"([^"]*)"',
                                1,
                            ),  # Result in dict with double quotes
                            (r"'([^']{20,})'", 1),  # Any long string in single quotes
                            (r'"([^"]{20,})"', 1),  # Any long string in double quotes
                        ]

                        import re

                        for pattern, group in patterns:
                            match = re.search(pattern, content)
                            if match:
                                extracted = match.group(group).strip()
                                if len(extracted) > 10:
                                    return extracted

                        # If patterns fail, try to find any meaningful text content
                        # Remove obvious object syntax
                        cleaned = re.sub(r'[{}()"\']', " ", content)
                        cleaned = re.sub(
                            r"\b(RunOutput|TeamRunOutput|StepOutput|content|result|success|error)\b",
                            " ",
                            cleaned,
                        )
                        cleaned = re.sub(r"\s+", " ", cleaned).strip()

                        if len(cleaned) > 20:
                            return cleaned

                    # Return the string as-is if it doesn't look like an object
                    return content

                # Fallback - convert to string and clean
                result = str(obj).strip()
                if len(result) > 20 and not any(
                    result.startswith(pattern)
                    for pattern in ["RunOutput(", "TeamRunOutput(", "StepOutput(", "<"]
                ):
                    return result

                return "Processing completed successfully"

            logger.info("🧹 CONTENT EXTRACTION PHASE:")
            logger.info(f"  📊 Raw result type: {type(result).__name__}")
            logger.info(f"  📏 Raw result length: {len(str(result))} chars")

            content = extract_clean_content(result)

            # Final validation - ensure content is clean
            if not isinstance(content, str):
                content = str(content)

            logger.info("📋 CONTENT VALIDATION:")
            logger.info(f"  ✅ Content extracted successfully: {len(content)} characters")
            logger.info(f"  📝 Content preview: {content[:150]}{'...' if len(content) > 150 else ''}")

            # Get metadata from session_state (set by selector)
            complexity_score = session_state.get("current_complexity_score", 0.0)
            strategy_used = session_state.get("current_strategy", "unknown")

            logger.info("📊 WORKFLOW RESULT COMPILATION:")
            logger.info(f"  🎯 Strategy used: {strategy_used}")
            logger.info(f"  📈 Complexity score: {complexity_score:.1f}")
            logger.info(f"  ⏱️  Processing time: {processing_time:.3f}s")

            workflow_result = WorkflowResult(
                content=content,
                strategy_used=strategy_used,
                processing_time=processing_time,
                complexity_score=complexity_score,
                step_name="workflow_execution",
            )

            logger.info("🎉 WORKFLOW COMPLETION:")
            logger.info(
                f"  ✅ Completed: strategy={strategy_used}, "
                f"time={processing_time:.3f}s, score={complexity_score:.1f}"
            )

            return workflow_result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Workflow execution failed after {processing_time:.3f}s: {e}")

            return WorkflowResult(
                content=f"Error processing thought: {str(e)}",
                strategy_used="error_fallback",
                processing_time=processing_time,
                complexity_score=0.0,
                step_name="error_handling",
            )

    def _create_single_agent_step(self) -> Step:
        """Create single-agent processing step for simple thoughts."""
        model = self.model_config.create_team_model()

        agent = Agent(
            name="SequentialThinkingAgent",
            role="Efficient Thought Processor",
            description="Processes thoughts efficiently without multi-agent overhead",
            model=model,
            tools=[ReasoningTools],
            instructions=[
                "You are processing a thought efficiently.",
                "Provide a focused, clear response.",
                "Include guidance for the next step.",
                "Be concise but helpful.",
            ],
            markdown=False,
        )

        async def single_agent_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                logger.info("🤖 SINGLE AGENT EXECUTION:")

                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get("thought", str(step_input.input))
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1

                # Calculate complexity score for this execution
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
                complexity_metrics = self.complexity_analyzer.analyze(thought_data)
                complexity_score = complexity_metrics.complexity_score

                # Store metadata in session_state for result compilation
                session_state["current_strategy"] = "single_agent"
                session_state["current_complexity_score"] = complexity_score

                logger.info(f"  📥 Input: {thought_content[:100]}...")
                logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")
                logger.info(f"  📈 Strategy: single_agent")
                logger.info(f"  🎯 Agent: {agent.name}")
                logger.info(f"  🧠 Model: {agent.model}")
                logger.info(f"  🚀 Starting single agent processing...")

                # Run the agent with session_state (async + parallel)
                result = await agent.arun(input=thought_content, session_state=session_state)

                logger.info("  ✅ Single agent completed successfully")
                logger.info(f"  📊 Result type: {type(result).__name__}")
                if hasattr(result, "content"):
                    logger.info(f"  📏 Content length: {len(str(result.content))} chars")
                    logger.info(f"  📝 Content preview: {str(result.content)[:200]}...")

                # Track performance in session_state
                self._update_session_state(
                    session_state, "single_agent", "single_agent_completed"
                )

                return self._create_step_output(
                    content=result, strategy="single_agent", session_state=session_state
                )
            except Exception as e:
                logger.error(f"  ❌ Single agent execution failed: {e}")
                return self._handle_execution_error(e, "single_agent")

        return Step(
            name="single_agent_processing",
            executor=single_agent_executor,
            description="Fast single-agent processing for simple thoughts",
        )

    def _create_hybrid_team_step(self) -> Step:
        """Create hybrid team processing step for moderate complexity."""
        model = self.model_config.create_team_model()

        # Selective specialist team using factory
        planner = AgentFactory.create_planner(model, "basic")
        analyzer = AgentFactory.create_analyzer(model, "basic")

        hybrid_team = Team(
            name="HybridSpecialistTeam",
            members=[planner, analyzer],
            model=model,
            respond_directly=False,
            delegate_task_to_all_members=False,
            determine_input_for_members=True,
            instructions=[
                "Coordinate between planning and analysis specialists.",
                "Synthesize insights from both perspectives.",
                "Provide balanced, thoughtful responses.",
            ],
        )

        async def hybrid_team_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get("thought", str(step_input.input))
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1

                # Calculate complexity score for this execution
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
                complexity_metrics = self.complexity_analyzer.analyze(thought_data)
                complexity_score = complexity_metrics.complexity_score

                # Store metadata in session_state for result compilation
                session_state["current_strategy"] = "hybrid"
                session_state["current_complexity_score"] = complexity_score

                logger.info("🤝 HYBRID TEAM EXECUTION:")
                logger.info(f"  📥 Input: {thought_content[:100]}...")
                logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")
                logger.info(f"  📈 Strategy: hybrid")
                logger.info(
                    f"  👥 Team members: {[member.name for member in hybrid_team.members]}"
                )
                logger.info(f"  🧠 Team model: {hybrid_team.model}")

                # Run the team with session_state (async + parallel)
                logger.info("  🚀 Starting hybrid team processing...")
                result = await hybrid_team.arun(
                    input=thought_content, session_state=session_state
                )

                logger.info("  ✅ Hybrid team completed successfully")
                logger.info(f"  📊 Result type: {type(result).__name__}")
                if hasattr(result, "content"):
                    logger.info(
                        f"  📏 Content length: {len(str(result.content))} chars"
                    )
                    logger.info(f"  📝 Content preview: {str(result.content)[:200]}...")

                # Track performance in session_state
                self._update_session_state(
                    session_state, "hybrid", "hybrid_team_completed"
                )

                return self._create_step_output(
                    content=result,
                    strategy="hybrid",
                    session_state=session_state,
                    specialists=["planner", "analyzer"],
                )
            except Exception as e:
                logger.error(f"  ❌ Hybrid team execution failed: {str(e)}")
                return self._handle_execution_error(e, "hybrid team")

        return Step(
            name="hybrid_team_processing",
            executor=hybrid_team_executor,
            description="Balanced processing with selective specialists",
        )

    def _create_full_team_step(self) -> Step:
        """Create full team processing step for complex thoughts."""
        model = self.model_config.create_team_model()

        # Complete specialist team
        specialists = [
            Agent(
                name="Planner",
                role="Strategic Planner",
                model=model,
                tools=[ReasoningTools],
                instructions=[
                    "Create comprehensive strategic approaches.",
                    "Plan multi-step solutions and methodologies.",
                    "Provide strategic oversight and direction.",
                ],
            ),
            Agent(
                name="Researcher",
                role="Information Gatherer",
                model=model,
                tools=[ReasoningTools],
                instructions=[
                    "Gather relevant information and context.",
                    "Verify facts and explore related concepts.",
                    "Provide research-backed insights.",
                ],
            ),
            Agent(
                name="Analyzer",
                role="Core Analyst",
                model=model,
                tools=[ReasoningTools],
                instructions=[
                    "Perform comprehensive analysis of complex thoughts.",
                    "Identify deep patterns and relationships.",
                    "Provide sophisticated analytical perspectives.",
                ],
            ),
            Agent(
                name="Critic",
                role="Quality Controller",
                model=model,
                tools=[ReasoningTools],
                instructions=[
                    "Critically evaluate ideas and proposals.",
                    "Identify potential issues and improvements.",
                    "Ensure quality and logical consistency.",
                ],
            ),
            Agent(
                name="Synthesizer",
                role="Integration Specialist",
                model=model,
                tools=[ReasoningTools],
                instructions=[
                    "Integrate insights from all specialists.",
                    "Create coherent final responses.",
                    "Synthesize multiple perspectives effectively.",
                ],
            ),
        ]

        full_team = Team(
            name="CompleteSpecialistTeam",
            members=specialists,
            model=model,
            respond_directly=False,
            delegate_task_to_all_members=False,
            determine_input_for_members=True,
            enable_user_memories=True,
            instructions=[
                "Coordinate comprehensive multi-specialist analysis.",
                "Ensure all perspectives are considered and integrated.",
                "Provide highest quality responses for complex thoughts.",
            ],
        )

        async def full_team_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                # Extract thought content and metadata
                if isinstance(step_input.input, dict):
                    thought_content = step_input.input.get("thought", str(step_input.input))
                    thought_number = step_input.input.get("thought_number", 1)
                    total_thoughts = step_input.input.get("total_thoughts", 1)
                else:
                    thought_content = str(step_input.input)
                    thought_number = 1
                    total_thoughts = 1

                # Calculate complexity score for this execution
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
                complexity_metrics = self.complexity_analyzer.analyze(thought_data)
                complexity_score = complexity_metrics.complexity_score

                # Store metadata in session_state for result compilation
                session_state["current_strategy"] = "multi_agent"
                session_state["current_complexity_score"] = complexity_score

                logger.info("👥 FULL TEAM EXECUTION:")
                logger.info(f"  📥 Input: {thought_content[:100]}...")
                logger.info(f"  📊 Complexity Score: {complexity_score:.1f}")
                logger.info(f"  📈 Strategy: multi_agent")
                logger.info(f"  👥 Team name: {full_team.name}")
                logger.info(f"  🎯 Team members: {[member.name for member in full_team.members]}")
                logger.info(f"  🧠 Model: {full_team.model}")
                logger.info("  🚀 Starting full team processing...")

                # Run the full team with session_state (async + parallel)
                result = await full_team.arun(
                    input=thought_content, session_state=session_state
                )

                logger.info("  ✅ Full team completed successfully")
                logger.info(f"  📊 Result type: {type(result).__name__}")
                if hasattr(result, "content"):
                    logger.info(f"  📏 Content length: {len(str(result.content))} chars")
                    logger.info(f"  📝 Content preview: {str(result.content)[:200]}...")

                # Track performance in session_state
                session_state["full_team_completed"] = True

                return StepOutput(
                    content=result,
                    success=True,
                )
            except Exception as e:
                logger.error(f"  ❌ Full team execution failed: {str(e)}")
                return StepOutput(
                    content=f"Full team processing failed: {str(e)}",
                    success=False,
                    error=str(e),
                )

        return Step(
            name="full_team_processing",
            executor=full_team_executor,
            description="Maximum quality processing with all specialists",
        )

