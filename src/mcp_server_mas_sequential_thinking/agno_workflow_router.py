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
            session_state["processing_strategy"] = strategy

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

        # Create processing steps
        self.single_agent_step = self._create_single_agent_step()
        self.hybrid_team_step = self._create_hybrid_team_step()
        self.full_team_step = self._create_full_team_step()
        self.parallel_analysis_step = self._create_parallel_analysis_step()

        # Create complexity-based router with custom selector
        self.complexity_router = Router(
            name="complexity_based_router",
            selector=self._complexity_selector,
            choices=[
                self.single_agent_step,
                self.hybrid_team_step,
                self.full_team_step,
                self.parallel_analysis_step,
            ],
        )

        # Create main workflow - single step to avoid condition override
        self.workflow = Workflow(
            name="adaptive_sequential_thinking_workflow",
            steps=[
                self.complexity_router  # Quality improvement handled within router
            ],
        )

        logger.info("AgnoWorkflowRouter initialized with Workflow + Router pattern")

    def _quality_evaluator(self, step_input: StepInput) -> bool:
        """
        Evaluate if additional quality improvement is needed.

        Returns True if quality improvement steps should run.
        """
        try:
            # Get previous step output for quality assessment
            previous_content = step_input.previous_step_content or ""

            # Simple quality heuristics
            quality_issues = []

            # Check content length (too short might indicate insufficient analysis)
            if len(previous_content.strip()) < 100:
                quality_issues.append("content_too_short")

            # Check for error indicators
            if any(
                error_keyword in previous_content.lower()
                for error_keyword in ["failed", "error", "could not", "unable to"]
            ):
                quality_issues.append("contains_errors")

            # Check for insufficient depth (simplified without complexity score)
            if len(previous_content.strip()) < 200:
                quality_issues.append("insufficient_depth")

            # Note: Cannot store in session_state since evaluators don't have access

            # Return True if quality improvement is needed
            needs_improvement = len(quality_issues) > 0

            if needs_improvement:
                logger.info(f"Quality issues detected: {quality_issues}")
            else:
                logger.info("Quality check passed - no improvement needed")

            return needs_improvement

        except Exception as e:
            logger.error(f"Error in quality evaluator: {e}")
            # If evaluation fails, don't run improvement (fail safe)
            return False

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

        def quality_improvement_executor(
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

                result = quality_improver.run(
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

    def _complexity_selector(self, step_input: StepInput) -> List[Step]:
        """
        Agno-standard selector function for complexity-based routing.

        Router selectors only receive StepInput, not session_state.
        """
        try:
            # Extract thought content from StepInput
            if isinstance(step_input.input, dict):
                thought_content = step_input.input.get("thought", "")
                thought_number = step_input.input.get("thought_number", 1)
                total_thoughts = step_input.input.get("total_thoughts", 1)
            else:
                thought_content = str(step_input.input)
                thought_number = 1  # Default fallback
                total_thoughts = 1  # Default fallback

            # Perform complexity analysis (no caching in selector)
            thought_data = ThoughtData(
                thought=thought_content,
                thought_number=thought_number,
                total_thoughts=total_thoughts,
                next_needed=True,
            )

            complexity_metrics = self.complexity_analyzer.analyze(thought_data)
            complexity_score = complexity_metrics.complexity_score
            complexity_level = self._determine_complexity_level(complexity_score)

            # Determine strategy
            if complexity_level == ComplexityLevel.SIMPLE:
                strategy = "single_agent"
                selected_step = self.single_agent_step
            elif complexity_level == ComplexityLevel.MODERATE:
                strategy = "hybrid"
                selected_step = self.hybrid_team_step
            elif complexity_level == ComplexityLevel.COMPLEX:
                strategy = "multi_agent"
                selected_step = self.full_team_step
            else:  # HIGHLY_COMPLEX
                strategy = "parallel_analysis"
                selected_step = self.parallel_analysis_step

            logger.info(
                f"Workflow routing: score={complexity_score:.1f}, "
                f"strategy={strategy}, thought={thought_number}/{total_thoughts}"
            )

            return [selected_step]

        except Exception as e:
            logger.error(f"Error in complexity selector: {e}")
            # Fallback to single agent for reliability
            return [self.single_agent_step]

    def _determine_complexity_level(self, score: float) -> ComplexityLevel:
        """Determine complexity level from score (adjusted for realistic ranges)."""
        if score < 5:
            return ComplexityLevel.SIMPLE
        elif score < 15:
            return ComplexityLevel.MODERATE
        elif score < 25:
            return ComplexityLevel.COMPLEX
        else:
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
            # Prepare workflow input as dictionary (Agno standard)
            workflow_input = {
                "thought": thought_data.thought,
                "thought_number": thought_data.thought_number,
                "total_thoughts": thought_data.total_thoughts,
                "context": context_prompt,
            }

            # Initialize session_state for metadata tracking
            session_state = {
                "start_time": start_time,
                "thought_number": thought_data.thought_number,
                "total_thoughts": thought_data.total_thoughts,
            }

            logger.info(
                f"Executing Agno workflow for thought #{thought_data.thought_number}"
            )

            # Execute Agno workflow with session_state
            result = await self.workflow.arun(
                input=workflow_input, session_state=session_state
            )

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

            content = extract_clean_content(result)

            # Final validation - ensure content is clean
            if not isinstance(content, str):
                content = str(content)

            # Log successful extraction for debugging
            logger.info(f"✅ Content extracted successfully: {len(content)} characters")
            logger.debug(
                f"📝 Content preview: {content[:100]}{'...' if len(content) > 100 else ''}"
            )

            # Get metadata from session_state (set by selector)
            complexity_score = session_state.get("current_complexity_score", 0.0)
            strategy_used = session_state.get("current_strategy", "unknown")

            workflow_result = WorkflowResult(
                content=content,
                strategy_used=strategy_used,
                processing_time=processing_time,
                complexity_score=complexity_score,
                step_name="workflow_execution",
            )

            logger.info(
                f"Workflow completed: strategy={strategy_used}, "
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

        def single_agent_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                # Extract thought content
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )

                # Run the agent with session_state
                result = agent.run(input=thought_content, session_state=session_state)

                # Track performance in session_state
                self._update_session_state(
                    session_state, "single_agent", "single_agent_completed"
                )

                return self._create_step_output(
                    content=result, strategy="single_agent", session_state=session_state
                )
            except Exception as e:
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

        def hybrid_team_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                # Extract thought content
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )

                logger.info("🤖 HYBRID TEAM EXECUTION:")
                logger.info(f"  📥 Input: {thought_content[:100]}...")
                logger.info(
                    f"  👥 Team members: {[member.name for member in hybrid_team.members]}"
                )

                # Run the team with session_state
                logger.info("  🚀 Starting hybrid team processing...")
                result = hybrid_team.run(
                    input=thought_content, session_state=session_state
                )

                logger.info("  ✅ Hybrid team completed successfully")
                logger.info(f"  📊 Result type: {type(result).__name__}")
                if hasattr(result, "content"):
                    logger.info(
                        f"  📏 Content length: {len(str(result.content))} chars"
                    )

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

        def full_team_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Custom executor that ensures StepOutput compliance."""
            try:
                # Extract thought content
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )

                # Run the full team with session_state
                result = full_team.run(
                    input=thought_content, session_state=session_state
                )

                # Track performance in session_state
                session_state["full_team_completed"] = True
                session_state["processing_strategy"] = "multi_agent"

                return StepOutput(
                    content=result,
                    success=True,
                )
            except Exception as e:
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

    def _create_parallel_analysis_step(self) -> Parallel:
        """Create parallel analysis step for highly complex thoughts."""
        model = self.model_config.create_team_model()

        # Create specialized agents for parallel analysis
        semantic_analyzer = Agent(
            name="SemanticAnalyzer",
            role="Semantic Analysis Specialist",
            model=model,
            tools=[ReasoningTools],
            instructions=[
                "Focus on semantic meaning and conceptual relationships.",
                "Analyze language patterns and meaning structures.",
                "Identify key concepts and their interconnections.",
            ],
        )

        technical_analyzer = Agent(
            name="TechnicalAnalyzer",
            role="Technical Analysis Specialist",
            model=model,
            tools=[ReasoningTools],
            instructions=[
                "Analyze technical aspects and implementation details.",
                "Focus on methodology and technical feasibility.",
                "Evaluate technical constraints and requirements.",
            ],
        )

        context_analyzer = Agent(
            name="ContextAnalyzer",
            role="Contextual Analysis Specialist",
            model=model,
            tools=[ReasoningTools],
            instructions=[
                "Analyze contextual implications and broader impact.",
                "Consider historical context and situational factors.",
                "Evaluate environmental and situational influences.",
            ],
        )

        def semantic_executor(step_input: StepInput, session_state: dict) -> StepOutput:
            """Semantic analysis executor."""
            try:
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )
                result = semantic_analyzer.run(
                    input=f"Perform semantic analysis of: {thought_content}",
                    session_state=session_state,
                )
                return StepOutput(
                    content=result,
                    success=True,
                )
            except Exception as e:
                return StepOutput(
                    content=f"Semantic analysis failed: {str(e)}",
                    success=False,
                    error=str(e),
                )

        def technical_executor(
            step_input: StepInput, session_state: dict
        ) -> StepOutput:
            """Technical analysis executor."""
            try:
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )
                result = technical_analyzer.run(
                    input=f"Perform technical analysis of: {thought_content}",
                    session_state=session_state,
                )
                return StepOutput(
                    content=result,
                    success=True,
                )
            except Exception as e:
                return StepOutput(
                    content=f"Technical analysis failed: {str(e)}",
                    success=False,
                    error=str(e),
                )

        def context_executor(step_input: StepInput, session_state: dict) -> StepOutput:
            """Contextual analysis executor."""
            try:
                thought_content = (
                    step_input.input.get("thought", str(step_input.input))
                    if isinstance(step_input.input, dict)
                    else str(step_input.input)
                )
                result = context_analyzer.run(
                    input=f"Perform contextual analysis of: {thought_content}",
                    session_state=session_state,
                )
                return StepOutput(
                    content=result,
                    success=True,
                )
            except Exception as e:
                return StepOutput(
                    content=f"Contextual analysis failed: {str(e)}",
                    success=False,
                    error=str(e),
                )

        # Create parallel step with independent analysis tasks
        return Parallel(
            Step(name="semantic_analysis", executor=semantic_executor),
            Step(name="technical_analysis", executor=technical_executor),
            Step(name="contextual_analysis", executor=context_executor),
            name="parallel_complex_analysis",
            description="Parallel multi-dimensional analysis for highly complex thoughts",
        )
