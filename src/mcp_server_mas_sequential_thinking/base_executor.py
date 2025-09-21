"""
Base Executor for MAS Sequential Thinking

Eliminates code duplication in workflow executors by providing a common pattern
for input extraction, complexity analysis, and error handling.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, Union

from agno.workflow.types import StepInput, StepOutput
from agno.agent import Agent
from agno.team import Team

from .models import ThoughtData
from .adaptive_routing import ComplexityAnalyzer, BasicComplexityAnalyzer
from .processing_constants import (
    ComplexityThresholds, LoggingLimits, get_complexity_level_name,
    is_content_sufficient_quality
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedInput:
    """Standardized extracted input data."""

    thought_content: str
    thought_number: int = 1
    total_thoughts: int = 1
    context: str = ""
    original_input: Any = None


@dataclass(frozen=True)
class ExecutionMetrics:
    """Metrics collected during execution."""

    start_time: float
    complexity_score: float
    strategy_used: str
    execution_time: float = 0.0
    success: bool = True
    error_message: str = ""


class Processor(Protocol):
    """Protocol for processors that can execute with extracted input."""

    async def arun(self, input: str, **kwargs) -> Any:
        """Run the processor with input and return result."""
        ...


class BaseExecutor(ABC):
    """Base class for workflow executors that eliminates duplication."""

    def __init__(
        self,
        strategy_name: str,
        complexity_analyzer: Optional[ComplexityAnalyzer] = None
    ):
        self.strategy_name = strategy_name
        self.complexity_analyzer = complexity_analyzer or BasicComplexityAnalyzer()

    async def execute(
        self,
        step_input: StepInput,
        session_state: Dict[str, Any],
        processor: Union[Agent, Team, Processor]
    ) -> StepOutput:
        """Execute the processor with standardized pattern."""
        start_time = time.time()

        try:
            # Step 1: Extract and validate input
            extracted = self._extract_input(step_input)
            self._log_execution_start(extracted)

            # Step 2: Analyze complexity
            complexity_score = self._analyze_complexity(extracted)
            self._log_complexity_analysis(complexity_score)

            # Step 3: Execute processor
            result = await self._execute_processor(processor, extracted)

            # Step 4: Extract and validate content
            content = self._extract_content(result)
            self._validate_content_quality(content)

            # Step 5: Update session state and create output
            execution_time = time.time() - start_time
            self._update_session_state(session_state, complexity_score, execution_time)
            self._log_execution_success(content, execution_time)

            return StepOutput(
                content=content,
                success=True,
                step_name=f"{self.strategy_name}_execution"
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = str(e)[:LoggingLimits.MAX_ERROR_LOG_LENGTH]
            self._log_execution_error(error_msg, execution_time)

            return StepOutput(
                content=f"{self.strategy_name} execution failed: {error_msg}",
                success=False,
                error=error_msg,
                step_name=f"{self.strategy_name}_error"
            )

    def _extract_input(self, step_input: StepInput) -> ExtractedInput:
        """Extract and standardize input data."""
        if isinstance(step_input.input, dict):
            return ExtractedInput(
                thought_content=step_input.input.get("thought", str(step_input.input)),
                thought_number=step_input.input.get("thought_number", 1),
                total_thoughts=step_input.input.get("total_thoughts", 1),
                context=step_input.input.get("context", ""),
                original_input=step_input.input
            )
        else:
            content = str(step_input.input)
            return ExtractedInput(
                thought_content=content,
                original_input=step_input.input
            )

    def _analyze_complexity(self, extracted: ExtractedInput) -> float:
        """Analyze complexity of the thought content."""
        thought_data = ThoughtData(
            thought=extracted.thought_content,
            thoughtNumber=extracted.thought_number,
            totalThoughts=extracted.total_thoughts,
            nextThoughtNeeded=True,
            isRevision=False,
            branchFromThought=None,
            branchId=None,
            needsMoreThoughts=False,
        )

        complexity_metrics = self.complexity_analyzer.analyze(thought_data)
        return complexity_metrics.complexity_score

    async def _execute_processor(
        self,
        processor: Union[Agent, Team, Processor],
        extracted: ExtractedInput
    ) -> Any:
        """Execute the processor with extracted input."""
        # Add strategy-specific context to input
        enhanced_input = self._enhance_input_for_strategy(extracted)
        return await processor.arun(input=enhanced_input)

    def _extract_content(self, result: Any) -> str:
        """Extract clean content from processor result."""
        # Handle different result types with early returns
        if isinstance(result, str):
            return result.strip()

        if hasattr(result, 'content'):
            content = result.content
            if isinstance(content, str):
                return content.strip()
            else:
                return str(content).strip()

        if isinstance(result, dict):
            # Try common content keys
            for key in ['content', 'result', 'output', 'response']:
                if key in result:
                    content = result[key]
                    if isinstance(content, str):
                        return content.strip()

        # Fallback to string representation
        return str(result).strip()

    def _validate_content_quality(self, content: str) -> None:
        """Validate content meets quality requirements."""
        if not is_content_sufficient_quality(content):
            logger.warning(f"Content quality concerns for {self.strategy_name}: length={len(content)}")

    def _update_session_state(
        self,
        session_state: Dict[str, Any],
        complexity_score: float,
        execution_time: float
    ) -> None:
        """Update session state with execution metrics."""
        session_state.update({
            "current_strategy": self.strategy_name,
            "current_complexity_score": complexity_score,
            "current_complexity_level": get_complexity_level_name(complexity_score),
            "last_execution_time": execution_time,
        })

    @abstractmethod
    def _enhance_input_for_strategy(self, extracted: ExtractedInput) -> str:
        """Enhance input with strategy-specific context."""
        pass

    def _log_execution_start(self, extracted: ExtractedInput) -> None:
        """Log execution start with truncated content."""
        content_preview = extracted.thought_content[:LoggingLimits.MAX_INPUT_LOG_LENGTH]
        if len(extracted.thought_content) > LoggingLimits.MAX_INPUT_LOG_LENGTH:
            content_preview += "..."

        logger.info(f"🚀 {self.strategy_name.upper()} EXECUTION:")
        logger.info(f"  📝 Input: {content_preview}")
        logger.info(f"  🔢 Progress: {extracted.thought_number}/{extracted.total_thoughts}")

    def _log_complexity_analysis(self, complexity_score: float) -> None:
        """Log complexity analysis results."""
        complexity_level = get_complexity_level_name(complexity_score)
        logger.info(f"  📊 Complexity: {complexity_score:.1f} ({complexity_level})")

    def _log_execution_success(self, content: str, execution_time: float) -> None:
        """Log successful execution."""
        content_preview = content[:LoggingLimits.MAX_OUTPUT_LOG_LENGTH]
        if len(content) > LoggingLimits.MAX_OUTPUT_LOG_LENGTH:
            content_preview += "..."

        logger.info(f"  ✅ Success: {self.strategy_name} completed in {execution_time:.3f}s")
        logger.info(f"  📄 Output: {content_preview}")

    def _log_execution_error(self, error_msg: str, execution_time: float) -> None:
        """Log execution error."""
        logger.error(f"  ❌ Error: {self.strategy_name} failed after {execution_time:.3f}s")
        logger.error(f"  🚨 Details: {error_msg}")


class SingleAgentExecutor(BaseExecutor):
    """Executor for single agent processing."""

    def __init__(self, complexity_analyzer: Optional[ComplexityAnalyzer] = None):
        super().__init__("single_agent", complexity_analyzer)

    def _enhance_input_for_strategy(self, extracted: ExtractedInput) -> str:
        """Enhance input for single agent processing."""
        base_input = extracted.thought_content

        if extracted.context:
            return f"Context: {extracted.context}\n\nThought: {base_input}"

        if extracted.total_thoughts > 1:
            return (
                f"Thought {extracted.thought_number} of {extracted.total_thoughts}: "
                f"{base_input}"
            )

        return base_input


class HybridTeamExecutor(BaseExecutor):
    """Executor for hybrid team processing."""

    def __init__(self, complexity_analyzer: Optional[ComplexityAnalyzer] = None):
        super().__init__("hybrid_team", complexity_analyzer)

    def _enhance_input_for_strategy(self, extracted: ExtractedInput) -> str:
        """Enhance input for hybrid team processing."""
        enhanced_parts = [
            f"Sequential thought {extracted.thought_number} of {extracted.total_thoughts}:",
            extracted.thought_content,
        ]

        if extracted.context:
            enhanced_parts.insert(1, f"Context: {extracted.context}")

        enhanced_parts.append("Please coordinate between team members for a balanced analysis.")

        return "\n\n".join(enhanced_parts)


class MultiAgentExecutor(BaseExecutor):
    """Executor for multi-agent team processing."""

    def __init__(self, complexity_analyzer: Optional[ComplexityAnalyzer] = None):
        super().__init__("multi_agent", complexity_analyzer)

    def _enhance_input_for_strategy(self, extracted: ExtractedInput) -> str:
        """Enhance input for multi-agent processing."""
        enhanced_parts = [
            f"Complex sequential thought {extracted.thought_number} of {extracted.total_thoughts}:",
            extracted.thought_content,
        ]

        if extracted.context:
            enhanced_parts.insert(1, f"Context: {extracted.context}")

        enhanced_parts.extend([
            "",
            "This requires comprehensive multi-agent analysis.",
            "Each specialist should contribute their expertise:",
            "- Strategic planning and roadmap development",
            "- In-depth research and information gathering",
            "- Critical analysis and evaluation",
            "- Quality assessment and improvement suggestions",
            "- Final synthesis and integration"
        ])

        return "\n".join(enhanced_parts)