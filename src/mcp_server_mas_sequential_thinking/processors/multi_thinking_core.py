"""Multi-Thinking Core Agent Architecture.

Multi-dimensional thinking direction Agent core implementation.
Strictly follows "focused thinking direction" principles, supporting intelligent sequences from single to multiple thinking modes.
"""

# Lazy import to break circular dependency
import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

from agno.agent import Agent
from agno.models.base import Model
from agno.tools.reasoning import ReasoningTools

logger = logging.getLogger(__name__)


class ThinkingDirection(Enum):
    """Multi-thinking direction enumeration."""

    FACTUAL = "factual"  # Facts and data
    EMOTIONAL = "emotional"  # Emotions and intuition
    CRITICAL = "critical"  # Criticism and risk
    OPTIMISTIC = "optimistic"  # Optimism and value
    CREATIVE = "creative"  # Creativity and innovation
    SYNTHESIS = "synthesis"  # Metacognition and integration


class ProcessingDepth(Enum):
    """Processing complexity levels."""

    SINGLE = "single"  # Single thinking mode
    DOUBLE = "double"  # Double thinking sequence
    TRIPLE = "triple"  # Triple thinking sequence
    FULL = "full"  # Complete multi-thinking


@dataclass(frozen=True)
class ThinkingTimingConfig:
    """Thinking direction timing configuration."""

    direction: ThinkingDirection
    default_time_seconds: int
    min_time_seconds: int
    max_time_seconds: int
    is_quick_reaction: bool = False  # Whether it's a quick reaction mode (like emotional thinking)


# Timing configuration constants
THINKING_TIMING_CONFIGS = {
    ThinkingDirection.FACTUAL: ThinkingTimingConfig(ThinkingDirection.FACTUAL, 120, 60, 300, False),
    ThinkingDirection.EMOTIONAL: ThinkingTimingConfig(ThinkingDirection.EMOTIONAL, 30, 15, 60, True),  # Quick intuition
    ThinkingDirection.CRITICAL: ThinkingTimingConfig(ThinkingDirection.CRITICAL, 120, 60, 240, False),
    ThinkingDirection.OPTIMISTIC: ThinkingTimingConfig(ThinkingDirection.OPTIMISTIC, 120, 60, 240, False),
    ThinkingDirection.CREATIVE: ThinkingTimingConfig(
        ThinkingDirection.CREATIVE, 240, 120, 360, False
    ),  # Creativity requires more time
    ThinkingDirection.SYNTHESIS: ThinkingTimingConfig(ThinkingDirection.SYNTHESIS, 60, 30, 120, False),
}


@dataclass(frozen=True)
class ThinkingCapability:
    """Multi-thinking capability definition."""

    thinking_direction: ThinkingDirection
    role: str
    description: str
    role_description: str

    # Cognitive characteristics
    thinking_mode: str
    cognitive_focus: str
    output_style: str

    # Time management
    timing_config: ThinkingTimingConfig

    # Enhanced features
    tools: list[Any] | None = None
    reasoning_level: int = 1
    memory_enabled: bool = False

    def __post_init__(self):
        if self.tools is None:
            object.__setattr__(self, "tools", [ReasoningTools])

    def get_instructions(
        self, context: str = "", previous_results: dict | None = None
    ) -> list[str]:
        """Generate specific analysis mode instructions."""
        base_instructions = [
            f"You are operating in {self.thinking_mode} analysis mode.",
            f"Role: {self.role}",
            f"Cognitive Focus: {self.cognitive_focus}",
            "",
            "CORE PRINCIPLES:",
            f"1. Apply ONLY {self.thinking_mode} approach - maintain strict focus",
            f"2. Time allocation: {self.timing_config.default_time_seconds} seconds for thorough analysis",
            f"3. Output approach: {self.output_style}",
            "",
            f"Your specific responsibility: {self.role_description}",
        ]

        # Add specific analysis mode detailed instructions
        specific_instructions = self._get_specific_instructions()
        base_instructions.extend(specific_instructions)

        # Add context and previous results
        if context:
            base_instructions.extend(
                [
                    "",
                    f"Context: {context}",
                ]
            )

        if previous_results:
            base_instructions.extend(
                [
                    "",
                    "Previous analysis insights from other perspectives:",
                    *[
                        f"  {self._format_previous_result_label(direction_name)}: {result[:100]}..."
                        for direction_name, result in previous_results.items()
                    ],
                ]
            )

        return base_instructions

    def _format_previous_result_label(self, direction_name: str) -> str:
        """Format previous result labels, using thinking direction concepts."""
        label_mapping = {
            "factual": "Factual analysis",
            "emotional": "Emotional perspective",
            "critical": "Critical evaluation",
            "optimistic": "Optimistic view",
            "creative": "Creative thinking",
            "synthesis": "Strategic synthesis"
        }
        return label_mapping.get(direction_name.lower(), "Analysis")

    @abstractmethod
    def _get_specific_instructions(self) -> list[str]:
        """Get specific thinking direction detailed instructions."""


class FactualThinkingCapability(ThinkingCapability):
    """Factual thinking capability: facts and data."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.FACTUAL,
            role="Factual Information Processor",
            description="Collect and process objective facts and data",
            role_description="I focus only on objective facts and data. I provide neutral information without personal interpretation.",
            thinking_mode="analytical_factual",
            cognitive_focus="Pure information processing, zero emotion or judgment",
            output_style="Objective fact lists, data-driven information",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.FACTUAL],
            reasoning_level=2,
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "FACTUAL ANALYSIS GUIDELINES:",
            "• Use simple statements to present facts: 'The data shows...', 'Known information is...'",
            "• Avoid technical jargon, explain data in everyday language",
            "• Present only verified facts and objective data",
            "• Avoid opinions, interpretations, or emotional reactions",
            "• Identify what information is missing and needed",
            "• Separate facts from assumptions clearly",
            "",
            "FORBIDDEN in factual analysis mode:",
            "- Personal opinions or judgments",
            "- Emotional responses or gut feelings",
            "- Speculation or 'what if' scenarios",
            "- Value judgments (good/bad, right/wrong)",
        ]


class EmotionalThinkingCapability(ThinkingCapability):
    """Emotional thinking capability: emotions and intuition."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.EMOTIONAL,
            role="Intuitive Emotional Processor",
            description="Provide emotional responses and intuitive insights",
            role_description="I express intuition and emotional reactions. No reasoning needed, just share feelings.",
            thinking_mode="intuitive_emotional",
            cognitive_focus="Emotional intelligence and intuitive processing",
            output_style="Intuitive reactions, emotional expression, humanized perspective",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.EMOTIONAL],
            reasoning_level=1,  # Lowest rationality, highest intuition
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "EMOTIONAL INTUITIVE GUIDELINES:",
            "• Start responses with 'I feel...', 'My intuition tells me...', 'My gut reaction is...'",
            "• Keep expressions brief and powerful - 30-second emotional snapshots",
            "• Express immediate gut reactions and feelings",
            "• Share intuitive hunches without justification",
            "• Include visceral, immediate responses",
            "• NO need to explain or justify feelings",
            "",
            "ENCOURAGED in emotional intuitive mode:",
            "- First impressions and gut reactions",
            "- Emotional responses to ideas or situations",
            "- Intuitive concerns or excitement",
            "- 'Sixth sense' about what might work",
            "",
            "Remember: This is a 30-second emotional snapshot, not analysis!",
        ]


class CriticalThinkingCapability(ThinkingCapability):
    """Critical thinking capability: criticism and risk."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.CRITICAL,
            role="Critical Risk Assessor",
            description="Critical analysis and risk identification",
            role_description="I identify risks and problems. Critical but not pessimistic, I point out real difficulties.",
            thinking_mode="critical_analytical",
            cognitive_focus="Critical thinking and risk assessment",
            output_style="Sharp questioning, risk warnings, logical verification",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.CRITICAL],
            reasoning_level=3,  # High logical reasoning
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "CRITICAL ASSESSMENT GUIDELINES:",
            "• Point out specific possible problems, not general pessimism",
            "• Use phrases like 'The risk is...', 'This could fail because...', 'A problem might be...'",
            "• Identify potential problems, risks, and weaknesses",
            "• Challenge assumptions and look for logical flaws",
            "• Consider worst-case scenarios and failure modes",
            "• Provide logical reasons for all concerns raised",
            "",
            "KEY AREAS TO EXAMINE:",
            "- Logical inconsistencies in arguments",
            "- Practical obstacles and implementation challenges",
            "- Resource constraints and limitations",
            "- Potential negative consequences",
            "- Missing information or unproven assumptions",
            "",
            "Note: Be critical but constructive - identify real problems, not just pessimism.",
        ]


class OptimisticThinkingCapability(ThinkingCapability):
    """Optimistic thinking capability: optimism and value."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.OPTIMISTIC,
            role="Optimistic Value Explorer",
            description="Positive thinking and value discovery",
            role_description="I find value and opportunities. Realistic optimism, I discover genuine benefits.",
            thinking_mode="optimistic_constructive",
            cognitive_focus="Positive psychology and opportunity identification",
            output_style="Positive exploration, value discovery, opportunity identification",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.OPTIMISTIC],
            reasoning_level=2,
            memory_enabled=False,
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "OPTIMISTIC VALUE EXPLORATION GUIDELINES:",
            "• Point out specific feasible benefits, not empty praise",
            "• Use phrases like 'The benefit is...', 'This creates... value', 'An opportunity here is...'",
            "• Focus on benefits, values, and positive outcomes",
            "• Explore best-case scenarios and opportunities",
            "• Identify feasible positive possibilities",
            "• Provide logical reasons for optimism",
            "",
            "KEY AREAS TO EXPLORE:",
            "- Benefits and positive outcomes",
            "- Opportunities for growth or improvement",
            "- Feasible best-case scenarios",
            "- Value creation possibilities",
            "- Strengths and positive aspects",
            "- Why this could work well",
            "",
            "Note: Be realistically optimistic - find genuine value, not false hope.",
        ]


class CreativeThinkingCapability(ThinkingCapability):
    """Creative thinking capability: creativity and innovation."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.CREATIVE,
            role="Creative Innovation Generator",
            description="Creative thinking and innovative solutions",
            role_description="I generate new ideas and alternative approaches. I break conventional limits and explore possibilities.",
            thinking_mode="creative_generative",
            cognitive_focus="Divergent thinking and creativity",
            output_style="Novel ideas, innovative solutions, alternative thinking",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.CREATIVE],
            reasoning_level=2,
            memory_enabled=True,  # Creativity may require memory combination
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "CREATIVE INNOVATION GUIDELINES:",
            "• Provide 3-5 specific creative ideas that could work",
            "• Use phrases like 'What if...', 'Another approach could be...', 'An alternative is...'",
            "• Generate new ideas, alternatives, and creative solutions",
            "• Think laterally - explore unconventional approaches",
            "• Break normal thinking patterns and assumptions",
            "• Suggest modifications, improvements, or entirely new approaches",
            "",
            "CREATIVE TECHNIQUES TO USE:",
            "- Lateral thinking and analogies",
            "- Random word associations",
            "- 'What if' scenarios and thought experiments",
            "- Reversal thinking (what's the opposite?)",
            "- Combination of unrelated elements",
            "- Alternative perspectives and viewpoints",
            "",
            "Note: Quantity over quality - generate many ideas without judgment.",
        ]


class SynthesisThinkingCapability(ThinkingCapability):
    """Synthesis thinking capability: metacognition and integration."""

    def __init__(self) -> None:
        super().__init__(
            thinking_direction=ThinkingDirection.SYNTHESIS,
            role="Metacognitive Orchestrator",
            description="Thinking process management and comprehensive coordination",
            role_description="I integrate all perspectives and provide the final balanced answer. My output is what users see - it must be practical and human-friendly.",
            thinking_mode="metacognitive_synthetic",
            cognitive_focus="Metacognition and executive control",
            output_style="Comprehensive integration, process management, unified conclusions",
            timing_config=THINKING_TIMING_CONFIGS[ThinkingDirection.SYNTHESIS],
            reasoning_level=3,  # Highest level of metacognition
            memory_enabled=True,  # Need to remember all other thinking direction results
        )

    def _get_specific_instructions(self) -> list[str]:
        return [
            "",
            "STRATEGIC SYNTHESIS GUIDELINES:",
            "• Your primary goal: Answer the original question using insights from other analyses",
            "• Avoid generic rehashing - focus specifically on the question asked",
            "• Use other analyses' contributions as evidence/perspectives to build your answer",
            "• Provide practical, actionable insights users can understand",
            "",
            "CRITICAL QUESTION-FOCUSED APPROACH:",
            "1. Extract insights from other analyses that directly address the question",
            "2. Ignore generic statements - focus on question-relevant content",
            "3. Build a coherent answer that uses multiple perspectives as support",
            "4. End with a clear, direct response to what was originally asked",
            "",
            "KEY RESPONSIBILITIES:",
            "- Return to the original question and answer it directly",
            "- Use other analyses' insights as building blocks for your answer",
            "- Synthesize perspectives into a cohesive response to the specific question",
            "- Avoid academic summarization - focus on practical question-answering",
            "- Ensure your entire response serves the original question",
            "",
            "FINAL OUTPUT REQUIREMENTS:",
            "• This is the user's ONLY answer - it must directly address their question",
            "• Don't just summarize - synthesize into a clear answer",
            "• Remove content that doesn't directly relate to the original question",
            "• For philosophical questions: provide thoughtful answers, not just analysis",
            "• You may mention different analysis types, perspectives, or methodologies if relevant",
            "• Present your synthesis process transparently, showing how different viewpoints contribute",
        ]


class MultiThinkingAgentFactory:
    """Multi-thinking Agent factory."""

    # Thinking direction capability mapping
    THINKING_CAPABILITIES = {
        ThinkingDirection.FACTUAL: FactualThinkingCapability(),
        ThinkingDirection.EMOTIONAL: EmotionalThinkingCapability(),
        ThinkingDirection.CRITICAL: CriticalThinkingCapability(),
        ThinkingDirection.OPTIMISTIC: OptimisticThinkingCapability(),
        ThinkingDirection.CREATIVE: CreativeThinkingCapability(),
        ThinkingDirection.SYNTHESIS: SynthesisThinkingCapability(),
    }

    def __init__(self) -> None:
        self._agent_cache: dict[str, Agent] = {}  # Cache for created agents

    def create_thinking_agent(
        self,
        thinking_direction: ThinkingDirection,
        model: Model,
        context: str = "",
        previous_results: dict | None = None,
        **kwargs,
    ) -> Agent:
        """Create a specific thinking direction Agent."""
        capability = self.THINKING_CAPABILITIES[thinking_direction]

        # Generate cache key
        cache_key = f"{thinking_direction.value}_{model.__class__.__name__}_{hash(context)}"

        if cache_key in self._agent_cache:
            # Return cached agent but update instructions
            agent = self._agent_cache[cache_key]
            agent.instructions = capability.get_instructions(context, previous_results)
            return agent

        # Create new agent
        agent = Agent(
            name=f"{thinking_direction.value.title()}AnalysisAgent",
            role=capability.role,
            description=capability.description,
            model=model,
            tools=capability.tools if capability.tools else None,
            instructions=capability.get_instructions(context, previous_results),
            markdown=True,
            **kwargs,
        )

        # Add special configuration
        if capability.memory_enabled:
            agent.enable_user_memories = True

        # Cache agent
        self._agent_cache[cache_key] = agent

        logger.info(
            f"Created {thinking_direction.value} thinking agent with {capability.timing_config.default_time_seconds}s time limit"
        )
        return agent

    def get_thinking_timing(self, thinking_direction: ThinkingDirection) -> ThinkingTimingConfig:
        """Get specific thinking direction timing configuration."""
        return THINKING_TIMING_CONFIGS[thinking_direction]

    def get_all_thinking_directions(self) -> list[ThinkingDirection]:
        """Get all available thinking directions."""
        return list(self.THINKING_CAPABILITIES.keys())

    def clear_cache(self) -> None:
        """Clear agent cache."""
        self._agent_cache.clear()
        logger.info("Thinking agent cache cleared")


# Global factory instance
_thinking_factory = MultiThinkingAgentFactory()


# Convenience functions
def create_thinking_agent(thinking_direction: ThinkingDirection, model: Model, **kwargs) -> Agent:
    """Convenience function to create thinking Agent."""
    return _thinking_factory.create_thinking_agent(thinking_direction, model, **kwargs)


def get_thinking_timing(thinking_direction: ThinkingDirection) -> ThinkingTimingConfig:
    """Convenience function to get thinking timing configuration."""
    return _thinking_factory.get_thinking_timing(thinking_direction)


def get_all_thinking_directions() -> list[ThinkingDirection]:
    """Convenience function to get all thinking directions."""
    return _thinking_factory.get_all_thinking_directions()
