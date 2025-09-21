"""Unified agent factory eliminating redundancy between standard and enhanced agents."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Type, Optional, Any
from agno.agent import Agent
from agno.models.base import Model
from agno.tools.reasoning import ReasoningTools

# Conditional import of ExaTools based on API key availability
_EXA_AVAILABLE = bool(os.environ.get("EXA_API_KEY"))
if _EXA_AVAILABLE:
    try:
        from agno.tools.exa import ExaTools
    except ImportError:
        _EXA_AVAILABLE = False
        ExaTools = None
else:
    ExaTools = None


def _get_research_tools() -> List[Type]:
    """Get available research tools based on API key availability."""
    tools = [ReasoningTools]
    if _EXA_AVAILABLE and ExaTools:
        tools.append(ExaTools)
    return tools


__all__ = [
    "AgentCapability",
    "UnifiedAgentFactory",
    "create_agent",
    "create_all_agents",
    "create_all_enhanced_agents",
    "create_hybrid_team",
    "create_enhanced_specialized_agents",
    "ReasoningLevel",
]


# Enhanced reasoning level constants with better organization
class ReasoningLevel:
    """Agent reasoning capability levels with descriptive values and validation."""

    # Core reasoning levels
    BASIC = 1
    INTERMEDIATE = 2
    ADVANCED = 3

    # Validation and utility methods
    ALL_LEVELS = (BASIC, INTERMEDIATE, ADVANCED)
    LEVEL_NAMES = {BASIC: "Basic", INTERMEDIATE: "Intermediate", ADVANCED: "Advanced"}

    @classmethod
    def validate(cls, level: int) -> bool:
        """Validate if a reasoning level is valid."""
        return level in cls.ALL_LEVELS

    @classmethod
    def get_name(cls, level: int) -> str:
        """Get human-readable name for reasoning level."""
        return cls.LEVEL_NAMES.get(level, f"Unknown({level})")


@dataclass(frozen=True)
class AgentCapability:
    """Unified agent capability definition with configurable enhancement levels."""

    role: str
    description: str
    tools: List[Type]
    role_description: str

    # Enhancement features
    reasoning_level: int = 1
    memory_enabled: bool = False
    structured_outputs: bool = True

    def get_instructions(self, enhanced_mode: bool = False) -> List[str]:
        """Generate instructions based on enhancement mode."""
        base_instructions = [
            f"You are a {'enhanced ' if enhanced_mode else ''}specialist agent receiving specific sub-tasks from the Team Coordinator.",
            f"Your role: {self.role_description}",
            "Process: 1) Understand the delegated sub-task, 2) Use tools as needed, 3) Provide focused results, 4) Return response to Coordinator.",
        ]

        if enhanced_mode:
            base_instructions.extend(
                [
                    "Focus on accuracy and relevance using advanced reasoning capabilities.",
                    "Apply structured thinking patterns when processing complex sub-tasks.",
                ]
            )

            if self.reasoning_level >= ReasoningLevel.INTERMEDIATE:
                base_instructions.append(
                    "Apply step-by-step reasoning and validate your logic at each step."
                )
            if self.reasoning_level >= ReasoningLevel.ADVANCED:
                base_instructions.append(
                    "Use chain-of-thought reasoning and consider alternative approaches."
                )
        else:
            base_instructions.append(
                "Focus on accuracy and relevance for your assigned task."
            )

        return base_instructions

    def create_tools(self) -> List:
        """Instantiate tools for this capability."""
        return [tool_class() for tool_class in self.tools]


class AgentBuilder(ABC):
    """Abstract builder for creating agents with different enhancement levels."""

    def __init__(self, capability: AgentCapability):
        self.capability = capability

    @abstractmethod
    def build_agent(self, model: Model, enhanced_mode: bool = False, **kwargs) -> Agent:
        """Build agent with specified enhancement level."""
        pass


class StandardAgentBuilder(AgentBuilder):
    """Builder for standard agents with optional enhancement features."""

    def build_agent(self, model: Model, enhanced_mode: bool = False, **kwargs) -> Agent:
        """Build agent with configurable enhancement level."""
        instructions = self.capability.get_instructions(enhanced_mode)

        # Add any additional instructions
        if "extra_instructions" in kwargs:
            instructions.extend(kwargs.pop("extra_instructions"))

        # Generate agent name with optimized string operations
        clean_role = self.capability.role.replace(" ", "").replace("&", "And")
        agent_name = f"Enhanced{clean_role}" if enhanced_mode else clean_role

        agent_kwargs = {
            "name": agent_name,
            "role": self.capability.role,
            "description": self.capability.description,
            "tools": self.capability.create_tools(),
            "instructions": instructions,
            "model": model,
            "markdown": True,
            **kwargs,
        }

        # Add enhancement features if enabled
        if enhanced_mode:
            if self.capability.memory_enabled:
                agent_kwargs["enable_user_memories"] = True
            # HOTFIX: Disable structured outputs for Deepseek due to instability
            if self.capability.structured_outputs and not self._is_deepseek_model(
                model
            ):
                agent_kwargs["structured_outputs"] = True

        return Agent(**agent_kwargs)

    def _is_deepseek_model(self, model: Model) -> bool:
        """Check if model is a Deepseek model (hotfix for structured outputs issue)."""
        model_class_name = model.__class__.__name__.lower()
        model_id = getattr(model, "id", "").lower()
        return "deepseek" in model_class_name or "deepseek" in model_id


class UnifiedAgentFactory:
    """Unified factory eliminating redundancy between standard and enhanced agents."""

    # Core capability definitions organized by complexity and purpose
    CAPABILITIES = {
        "planner": AgentCapability(
            role="Strategic Planner",
            description="Develops strategic plans and roadmaps based on delegated sub-tasks",
            tools=[ReasoningTools],
            role_description="Develop strategic plans, roadmaps, and process designs for planning-related sub-tasks",
            reasoning_level=ReasoningLevel.INTERMEDIATE,
            memory_enabled=True,
            structured_outputs=True,
        ),
        "researcher": AgentCapability(
            role="Information Gatherer",
            description="Gathers and validates information based on delegated research sub-tasks",
            tools=_get_research_tools(),  # Conditional research tools based on API availability
            role_description="Find, gather, and validate information using research tools for information-related sub-tasks",
            reasoning_level=ReasoningLevel.BASIC,  # Simpler reasoning for data gathering
            memory_enabled=False,
            structured_outputs=True,
        ),
        "analyzer": AgentCapability(
            role="Core Analyst",
            description="Performs analysis based on delegated analytical sub-tasks",
            tools=[ReasoningTools],
            role_description="Analyze patterns, evaluate logic, and generate insights for analytical sub-tasks",
            reasoning_level=ReasoningLevel.INTERMEDIATE,
            memory_enabled=False,
            structured_outputs=True,
        ),
        "critic": AgentCapability(
            role="Quality Controller",
            description="Critically evaluates ideas or assumptions based on delegated critique sub-tasks",
            tools=[ReasoningTools],
            role_description="Evaluate assumptions, identify flaws, and provide constructive critique for evaluation sub-tasks",
            reasoning_level=ReasoningLevel.ADVANCED,  # Advanced reasoning for critical evaluation
            memory_enabled=False,
            structured_outputs=True,
        ),
        "synthesizer": AgentCapability(
            role="Integration Specialist",
            description="Integrates information or forms conclusions based on delegated synthesis sub-tasks",
            tools=[ReasoningTools],
            role_description="Integrate information, synthesize ideas, and form conclusions for synthesis sub-tasks",
            reasoning_level=ReasoningLevel.INTERMEDIATE,
            memory_enabled=True,  # Memory for context integration
            structured_outputs=True,
        ),
    }

    # Enhanced-only specialized roles
    ENHANCED_ONLY_CAPABILITIES = {
        "reasoning_planner": AgentCapability(
            role="Strategic Reasoning Planner",
            description="Advanced strategic planning with multi-step reasoning",
            tools=[ReasoningTools],
            role_description="Develop complex strategic plans using advanced reasoning patterns",
            reasoning_level=ReasoningLevel.ADVANCED,
            memory_enabled=True,
            structured_outputs=True,
        ),
        "research_analyst": AgentCapability(
            role="Research & Analysis Specialist",
            description="Combined research and analysis with memory",
            tools=_get_research_tools(),  # Conditional research tools based on API availability
            role_description="Conduct research and perform analysis with context memory",
            reasoning_level=ReasoningLevel.INTERMEDIATE,
            memory_enabled=True,
            structured_outputs=True,
        ),
        "critical_thinker": AgentCapability(
            role="Critical Reasoning Specialist",
            description="Advanced critical thinking with structured outputs",
            tools=[ReasoningTools],
            role_description="Apply critical thinking with logical reasoning chains",
            reasoning_level=ReasoningLevel.ADVANCED,
            memory_enabled=False,
            structured_outputs=True,
        ),
        "creative_synthesizer": AgentCapability(
            role="Creative Synthesis Specialist",
            description="Creative synthesis with multi-modal reasoning",
            tools=[ReasoningTools],
            role_description="Synthesize ideas creatively using advanced reasoning",
            reasoning_level=ReasoningLevel.INTERMEDIATE,
            memory_enabled=True,
            structured_outputs=True,
        ),
    }

    def __init__(self):
        self._builders = {
            capability_name: StandardAgentBuilder(capability)
            for capability_name, capability in {
                **self.CAPABILITIES,
                **self.ENHANCED_ONLY_CAPABILITIES,
            }.items()
        }

    def create_agent(
        self, agent_type: str, model: Model, enhanced_mode: bool = False, **kwargs
    ) -> Agent:
        """Create agent with unified factory eliminating duplication."""
        available_types = list(self.CAPABILITIES.keys())

        # Enhanced-only types require enhanced_mode=True
        if agent_type in self.ENHANCED_ONLY_CAPABILITIES and not enhanced_mode:
            raise ValueError(
                f"Agent type '{agent_type}' is only available in enhanced mode"
            )

        if enhanced_mode:
            available_types.extend(list(self.ENHANCED_ONLY_CAPABILITIES.keys()))

        if agent_type not in available_types:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {available_types}"
            )

        builder = self._builders[agent_type]
        return builder.build_agent(model, enhanced_mode, **kwargs)

    def create_team_agents(
        self, model: Model, team_type: str = "standard"
    ) -> Dict[str, Agent]:
        """Create complete agent teams based on team type."""
        if team_type == "standard":
            return {
                agent_type: self.create_agent(agent_type, model, enhanced_mode=False)
                for agent_type in self.CAPABILITIES.keys()
            }
        elif team_type == "enhanced":
            return {
                agent_type: self.create_agent(agent_type, model, enhanced_mode=True)
                for agent_type in self.CAPABILITIES.keys()
            }
        elif team_type == "enhanced_specialized":
            return {
                agent_type: self.create_agent(agent_type, model, enhanced_mode=True)
                for agent_type in self.ENHANCED_ONLY_CAPABILITIES.keys()
            }
        elif team_type == "hybrid":
            # Mix standard and enhanced agents strategically
            agents = {}

            # Use standard agents for simpler tasks
            for agent_type in ["researcher", "analyzer"]:
                agents[agent_type] = self.create_agent(
                    agent_type, model, enhanced_mode=False
                )

            # Use enhanced agents for complex reasoning tasks
            for agent_type in ["planner", "critic", "synthesizer"]:
                agents[agent_type] = self.create_agent(
                    agent_type, model, enhanced_mode=True
                )

            return agents
        else:
            raise ValueError(
                f"Unknown team type: {team_type}. Available: standard, enhanced, enhanced_specialized, hybrid"
            )


# Singleton instance
_factory = UnifiedAgentFactory()


# Convenience functions for backward compatibility
def create_agent(
    agent_type: str, model: Model, enhanced_mode: bool = False, **kwargs
) -> Agent:
    """Create agent using unified factory (backward compatible)."""
    return _factory.create_agent(agent_type, model, enhanced_mode, **kwargs)


def create_all_agents(model: Model) -> Dict[str, Agent]:
    """Create standard agents (backward compatible)."""
    return _factory.create_team_agents(model, "standard")


def create_all_enhanced_agents(model: Model) -> Dict[str, Agent]:
    """Create enhanced agents (backward compatible)."""
    return _factory.create_team_agents(model, "enhanced")


def create_hybrid_team(model: Model) -> Dict[str, Agent]:
    """Create hybrid team (backward compatible)."""
    return _factory.create_team_agents(model, "hybrid")


def create_enhanced_specialized_agents(model: Model) -> Dict[str, Agent]:
    """Create enhanced specialized agents."""
    return _factory.create_team_agents(model, "enhanced_specialized")
