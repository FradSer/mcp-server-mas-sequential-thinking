"""Simplified agent factory using composition and capability patterns."""

from dataclasses import dataclass
from typing import Dict, List, Type
from agno.agent import Agent
from agno.models.base import Model
from agno.tools.thinking import ThinkingTools
from agno.tools.exa import ExaTools


@dataclass(frozen=True)
class AgentCapability:
    """Defines agent capabilities and configuration."""

    role: str
    description: str
    tools: List[Type]
    role_description: str

    def get_instructions(self) -> List[str]:
        """Generate instructions for this capability."""
        return [
            "You are a specialist agent receiving specific sub-tasks from the Team Coordinator.",
            f"Your role: {self.role_description}",
            "Process: 1) Understand the delegated sub-task, 2) Use tools as needed, 3) Provide focused results, 4) Return response to Coordinator.",
            "Focus on accuracy and relevance for your assigned task.",
        ]

    def create_tools(self) -> List:
        """Instantiate tools for this capability."""
        return [tool_class() for tool_class in self.tools]


class AgentFactory:
    """Factory for creating specialized agents with capability composition."""

    # Core capabilities registry
    CAPABILITIES = {
        "planner": AgentCapability(
            role="Strategic Planner",
            description="Develops strategic plans and roadmaps based on delegated sub-tasks",
            tools=[ThinkingTools],
            role_description="Develop strategic plans, roadmaps, and process designs for planning-related sub-tasks",
        ),
        "researcher": AgentCapability(
            role="Information Gatherer",
            description="Gathers and validates information based on delegated research sub-tasks",
            tools=[ThinkingTools, ExaTools],
            role_description="Find, gather, and validate information using research tools for information-related sub-tasks",
        ),
        "analyzer": AgentCapability(
            role="Core Analyst",
            description="Performs analysis based on delegated analytical sub-tasks",
            tools=[ThinkingTools],
            role_description="Analyze patterns, evaluate logic, and generate insights for analytical sub-tasks",
        ),
        "critic": AgentCapability(
            role="Quality Controller",
            description="Critically evaluates ideas or assumptions based on delegated critique sub-tasks",
            tools=[ThinkingTools],
            role_description="Evaluate assumptions, identify flaws, and provide constructive critique for evaluation sub-tasks",
        ),
        "synthesizer": AgentCapability(
            role="Integration Specialist",
            description="Integrates information or forms conclusions based on delegated synthesis sub-tasks",
            tools=[ThinkingTools],
            role_description="Integrate information, synthesize ideas, and form conclusions for synthesis sub-tasks",
        ),
    }

    @classmethod
    def create_agent(cls, agent_type: str, model: Model, **kwargs) -> Agent:
        """Create a specialized agent using capability composition."""
        if agent_type not in cls.CAPABILITIES:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available: {list(cls.CAPABILITIES.keys())}"
            )

        capability = cls.CAPABILITIES[agent_type]
        instructions = capability.get_instructions()

        # Add any additional instructions
        if "extra_instructions" in kwargs:
            instructions.extend(kwargs.pop("extra_instructions"))

        return Agent(
            name=agent_type.title(),
            role=capability.role,
            description=capability.description,
            tools=capability.create_tools(),
            instructions=instructions,
            model=model,
            add_datetime_to_instructions=True,
            markdown=True,
            **kwargs,
        )

    @classmethod
    def create_all_agents(cls, model: Model) -> Dict[str, Agent]:
        """Create all specialist agents using factory pattern."""
        return {
            agent_type: cls.create_agent(agent_type, model)
            for agent_type in cls.CAPABILITIES.keys()
        }


# Convenience functions for backward compatibility
def create_agent(agent_type: str, model: Model, **kwargs) -> Agent:
    """Create a specialized agent (backward compatible)."""
    return AgentFactory.create_agent(agent_type, model, **kwargs)


def create_all_agents(model: Model) -> Dict[str, Agent]:
    """Create all specialist agents (backward compatible)."""
    return AgentFactory.create_all_agents(model)
