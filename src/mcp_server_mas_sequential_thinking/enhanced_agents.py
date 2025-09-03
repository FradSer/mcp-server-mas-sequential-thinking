"""Enhanced agent factory with new Agno 1.8+ features."""

from dataclasses import dataclass
from typing import Dict, List, Type, Optional
from agno.agent import Agent
from agno.models.base import Model
from agno.tools.thinking import ThinkingTools
from agno.tools.exa import ExaTools


@dataclass(frozen=True)
class EnhancedAgentCapability:
    """Enhanced agent capabilities with new Agno features."""

    role: str
    description: str
    tools: List[Type]
    role_description: str
    reasoning_level: int = 1  # New: Reasoning complexity level
    memory_enabled: bool = False  # New: Enable agent memory
    structured_outputs: bool = True  # New: Use structured outputs

    def get_instructions(self) -> List[str]:
        """Generate enhanced instructions with reasoning patterns."""
        base_instructions = [
            "You are an enhanced specialist agent with advanced reasoning capabilities.",
            f"Your role: {self.role_description}",
            "Process: 1) Analyze sub-task with reasoning, 2) Use available tools, 3) Structure response, 4) Return to Coordinator.",
        ]
        
        # Add reasoning-specific instructions
        if self.reasoning_level >= 2:
            base_instructions.append(
                "Apply step-by-step reasoning and validate your logic at each step."
            )
        if self.reasoning_level >= 3:
            base_instructions.append(
                "Use chain-of-thought reasoning and consider alternative approaches."
            )
            
        return base_instructions

    def create_tools(self) -> List:
        """Instantiate enhanced tools."""
        return [tool_class() for tool_class in self.tools]


class EnhancedAgentFactory:
    """Enhanced factory with new Agno 1.8+ capabilities."""

    # Enhanced capabilities with new features
    ENHANCED_CAPABILITIES = {
        "reasoning_planner": EnhancedAgentCapability(
            role="Strategic Reasoning Planner",
            description="Advanced strategic planning with multi-step reasoning",
            tools=[ThinkingTools],
            role_description="Develop complex strategic plans using advanced reasoning patterns",
            reasoning_level=3,
            memory_enabled=True,
            structured_outputs=True,
        ),
        "research_analyst": EnhancedAgentCapability(
            role="Research & Analysis Specialist",
            description="Combined research and analysis with memory",
            tools=[ThinkingTools, ExaTools],
            role_description="Conduct research and perform analysis with context memory",
            reasoning_level=2,
            memory_enabled=True,
            structured_outputs=True,
        ),
        "critical_thinker": EnhancedAgentCapability(
            role="Critical Reasoning Specialist",
            description="Advanced critical thinking with structured outputs",
            tools=[ThinkingTools],
            role_description="Apply critical thinking with logical reasoning chains",
            reasoning_level=3,
            memory_enabled=False,
            structured_outputs=True,
        ),
        "creative_synthesizer": EnhancedAgentCapability(
            role="Creative Synthesis Specialist",
            description="Creative synthesis with multi-modal reasoning",
            tools=[ThinkingTools],
            role_description="Synthesize ideas creatively using advanced reasoning",
            reasoning_level=2,
            memory_enabled=True,
            structured_outputs=True,
        ),
    }

    @classmethod
    def create_enhanced_agent(
        cls, 
        agent_type: str, 
        model: Model, 
        enable_memory: Optional[bool] = None,
        reasoning_level: Optional[int] = None,
        **kwargs
    ) -> Agent:
        """Create enhanced agent with new Agno features."""
        if agent_type not in cls.ENHANCED_CAPABILITIES:
            raise ValueError(
                f"Unknown enhanced agent type: {agent_type}. "
                f"Available: {list(cls.ENHANCED_CAPABILITIES.keys())}"
            )

        capability = cls.ENHANCED_CAPABILITIES[agent_type]
        instructions = capability.get_instructions()

        # Override capability settings if provided
        memory_enabled = enable_memory if enable_memory is not None else capability.memory_enabled
        
        # Add any additional instructions
        if "extra_instructions" in kwargs:
            instructions.extend(kwargs.pop("extra_instructions"))

        agent_kwargs = {
            "name": f"Enhanced{agent_type.title().replace('_', '')}",
            "role": capability.role,
            "description": capability.description,
            "tools": capability.create_tools(),
            "instructions": instructions,
            "model": model,
            "add_datetime_to_instructions": True,
            "markdown": True,
            **kwargs,
        }

        # Add new Agno features if supported
        if memory_enabled:
            # Enable memory features (requires proper memory storage setup)
            agent_kwargs["enable_memory"] = True
            
        if capability.structured_outputs:
            # Enable structured outputs
            agent_kwargs["structured_outputs"] = True

        return Agent(**agent_kwargs)

    @classmethod
    def create_all_enhanced_agents(cls, model: Model) -> Dict[str, Agent]:
        """Create all enhanced specialist agents."""
        return {
            agent_type: cls.create_enhanced_agent(agent_type, model)
            for agent_type in cls.ENHANCED_CAPABILITIES.keys()
        }

    @classmethod
    def create_hybrid_team(cls, model: Model) -> Dict[str, Agent]:
        """Create a hybrid team mixing standard and enhanced agents."""
        from .agents import AgentFactory
        
        # Standard agents
        standard_agents = {
            "planner": AgentFactory.create_agent("planner", model),
            "researcher": AgentFactory.create_agent("researcher", model),
        }
        
        # Enhanced agents
        enhanced_agents = {
            "reasoning_planner": cls.create_enhanced_agent("reasoning_planner", model),
            "research_analyst": cls.create_enhanced_agent("research_analyst", model),
        }
        
        return {**standard_agents, **enhanced_agents}


# Convenience functions
def create_enhanced_agent(agent_type: str, model: Model, **kwargs) -> Agent:
    """Create an enhanced agent with new features."""
    return EnhancedAgentFactory.create_enhanced_agent(agent_type, model, **kwargs)


def create_all_enhanced_agents(model: Model) -> Dict[str, Agent]:
    """Create all enhanced specialist agents."""
    return EnhancedAgentFactory.create_all_enhanced_agents(model)


def create_hybrid_team(model: Model) -> Dict[str, Agent]:
    """Create hybrid team with standard and enhanced agents."""
    return EnhancedAgentFactory.create_hybrid_team(model)