"""Six Thinking Hats Agent Factory - Complete Rewrite

纯净的Six Thinking Hats实现，无任何Legacy代码。
基于Edward de Bono的六帽思维方法论。
"""

import logging
import os
from dataclasses import dataclass
from typing import Any

from agno.agent import Agent
from agno.models.base import Model

# Import Six Hats support
from .six_hats_core import (
    HatColor,
    SixHatsAgentFactory,
    create_hat_agent,
    get_all_hat_colors,
)

logger = logging.getLogger(__name__)

# Conditional import of ExaTools for research capabilities
_EXA_AVAILABLE = bool(os.environ.get("EXA_API_KEY"))
if _EXA_AVAILABLE:
    try:
        from agno.tools.exa import ExaTools
    except ImportError:
        _EXA_AVAILABLE = False
        ExaTools = None
else:
    ExaTools = None


@dataclass(frozen=True)
class SixHatsTeamConfig:
    """Six Hats team configuration."""

    name: str
    hat_sequence: list[HatColor]
    description: str


class SixHatsAgentManager:
    """Manages Six Thinking Hats agents exclusively."""

    def __init__(self):
        """Initialize Six Hats agent manager."""
        self._hat_factory = SixHatsAgentFactory()
        logger.info("Six Hats Agent Manager initialized")

    def create_hat_agent(
        self,
        hat_color: HatColor,
        model: Model,
        context: str = "",
        previous_results: dict = None,
        **kwargs
    ) -> Agent:
        """Create a single Six Thinking Hats agent."""
        return self._hat_factory.create_hat_agent(
            hat_color, model, context, previous_results, **kwargs
        )

    def create_hat_sequence_agents(
        self,
        hat_sequence: list[HatColor],
        model: Model,
        context: str = "",
        **kwargs
    ) -> list[Agent]:
        """Create a sequence of Six Hats agents."""
        agents = []
        previous_results = {}

        for hat_color in hat_sequence:
            agent = self.create_hat_agent(
                hat_color, model, context, previous_results, **kwargs
            )
            agents.append(agent)

        return agents

    def get_predefined_sequences(self) -> dict[str, list[HatColor]]:
        """Get predefined Six Hats sequences."""
        return {
            "single_white": [HatColor.WHITE],
            "single_red": [HatColor.RED],
            "single_black": [HatColor.BLACK],
            "single_yellow": [HatColor.YELLOW],
            "single_green": [HatColor.GREEN],
            "single_blue": [HatColor.BLUE],

            "double_analytical": [HatColor.WHITE, HatColor.BLACK],
            "double_emotional": [HatColor.RED, HatColor.YELLOW],
            "double_creative": [HatColor.GREEN, HatColor.BLUE],

            "triple_factual": [HatColor.WHITE, HatColor.GREEN, HatColor.BLUE],
            "triple_emotional": [HatColor.RED, HatColor.YELLOW, HatColor.BLUE],
            "triple_critical": [HatColor.WHITE, HatColor.BLACK, HatColor.BLUE],

            "philosophical": [HatColor.WHITE, HatColor.GREEN, HatColor.BLUE],
            "decision_making": [HatColor.WHITE, HatColor.BLACK, HatColor.YELLOW, HatColor.BLUE],
            "creative_problem": [HatColor.RED, HatColor.GREEN, HatColor.YELLOW, HatColor.BLUE],

            "full_sequence": [
                HatColor.BLUE,   # Process control
                HatColor.WHITE,  # Facts and data
                HatColor.RED,    # Emotions and intuition
                HatColor.YELLOW, # Benefits and optimism
                HatColor.BLACK,  # Caution and criticism
                HatColor.GREEN,  # Creativity and alternatives
                HatColor.BLUE    # Summary and conclusion
            ]
        }

    def get_available_hat_colors(self) -> list[HatColor]:
        """Get all available hat colors."""
        return get_all_hat_colors()

    def clear_cache(self):
        """Clear agent cache."""
        self._hat_factory.clear_cache()
        logger.info("Six Hats agent cache cleared")


class SixHatsTeamFactory:
    """Factory for creating Six Thinking Hats teams."""

    def __init__(self):
        """Initialize Six Hats team factory."""
        self._agent_manager = SixHatsAgentManager()

    def create_team_by_sequence(
        self,
        sequence_name: str,
        model: Model,
        **kwargs
    ) -> dict[str, Agent]:
        """Create a team based on predefined sequence."""
        sequences = self._agent_manager.get_predefined_sequences()

        if sequence_name not in sequences:
            available = ", ".join(sequences.keys())
            raise ValueError(f"Unknown sequence '{sequence_name}'. Available: {available}")

        hat_sequence = sequences[sequence_name]
        agents = {}

        for i, hat_color in enumerate(hat_sequence):
            agent_key = f"{hat_color.value}_{i}" if hat_sequence.count(hat_color) > 1 else hat_color.value
            agents[agent_key] = self._agent_manager.create_hat_agent(
                hat_color, model, **kwargs
            )

        logger.info(f"Created Six Hats team '{sequence_name}' with {len(agents)} agents")
        return agents

    def create_custom_team(
        self,
        hat_sequence: list[HatColor],
        model: Model,
        team_name: str = "custom",
        **kwargs
    ) -> dict[str, Agent]:
        """Create a custom Six Hats team."""
        agents = {}

        for i, hat_color in enumerate(hat_sequence):
            agent_key = f"{hat_color.value}_{i}" if hat_sequence.count(hat_color) > 1 else hat_color.value
            agents[agent_key] = self._agent_manager.create_hat_agent(
                hat_color, model, **kwargs
            )

        logger.info(f"Created custom Six Hats team '{team_name}' with {len(agents)} agents")
        return agents

    def get_available_sequences(self) -> list[str]:
        """Get available predefined sequences."""
        return list(self._agent_manager.get_predefined_sequences().keys())


# Global factory instance
_six_hats_factory = SixHatsTeamFactory()


# Public API functions
def create_hat_agent(hat_color: HatColor, model: Model, **kwargs) -> Agent:
    """Create a single Six Thinking Hats agent."""
    return _six_hats_factory._agent_manager.create_hat_agent(hat_color, model, **kwargs)


def create_six_hats_team(sequence_name: str, model: Model, **kwargs) -> dict[str, Agent]:
    """Create a Six Thinking Hats team by sequence name."""
    return _six_hats_factory.create_team_by_sequence(sequence_name, model, **kwargs)


def create_custom_six_hats_team(
    hat_sequence: list[HatColor],
    model: Model,
    team_name: str = "custom",
    **kwargs
) -> dict[str, Agent]:
    """Create a custom Six Thinking Hats team."""
    return _six_hats_factory.create_custom_team(hat_sequence, model, team_name, **kwargs)


def get_available_sequences() -> list[str]:
    """Get available Six Hats sequences."""
    return _six_hats_factory.get_available_sequences()


def get_available_hats() -> list[HatColor]:
    """Get available hat colors."""
    return _six_hats_factory._agent_manager.get_available_hat_colors()


# Backward compatibility - minimal interface
class UnifiedAgentFactory:
    """Minimal factory for backward compatibility - Six Hats only."""

    def __init__(self):
        self._six_hats_factory = _six_hats_factory
        logger.info("UnifiedAgentFactory initialized with Six Hats only")

    def create_agent(self, agent_type: str, model: Model, **kwargs) -> Agent:
        """Create agent - Six Hats only."""
        if agent_type.endswith("_hat"):
            hat_color_name = agent_type.replace("_hat", "")
            try:
                hat_color = HatColor(hat_color_name)
                return create_hat_agent(hat_color, model, **kwargs)
            except ValueError:
                pass

        raise ValueError(f"Unknown agent type: {agent_type}. Use Six Hats: {[f'{h.value}_hat' for h in get_available_hats()]}")

    def create_team_agents(self, model: Model, team_type: str = "six_hats_triple") -> dict[str, Agent]:
        """Create team agents - Six Hats only."""
        if team_type.startswith("six_hats_"):
            sequence_name = team_type.replace("six_hats_", "")
            if sequence_name in ["triple", "full", "philosophical"]:
                return create_six_hats_team(sequence_name, model)

        # Default to triple sequence
        return create_six_hats_team("triple_factual", model)

    def get_available_team_types(self) -> list[str]:
        """Get available team types - Six Hats sequences only."""
        sequences = get_available_sequences()
        return [f"six_hats_{seq}" for seq in sequences]


# Global singleton for backward compatibility
_factory = UnifiedAgentFactory()


# Backward compatibility functions
def create_agent(agent_type: str, model: Model, **kwargs) -> Agent:
    """Backward compatible agent creation."""
    return _factory.create_agent(agent_type, model, **kwargs)


def create_team_agents(model: Model, team_type: str = "six_hats_triple") -> dict[str, Agent]:
    """Backward compatible team creation."""
    return _factory.create_team_agents(model, team_type)