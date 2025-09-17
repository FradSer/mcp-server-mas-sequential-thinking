"""Unified team factory with simplified creation logic and eliminated conditional complexity."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from agno.team.team import Team
from agno.models.base import Model

from .modernized_config import ModelConfig, get_model_config
from .unified_agents import UnifiedAgentFactory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeamConfiguration:
    """Team configuration with all necessary settings."""

    name: str
    description: str
    instructions: List[str]
    success_criteria: List[str]
    team_type: str
    enable_advanced_features: bool = False


class TeamBuilder(ABC):
    """Abstract builder for creating teams with different configurations."""

    @abstractmethod
    def get_configuration(self) -> TeamConfiguration:
        """Return team configuration."""
        pass

    def build_team(
        self, config: ModelConfig, agent_factory: UnifiedAgentFactory
    ) -> Team:
        """Build team with specified configuration."""
        team_config = self.get_configuration()

        # Create model instances
        team_model = config.provider_class(id=config.team_model_id)
        agent_model = config.provider_class(id=config.agent_model_id)

        # Create agents using factory
        agents = agent_factory.create_team_agents(agent_model, team_config.team_type)

        # Create team with unified configuration (v2 compatible)
        team = Team(
            name=team_config.name,
            members=list(agents.values()),
            model=team_model,
            description=team_config.description,
            instructions=team_config.instructions,
            # v2 attributes replacing deprecated mode="coordinate"
            respond_directly=False,  # Team leader processes responses from members
            delegate_task_to_all_members=False,  # Delegate one by one
            determine_input_for_members=True,  # Team leader synthesizes input
            enable_agentic_state=team_config.enable_advanced_features,
            share_member_interactions=team_config.enable_advanced_features,
            markdown=True,
        )

        logger.info(
            f"Team '{team_config.name}' created with {config.provider_class.__name__} provider"
        )
        return team


class StandardTeamBuilder(TeamBuilder):
    """Builder for standard sequential thinking team."""

    def get_configuration(self) -> TeamConfiguration:
        return TeamConfiguration(
            name="SequentialThinkingTeam",
            description="Coordinator for sequential thinking specialist team",
            team_type="standard",
            instructions=[
                "You coordinate specialists (Planner, Researcher, Analyzer, Critic, Synthesizer) for sequential thinking.",
                "HOTFIX: Optimize for speed - delegate to maximum 2 specialists per thought unless complexity demands more.",
                "Process: 1) Quick analysis (< 5s), 2) Select MINIMAL specialists needed, 3) Delegate ONE clear sub-task each, 4) Fast synthesis, 5) Provide guidance.",
                "Include recommendations: 'RECOMMENDATION: Revise thought #X...' or 'SUGGESTION: Consider branching from thought #Y...'",
                "Skip specialists if their contribution won't significantly improve the response quality.",
            ],
            success_criteria=[
                "Efficiently delegate sub-tasks to relevant specialists",
                "Synthesize specialist responses coherently",
                "Recommend revisions or branches based on analysis",
            ],
            enable_advanced_features=False,
        )


class EnhancedTeamBuilder(TeamBuilder):
    """Builder for enhanced team with Agno 1.8+ features."""

    def get_configuration(self) -> TeamConfiguration:
        return TeamConfiguration(
            name="EnhancedSequentialThinkingTeam",
            description="Enhanced coordinator with advanced reasoning capabilities",
            team_type="enhanced",
            instructions=[
                "You coordinate enhanced specialists with advanced reasoning capabilities for sequential thinking.",
                "Available specialists: Enhanced agents with memory, reasoning, and structured outputs.",
                "Process: 1) Analyze input with reasoning, 2) Select optimal specialists, 3) Delegate with context, 4) Synthesize structured responses, 5) Provide actionable guidance.",
                "Use structured outputs for recommendations: 'RECOMMENDATION: ...' or 'SUGGESTION: ...'",
                "Leverage agent memory and reasoning capabilities for complex tasks.",
                "Prioritize accuracy and depth over speed when dealing with complex reasoning tasks.",
            ],
            success_criteria=[
                "Efficiently delegate sub-tasks to relevant specialists",
                "Synthesize specialist responses coherently",
                "Recommend revisions or branches based on analysis",
                "Utilize agent memory for context retention",
                "Apply structured reasoning patterns",
                "Generate structured outputs",
            ],
            enable_advanced_features=True,
        )


class HybridTeamBuilder(TeamBuilder):
    """Builder for hybrid team mixing standard and enhanced agents."""

    def get_configuration(self) -> TeamConfiguration:
        return TeamConfiguration(
            name="HybridSequentialThinkingTeam",
            description="Hybrid coordinator with mixed standard and enhanced capabilities",
            team_type="hybrid",
            instructions=[
                "You coordinate a hybrid team mixing standard and enhanced specialists for sequential thinking.",
                "Standard agents (researcher, analyzer): Fast, efficient processing for routine tasks.",
                "Enhanced agents (planner, critic, synthesizer): Advanced reasoning for complex tasks.",
                "Process: 1) Assess task complexity, 2) Select appropriate agent types, 3) Delegate strategically, 4) Synthesize diverse responses, 5) Provide balanced guidance.",
                "Balance speed and accuracy based on task requirements.",
                "Leverage each agent type's strengths for optimal results.",
            ],
            success_criteria=[
                "Efficiently utilize both standard and enhanced specialists",
                "Balance speed and accuracy based on task complexity",
                "Synthesize responses from diverse agent capabilities",
                "Adapt delegation strategy to agent strengths",
            ],
            enable_advanced_features=True,
        )


class EnhancedSpecializedTeamBuilder(TeamBuilder):
    """Builder for team with enhanced specialized agents only."""

    def get_configuration(self) -> TeamConfiguration:
        return TeamConfiguration(
            name="EnhancedSpecializedThinkingTeam",
            description="Coordinator for specialized enhanced agents with advanced capabilities",
            team_type="enhanced_specialized",
            instructions=[
                "You coordinate highly specialized enhanced agents for complex sequential thinking.",
                "Available specialists: Reasoning Planner, Research Analyst, Critical Thinker, Creative Synthesizer.",
                "Process: 1) Deep analysis with multi-step reasoning, 2) Strategic specialist selection, 3) Context-rich delegation, 4) Advanced synthesis, 5) Structured guidance.",
                "Focus on maximum accuracy and depth for complex problems requiring advanced reasoning.",
                "Utilize full reasoning capabilities and structured outputs for optimal results.",
            ],
            success_criteria=[
                "Maximize reasoning depth and accuracy",
                "Utilize advanced agent capabilities effectively",
                "Generate highly structured and comprehensive outputs",
                "Handle complex problems requiring specialized expertise",
            ],
            enable_advanced_features=True,
        )


class UnifiedTeamFactory:
    """Unified factory for creating all team types with eliminated conditional complexity."""

    def __init__(self):
        self._builders = {
            "standard": StandardTeamBuilder(),
            "enhanced": EnhancedTeamBuilder(),
            "hybrid": HybridTeamBuilder(),
            "enhanced_specialized": EnhancedSpecializedTeamBuilder(),
        }
        self._agent_factory = UnifiedAgentFactory()

    def create_team(
        self, team_type: str = "standard", config: Optional[ModelConfig] = None
    ) -> Team:
        """Create team using unified factory with simplified logic."""
        if team_type not in self._builders:
            available_types = ", ".join(sorted(self._builders.keys()))
            raise ValueError(
                f"Unknown team type '{team_type}'. Available types: {available_types}"
            )

        if config is None:
            config = get_model_config()

        builder = self._builders[team_type]
        return builder.build_team(config, self._agent_factory)

    def get_available_team_types(self) -> List[str]:
        """Get list of available team types."""
        return list(self._builders.keys())


# Singleton instance
_team_factory = UnifiedTeamFactory()


# Convenience functions for backward compatibility and external use
def create_team(config: Optional[ModelConfig] = None) -> Team:
    """Create standard team (backward compatible)."""
    return _team_factory.create_team("standard", config)


def create_enhanced_team(config: Optional[ModelConfig] = None) -> Team:
    """Create enhanced team (backward compatible)."""
    return _team_factory.create_team("enhanced", config)


def create_hybrid_team_instance(config: Optional[ModelConfig] = None) -> Team:
    """Create hybrid team (backward compatible)."""
    return _team_factory.create_team("hybrid", config)


def create_enhanced_specialized_team(config: Optional[ModelConfig] = None) -> Team:
    """Create enhanced specialized team."""
    return _team_factory.create_team("enhanced_specialized", config)


def create_team_by_type(team_type: str, config: Optional[ModelConfig] = None) -> Team:
    """Create team by type with unified interface."""
    return _team_factory.create_team(team_type, config)
