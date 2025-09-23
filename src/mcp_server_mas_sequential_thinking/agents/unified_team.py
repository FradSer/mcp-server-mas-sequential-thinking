"""Multi-Thinking Team Factory - Complete Rewrite.

纯净的多向思维团队实现，无任何Legacy代码。
基于多向思维方法论，支持智能路由和动态序列。
"""

from dataclasses import dataclass
from typing import Any

from agno.team.team import Team

from mcp_server_mas_sequential_thinking.config.modernized_config import (
    ModelConfig,
    get_model_config,
)
from mcp_server_mas_sequential_thinking.infrastructure.logging_config import get_logger

# Import Six Hats support
from mcp_server_mas_sequential_thinking.processors.six_hats_core import (
    HatColor,
    SixHatsAgentFactory,
)
from mcp_server_mas_sequential_thinking.routing.six_hats_router import (
    SixHatsIntelligentRouter,
)

logger = get_logger(__name__)


@dataclass(frozen=True)
class SixHatsTeamConfiguration:
    """Six Thinking Hats team configuration."""

    name: str
    description: str
    hat_sequence: list[HatColor]
    instructions: list[str]
    routing_strategy: str = "intelligent"  # intelligent, sequential, adaptive


class SixHatsTeamBuilder:
    """Builder for Six Thinking Hats teams."""

    def __init__(self, config: SixHatsTeamConfiguration) -> None:
        self.config = config
        self._hat_factory = SixHatsAgentFactory()
        self._router = SixHatsIntelligentRouter()

    def build_team(self, model_config: ModelConfig) -> Team:
        """Build Six Thinking Hats team."""
        logger.info(f"Building Six Hats team: {self.config.name}")

        # Create models
        team_model = model_config.provider_class(id=model_config.enhanced_model_id)
        agent_model = model_config.provider_class(id=model_config.standard_model_id)

        # Create hat agents for the sequence
        hat_agents = []
        for hat_color in self.config.hat_sequence:
            agent = self._hat_factory.create_hat_agent(hat_color, agent_model)
            hat_agents.append(agent)

        # Create coordinating team with Blue Hat leadership
        team = Team(
            name=self.config.name,
            members=hat_agents,
            model=team_model,
            description=self.config.description,
            instructions=self.config.instructions,
            # Six Hats specific coordination
            respond_directly=False,  # Blue Hat coordinates final response
            delegate_task_to_all_members=False,  # Sequential hat thinking
            determine_input_for_members=True,  # Pass context between hats
            enable_agentic_state=True,
            share_member_interactions=True,
            markdown=True,
        )

        logger.info(
            f"Six Hats team '{self.config.name}' created with sequence: {[h.value for h in self.config.hat_sequence]}"
        )
        return team


class SixHatsTeamFactory:
    """Factory for creating Six Thinking Hats teams exclusively."""

    def __init__(self) -> None:
        """Initialize Six Hats team factory."""
        self._predefined_configs = {
            # Single Hat Teams
            "white_hat": SixHatsTeamConfiguration(
                name="WhiteHatFactualTeam",
                description="Pure factual and data-driven analysis",
                hat_sequence=[HatColor.WHITE],
                instructions=[
                    "Focus exclusively on facts, data, and objective information.",
                    "Present neutral, verified information without interpretation.",
                    "Identify missing information that would be helpful.",
                ],
            ),
            "red_hat": SixHatsTeamConfiguration(
                name="RedHatEmotionalTeam",
                description="Intuitive and emotional response analysis",
                hat_sequence=[HatColor.RED],
                instructions=[
                    "Express immediate emotional reactions and gut feelings.",
                    "Share intuitive responses without justification.",
                    "Provide the human emotional perspective on the situation.",
                ],
            ),
            # Core Philosophical Thinking (Default)
            "philosophical": SixHatsTeamConfiguration(
                name="PhilosophicalThinkingTeam",
                description="Specialized for philosophical and existential questions",
                hat_sequence=[HatColor.WHITE, HatColor.GREEN, HatColor.BLUE],
                instructions=[
                    "Process philosophical questions through structured Six Hats thinking.",
                    "White Hat: Present relevant facts and philosophical concepts.",
                    "Green Hat: Explore creative perspectives and new ideas.",
                    "Blue Hat: Synthesize insights into a coherent, human-friendly response.",
                    "Focus on practical wisdom while acknowledging complexity.",
                ],
            ),
            # Creative Problem Solving
            "creative": SixHatsTeamConfiguration(
                name="CreativeThinkingTeam",
                description="Creative problem-solving and innovation",
                hat_sequence=[
                    HatColor.RED,
                    HatColor.GREEN,
                    HatColor.YELLOW,
                    HatColor.BLUE,
                ],
                instructions=[
                    "Apply creative thinking to generate innovative solutions.",
                    "Red Hat: Express initial feelings and reactions.",
                    "Green Hat: Generate creative alternatives and new ideas.",
                    "Yellow Hat: Explore potential benefits and opportunities.",
                    "Blue Hat: Synthesize creative insights into actionable recommendations.",
                ],
            ),
            # Decision Making
            "decision": SixHatsTeamConfiguration(
                name="DecisionMakingTeam",
                description="Balanced decision analysis and evaluation",
                hat_sequence=[
                    HatColor.WHITE,
                    HatColor.BLACK,
                    HatColor.YELLOW,
                    HatColor.BLUE,
                ],
                instructions=[
                    "Analyze decisions through multiple perspectives.",
                    "White Hat: Present relevant facts and data.",
                    "Black Hat: Identify risks, problems, and potential failures.",
                    "Yellow Hat: Explore benefits, opportunities, and positive outcomes.",
                    "Blue Hat: Synthesize analysis into balanced decision guidance.",
                ],
            ),
            # Comprehensive Analysis
            "full": SixHatsTeamConfiguration(
                name="FullSixHatsThinkingTeam",
                description="Complete Six Thinking Hats analysis",
                hat_sequence=[
                    HatColor.BLUE,  # Process overview
                    HatColor.WHITE,  # Facts and data
                    HatColor.RED,  # Emotions and intuition
                    HatColor.YELLOW,  # Benefits and optimism
                    HatColor.BLACK,  # Caution and criticism
                    HatColor.GREEN,  # Creativity and alternatives
                    HatColor.BLUE,  # Final synthesis
                ],
                instructions=[
                    "Apply complete Six Thinking Hats methodology for comprehensive analysis.",
                    "Follow the hat sequence strictly, maintaining thinking mode purity.",
                    "Blue Hat manages process and provides final integration.",
                    "Each hat contributes its unique perspective without mixing modes.",
                    "Focus on practical, actionable insights.",
                ],
            ),
        }

        self._router = SixHatsIntelligentRouter()
        logger.info("Six Hats Team Factory initialized")

    def create_team(
        self, team_type: str = "philosophical", model_config: ModelConfig | None = None
    ) -> Team:
        """Create a Six Thinking Hats team."""
        if model_config is None:
            model_config = get_model_config()

        if team_type not in self._predefined_configs:
            available = ", ".join(self._predefined_configs.keys())
            raise ValueError(f"Unknown team type '{team_type}'. Available: {available}")

        config = self._predefined_configs[team_type]
        builder = SixHatsTeamBuilder(config)
        return builder.build_team(model_config)

    def create_adaptive_team(
        self, thought_content: str, model_config: ModelConfig | None = None
    ) -> Team:
        """Create an adaptive Six Hats team based on thought content."""
        if model_config is None:
            model_config = get_model_config()

        # Use intelligent router to determine optimal sequence
        optimal_sequence = self._router.route_to_hat_sequence(thought_content)

        # Create custom configuration
        config = SixHatsTeamConfiguration(
            name="AdaptiveSixHatsTeam",
            description=f"Adaptive team with sequence: {' → '.join(h.value for h in optimal_sequence)}",
            hat_sequence=optimal_sequence,
            instructions=[
                "Apply adaptive Six Thinking Hats based on problem characteristics.",
                "Follow the intelligently determined hat sequence.",
                "Each hat maintains its unique thinking mode strictly.",
                "Blue Hat provides final synthesis and integration.",
            ],
        )

        builder = SixHatsTeamBuilder(config)
        return builder.build_team(model_config)

    def create_custom_team(
        self,
        hat_sequence: list[HatColor],
        team_name: str = "CustomSixHatsTeam",
        model_config: ModelConfig | None = None,
    ) -> Team:
        """Create a custom Six Thinking Hats team with specified sequence."""
        if model_config is None:
            model_config = get_model_config()

        config = SixHatsTeamConfiguration(
            name=team_name,
            description=f"Custom team with sequence: {' → '.join(h.value for h in hat_sequence)}",
            hat_sequence=hat_sequence,
            instructions=[
                "Apply custom Six Thinking Hats sequence.",
                "Maintain strict hat discipline - one mode at a time.",
                "Follow the specified sequence exactly.",
                "Provide integrated output through Blue Hat coordination.",
            ],
        )

        builder = SixHatsTeamBuilder(config)
        return builder.build_team(model_config)

    def get_available_team_types(self) -> list[str]:
        """Get available predefined team types."""
        return list(self._predefined_configs.keys())

    def get_team_info(self, team_type: str) -> dict[str, Any]:
        """Get information about a specific team type."""
        if team_type not in self._predefined_configs:
            return {}

        config = self._predefined_configs[team_type]
        return {
            "name": config.name,
            "description": config.description,
            "hat_sequence": [h.value for h in config.hat_sequence],
            "routing_strategy": config.routing_strategy,
        }


# Global factory instance
_six_hats_team_factory = SixHatsTeamFactory()


# Public API functions
def create_six_hats_team(
    team_type: str = "philosophical", model_config: ModelConfig | None = None
) -> Team:
    """Create a Six Thinking Hats team by type."""
    return _six_hats_team_factory.create_team(team_type, model_config)


def create_adaptive_six_hats_team(
    thought_content: str, model_config: ModelConfig | None = None
) -> Team:
    """Create an adaptive Six Thinking Hats team based on content."""
    return _six_hats_team_factory.create_adaptive_team(thought_content, model_config)


def create_custom_six_hats_team(
    hat_sequence: list[HatColor],
    team_name: str = "CustomSixHatsTeam",
    model_config: ModelConfig | None = None,
) -> Team:
    """Create a custom Six Thinking Hats team."""
    return _six_hats_team_factory.create_custom_team(
        hat_sequence, team_name, model_config
    )


def get_available_six_hats_teams() -> list[str]:
    """Get available Six Thinking Hats team types."""
    return _six_hats_team_factory.get_available_team_types()


# Backward compatibility functions
def create_team(model_config: ModelConfig | None = None) -> Team:
    """Create default Six Thinking Hats team."""
    return create_six_hats_team("philosophical", model_config)


def create_team_by_type(
    team_type: str, model_config: ModelConfig | None = None
) -> Team:
    """Create team by type - Six Hats only."""
    # Map legacy types to Six Hats equivalents
    team_type_mapping = {
        "standard": "philosophical",
        "enhanced": "full",
        "hybrid": "creative",
        "enhanced_specialized": "decision",
    }

    # Use mapping if it's a legacy type
    if team_type in team_type_mapping:
        logger.info(
            f"Mapping legacy team type '{team_type}' to Six Hats '{team_type_mapping[team_type]}'"
        )
        team_type = team_type_mapping[team_type]

    # Handle six_hats_ prefixed types
    if team_type.startswith("six_hats_"):
        team_type = team_type.replace("six_hats_", "")

    return create_six_hats_team(team_type, model_config)


# Completely remove old UnifiedTeamFactory and other legacy classes
# This file now only contains Six Thinking Hats implementations
