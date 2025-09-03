"""Team factory for creating the sequential thinking team."""

import logging
from agno.team.team import Team
from .config import get_model_config
from .agents import create_all_agents

logger = logging.getLogger(__name__)

# Simplified coordinator instructions
COORDINATOR_INSTRUCTIONS = [
    "You coordinate specialists (Planner, Researcher, Analyzer, Critic, Synthesizer) for sequential thinking.",
    "Process: 1) Analyze input thought, 2) Identify required specialists (minimum needed), 3) Delegate clear sub-tasks, 4) Synthesize responses, 5) Provide guidance.",
    "Include recommendations: 'RECOMMENDATION: Revise thought #X...' or 'SUGGESTION: Consider branching from thought #Y...'",
    "Prioritize efficiency - only delegate to specialists whose expertise is strictly necessary.",
]


def create_team() -> Team:
    """Create the sequential thinking team with simplified configuration."""
    config = get_model_config()

    # Create model instances
    team_model = config.provider_class(id=config.team_model_id)
    agent_model = config.provider_class(id=config.agent_model_id)

    # Create specialist agents
    agents = create_all_agents(agent_model)

    # Create and configure team
    team = Team(
        name="SequentialThinkingTeam",
        mode="coordinate",
        members=list(agents.values()),
        model=team_model,
        description="Coordinator for sequential thinking specialist team",
        instructions=COORDINATOR_INSTRUCTIONS,
        success_criteria=[
            "Efficiently delegate sub-tasks to relevant specialists",
            "Synthesize specialist responses coherently",
            "Recommend revisions or branches based on analysis",
        ],
        enable_agentic_context=False,
        share_member_interactions=False,
        markdown=True,
        add_datetime_to_instructions=True,
    )

    logger.info(f"Team created with {config.provider_class.__name__} provider")
    return team
