"""Team factory for creating the sequential thinking team."""

import logging
import os
from agno.team.team import Team
from .config import get_model_config
from .agents import create_all_agents
from .enhanced_agents import create_all_enhanced_agents, create_hybrid_team

logger = logging.getLogger(__name__)

# Enhanced coordinator instructions for Agno 1.8+
ENHANCED_COORDINATOR_INSTRUCTIONS = [
    "You coordinate enhanced specialists with advanced reasoning capabilities for sequential thinking.",
    "Available specialists: Enhanced agents with memory, reasoning, and structured outputs.",
    "Process: 1) Analyze input with reasoning, 2) Select optimal specialists, 3) Delegate with context, 4) Synthesize structured responses, 5) Provide actionable guidance.",
    "Use structured outputs for recommendations: 'RECOMMENDATION: ...' or 'SUGGESTION: ...'",
    "Leverage agent memory and reasoning capabilities for complex tasks.",
    "Prioritize accuracy and depth over speed when dealing with complex reasoning tasks.",
]

# Standard coordinator instructions (backward compatibility)
COORDINATOR_INSTRUCTIONS = [
    "You coordinate specialists (Planner, Researcher, Analyzer, Critic, Synthesizer) for sequential thinking.",
    "Process: 1) Analyze input thought, 2) Identify required specialists (minimum needed), 3) Delegate clear sub-tasks, 4) Synthesize responses, 5) Provide guidance.",
    "Include recommendations: 'RECOMMENDATION: Revise thought #X...' or 'SUGGESTION: Consider branching from thought #Y...'",
    "Prioritize efficiency - only delegate to specialists whose expertise is strictly necessary.",
]


def create_team(enhanced: bool = None) -> Team:
    """Create the sequential thinking team with optional enhanced features."""
    config = get_model_config()
    
    # Auto-detect enhanced mode from environment or default to True
    if enhanced is None:
        enhanced = os.getenv("USE_ENHANCED_AGENTS", "true").lower() == "true"

    # Create model instances
    team_model = config.provider_class(id=config.team_model_id)
    agent_model = config.provider_class(id=config.agent_model_id)

    # Create specialist agents based on mode
    if enhanced:
        agents = create_all_enhanced_agents(agent_model)
        instructions = ENHANCED_COORDINATOR_INSTRUCTIONS
        team_name = "EnhancedSequentialThinkingTeam"
        description = "Enhanced coordinator with advanced reasoning capabilities"
        logger.info("Creating enhanced team with new Agno features")
    else:
        agents = create_all_agents(agent_model)
        instructions = COORDINATOR_INSTRUCTIONS
        team_name = "SequentialThinkingTeam"
        description = "Coordinator for sequential thinking specialist team"
        logger.info("Creating standard team")

    # Create and configure team
    team = Team(
        name=team_name,
        mode="coordinate",
        members=list(agents.values()),
        model=team_model,
        description=description,
        instructions=instructions,
        success_criteria=[
            "Efficiently delegate sub-tasks to relevant specialists",
            "Synthesize specialist responses coherently", 
            "Recommend revisions or branches based on analysis",
            *([
                "Utilize agent memory for context retention",
                "Apply structured reasoning patterns",
                "Generate structured outputs",
            ] if enhanced else [])
        ],
        enable_agentic_context=enhanced,  # Enhanced context for enhanced mode
        share_member_interactions=enhanced,  # Share interactions in enhanced mode
        markdown=True,
        add_datetime_to_instructions=True,
    )

    logger.info(f"Team created with {config.provider_class.__name__} provider (enhanced={enhanced})")
    return team


def create_enhanced_team() -> Team:
    """Create enhanced team with new Agno 1.8+ features."""
    return create_team(enhanced=True)


def create_hybrid_team_instance() -> Team:
    """Create hybrid team mixing standard and enhanced agents."""
    config = get_model_config()

    # Create model instances
    team_model = config.provider_class(id=config.team_model_id)
    agent_model = config.provider_class(id=config.agent_model_id)

    # Create hybrid agent mix
    agents = create_hybrid_team(agent_model)

    # Create and configure hybrid team
    team = Team(
        name="HybridSequentialThinkingTeam",
        mode="coordinate",
        members=list(agents.values()),
        model=team_model,
        description="Hybrid coordinator with mixed standard and enhanced capabilities",
        instructions=ENHANCED_COORDINATOR_INSTRUCTIONS,
        success_criteria=[
            "Efficiently utilize both standard and enhanced specialists",
            "Balance speed and accuracy based on task complexity",
            "Synthesize responses from diverse agent capabilities",
            "Adapt delegation strategy to agent strengths",
        ],
        enable_agentic_context=True,
        share_member_interactions=True,
        markdown=True,
        add_datetime_to_instructions=True,
    )

    logger.info(f"Hybrid team created with {config.provider_class.__name__} provider")
    return team
