"""Streamlined team factory with consolidated imports and reduced redundancy."""

import logging
import os
from agno.team.team import Team
from .modernized_config import get_model_config
from .unified_agents import UnifiedAgentFactory

logger = logging.getLogger(__name__)

# Optimized coordinator instruction sets
_ENHANCED_INSTRUCTIONS = [
    "You coordinate enhanced specialists with advanced reasoning capabilities for sequential thinking.",
    "Available specialists: Enhanced agents with memory, reasoning, and structured outputs.",
    "Process: 1) Analyze input with reasoning, 2) Select optimal specialists, 3) Delegate with context, 4) Synthesize structured responses, 5) Provide actionable guidance.",
    "Use structured outputs for recommendations: 'RECOMMENDATION: ...' or 'SUGGESTION: ...'",
    "Leverage agent memory and reasoning capabilities for complex tasks.",
    "Prioritize accuracy and depth over speed when dealing with complex reasoning tasks.",
]

_STANDARD_INSTRUCTIONS = [
    "You coordinate specialists (Planner, Researcher, Analyzer, Critic, Synthesizer) for sequential thinking.",
    "Process: 1) Analyze input thought, 2) Identify required specialists (minimum needed), 3) Delegate clear sub-tasks, 4) Synthesize responses, 5) Provide guidance.",
    "Include recommendations: 'RECOMMENDATION: Revise thought #X...' or 'SUGGESTION: Consider branching from thought #Y...'",
    "Prioritize efficiency - only delegate to specialists whose expertise is strictly necessary.",
]

def _get_team_config(enhanced: bool) -> dict:
    """Get team configuration based on enhancement level."""
    if enhanced:
        return {
            "name": "EnhancedSequentialThinkingTeam",
            "description": "Enhanced coordinator with advanced reasoning capabilities",
            "instructions": _ENHANCED_INSTRUCTIONS,
            "success_criteria": [
                "Efficiently delegate sub-tasks to relevant specialists",
                "Synthesize specialist responses coherently", 
                "Recommend revisions or branches based on analysis",
                "Utilize agent memory for context retention",
                "Apply structured reasoning patterns",
                "Generate structured outputs",
            ],
        }
    return {
        "name": "SequentialThinkingTeam", 
        "description": "Coordinator for sequential thinking specialist team",
        "instructions": _STANDARD_INSTRUCTIONS,
        "success_criteria": [
            "Efficiently delegate sub-tasks to relevant specialists",
            "Synthesize specialist responses coherently",
            "Recommend revisions or branches based on analysis",
        ],
    }


def create_team(enhanced: bool = None) -> Team:
    """Create sequential thinking team with optimized configuration and unified agents."""
    config = get_model_config()
    
    # Auto-detect enhanced mode from environment with default to True
    if enhanced is None:
        enhanced = os.getenv("USE_ENHANCED_AGENTS", "true").lower() == "true"

    # Create model instances
    team_model = config.provider_class(id=config.team_model_id)
    agent_model = config.provider_class(id=config.agent_model_id)

    # Use unified factory for agent creation 
    factory = UnifiedAgentFactory()
    team_type = "enhanced" if enhanced else "standard"
    agents = factory.create_team_agents(agent_model, team_type)
    
    # Configure team parameters based on enhancement level
    team_config = _get_team_config(enhanced)
    logger.info(f"Creating {team_config['name']} with unified agent factory")

    # Create and configure team with optimized configuration
    return Team(
        name=team_config["name"],
        members=list(agents.values()),
        model=team_model,
        description=team_config["description"],
        instructions=team_config["instructions"], 
        success_criteria=team_config["success_criteria"],
        # Agno v2 configuration attributes
        respond_directly=False,  # Team leader processes responses from members
        delegate_task_to_all_members=False,  # Delegate one by one
        determine_input_for_members=True,  # Team leader synthesizes input
        enable_agentic_context=enhanced,  # Enhanced context for enhanced mode
        share_member_interactions=enhanced,  # Share interactions in enhanced mode
        markdown=True,
    )


def create_enhanced_team() -> Team:
    """Create enhanced team with Agno v2 features."""
    return create_team(enhanced=True)


def create_hybrid_team_instance() -> Team:
    """Create hybrid team with mixed standard and enhanced agents."""
    config = get_model_config()

    # Create model instances  
    team_model = config.provider_class(id=config.team_model_id)
    agent_model = config.provider_class(id=config.agent_model_id)

    # Use unified factory for hybrid agent creation
    factory = UnifiedAgentFactory()
    agents = factory.create_team_agents(agent_model, "hybrid")

    logger.info("Creating hybrid team with unified agent factory")
    
    # Create hybrid team with specialized configuration
    return Team(
        name="HybridSequentialThinkingTeam",
        members=list(agents.values()),
        model=team_model,
        description="Hybrid coordinator with mixed standard and enhanced capabilities",
        instructions=_ENHANCED_INSTRUCTIONS,  # Use enhanced instructions for hybrid
        success_criteria=[
            "Efficiently utilize both standard and enhanced specialists",
            "Balance speed and accuracy based on task complexity", 
            "Synthesize responses from diverse agent capabilities",
            "Adapt delegation strategy to agent strengths",
        ],
        # Agno v2 configuration
        respond_directly=False,
        delegate_task_to_all_members=False,
        determine_input_for_members=True,
        enable_agentic_context=True,
        share_member_interactions=True,
        markdown=True,
    )
