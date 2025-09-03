"""
MCP Sequential Thinking Server

A refactored Multi-Agent System (MAS) for sequential thinking built with the Agno framework and served via MCP.

REFACTORED ARCHITECTURE:
- Eliminated redundancy between agents.py and enhanced_agents.py via unified factory
- Simplified team creation with factory pattern and eliminated conditional complexity  
- Separated server initialization concerns for better maintainability
- Modernized configuration management with dependency injection patterns
- Optimized imports and eliminated circular dependencies

Key Features:
- Multi-agent coordination with specialized roles
- Sequential thinking with branching and revision support
- Configurable provider support (DeepSeek, Groq, OpenRouter, Ollama, GitHub)
- Session memory for thought history and branch management
- Enhanced error handling and structured logging

Usage:
    from mcp_server_mas_sequential_thinking import main
    main.run()

New Unified Architecture:
    - UnifiedAgentFactory: Eliminates duplication, supports standard/enhanced modes
    - UnifiedTeamFactory: Simplified team creation with builder pattern
    - ServerCore: Separated concerns for initialization, processing, and lifecycle
    - ModernizedConfig: Dependency injection with strategy pattern

Team Types:
    - "standard": Basic agents with core functionality
    - "enhanced": Enhanced agents with advanced reasoning and memory
    - "hybrid": Mix of standard and enhanced agents for optimal performance
    - "enhanced_specialized": Specialized enhanced agents for complex reasoning

Configuration:
    Environment variables:
    - LLM_PROVIDER: Provider selection (deepseek, groq, openrouter, ollama, github)
    - TEAM_MODE: Team type selection (standard, enhanced, hybrid, enhanced_specialized)
    - {PROVIDER}_API_KEY: API keys for each provider
    - {PROVIDER}_{TEAM|AGENT}_MODEL_ID: Model selection
    - EXA_API_KEY: For research capabilities

Migration Notes:
    - Old modules (agents.py, enhanced_agents.py, team.py, config.py) are deprecated
    - Use unified_agents, unified_team, and modernized_config instead
    - All existing functionality is preserved with improved architecture
"""

__version__ = "0.5.1-refactored"

# Export refactored components for external use
from .modernized_config import get_model_config, get_available_providers
from .unified_agents import UnifiedAgentFactory, create_agent
from .unified_team import create_team_by_type
from .server_core import ServerConfig, ThoughtProcessor
from .models import ThoughtData, ThoughtType
