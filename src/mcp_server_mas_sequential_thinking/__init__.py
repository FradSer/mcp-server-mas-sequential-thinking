"""
MCP Sequential Thinking Server with Adaptive Architecture

A sophisticated Multi-Agent System (MAS) for sequential thinking with intelligent routing,
cost optimization, and persistent memory management.

ADAPTIVE ARCHITECTURE FEATURES:
- Intelligent complexity-based routing (simple → single agent, complex → multi-agent)
- Cost optimization with budget controls and provider selection
- Persistent memory management with SQLAlchemy storage
- Performance monitoring and optimization suggestions
- Token usage reduction of 60-80% for simple thoughts

CORE ARCHITECTURE:
- Unified agent factory eliminating code duplication
- Builder pattern team creation with simplified complexity
- Separated server concerns for better maintainability
- Protocol-based dependency injection for providers
- Optimized performance with O(1) lookups and caching

Key Features:
- **Adaptive Routing**: Automatic complexity analysis and strategy selection
- **Cost Optimization**: Budget-aware provider selection and usage monitoring
- **Persistent Memory**: SQLAlchemy-based storage with memory pruning
- **Multi-Provider Support**: DeepSeek, Groq, OpenRouter, GitHub, Ollama
- **Performance Analytics**: Token usage, cost tracking, and optimization metrics

Usage:
    # Basic usage
    from mcp_server_mas_sequential_thinking import main
    main.run()

    # Adaptive processing
    from mcp_server_mas_sequential_thinking.adaptive_server import create_adaptive_processor
    processor = create_adaptive_processor()
    result = await processor.process_thought_adaptive(thought_data)

Adaptive Strategies:
    - "single_agent": Fast, cost-effective for simple thoughts (60-80% cost savings)
    - "multi_agent": Comprehensive analysis for complex thoughts
    - "hybrid": Adaptive mix based on complexity and budget constraints

Configuration:
    Environment variables:
    - ENABLE_ADAPTIVE_ROUTING: Enable intelligent routing (default: true)
    - DAILY_BUDGET_LIMIT: Daily spending limit in USD
    - MONTHLY_BUDGET_LIMIT: Monthly spending limit in USD
    - QUALITY_THRESHOLD: Minimum quality score for provider selection
    - DATABASE_URL: Database connection for persistent memory
    - LLM_PROVIDER: Primary provider (deepseek, groq, openrouter, github, ollama)

Performance Benefits:
    - 60-80% token reduction for simple thoughts
    - Automatic cost optimization and budget management
    - Persistent session memory across restarts
    - Real-time usage analytics and optimization suggestions
"""

__version__ = "0.6.0-adaptive"

# Export AI-powered routing components
from .ai_routing import (
    create_ai_router,
    HybridComplexityAnalyzer,
    ComplexityLevel,
    ProcessingStrategy,
)
from .cost_optimization import (
    CostOptimizer,
    create_cost_optimizer,
    ProviderProfile,
    BudgetConstraints,
)
from .persistent_memory import PersistentMemoryManager, create_persistent_memory
from .adaptive_server import AdaptiveThoughtProcessor, create_adaptive_processor

# Export core components for external use
from .modernized_config import get_model_config, get_available_providers
from .unified_agents import UnifiedAgentFactory, create_agent
from .unified_team import create_team_by_type
from .server_core import ServerConfig, ThoughtProcessor
from .models import ThoughtData, ThoughtType
