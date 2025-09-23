"""MCP Sequential Thinking Server with AI-Powered Routing.

A sophisticated Multi-Agent System (MAS) for sequential thinking with intelligent
AI-based routing using advanced multi-thinking methodology.

AI ROUTING FEATURES:
- AI-powered complexity analysis replacing rule-based scoring
- Intelligent thinking sequence selection (single, double, triple, full)
- Semantic understanding of philosophical and technical depth
- Automatic synthesis integration for complex thought processing

CORE ARCHITECTURE:
- AI-first routing with semantic complexity understanding
- Multi-thinking methodology with intelligent orchestration
- Unified agent factory eliminating code duplication
- Separated server concerns for better maintainability
- Optimized performance with async processing

Key Features:
- **AI Routing**: Semantic complexity analysis and strategy selection
- **Multi-Thinking Integration**: Factual, emotional, critical, optimistic, creative, and synthesis processing
- **Multi-Provider Support**: DeepSeek, Groq, OpenRouter, GitHub, Ollama
- **Philosophical Understanding**: Deep analysis for existential questions

Usage:
    # Server usage
    uv run mcp-server-mas-sequential-thinking

    # Direct processing
    from mcp_server_mas_sequential_thinking.services.server_core import ThoughtProcessor
    from mcp_server_mas_sequential_thinking.core.session import SessionMemory

    processor = ThoughtProcessor(session_memory)
    result = await processor.process_thought(thought_data)

AI Routing Strategies:
    - Single Direction: Simple factual questions → quick focused processing
    - Double/Triple Direction: Moderate complexity → focused thinking sequences
    - Full Multi-Thinking: Complex philosophical questions → comprehensive analysis

Configuration:
    Environment variables:
    - LLM_PROVIDER: Primary provider (deepseek, groq, openrouter, github, ollama)
    - {PROVIDER}_API_KEY: API keys for providers
    - {PROVIDER}_{TEAM|AGENT}_MODEL_ID: Model selection
    - AI_CONFIDENCE_THRESHOLD: Routing confidence threshold (default: 0.7)

Performance Benefits:
    - Intelligent complexity assessment using AI understanding
    - Automated synthesis processing for coherent responses
    - Semantic depth recognition for philosophical questions
    - Efficient thinking sequence selection based on content analysis
"""

__version__ = "0.6.0-ai-routing"


def get_version() -> str:
    """Get package version."""
    return __version__
