"""MCP Sequential Thinking Server with AI-Powered Routing.

A sophisticated Multi-Agent System (MAS) for sequential thinking with intelligent
AI-based routing using the Six Hats methodology.

AI ROUTING FEATURES:
- AI-powered complexity analysis replacing rule-based scoring
- Intelligent Six Hats sequence selection (single, double, triple, full)
- Semantic understanding of philosophical and technical depth
- Automatic Blue Hat integration for complex thought synthesis

CORE ARCHITECTURE:
- AI-first routing with semantic complexity understanding
- Six Hats thinking methodology with intelligent orchestration
- Unified agent factory eliminating code duplication
- Separated server concerns for better maintainability
- Optimized performance with async processing

Key Features:
- **AI Routing**: Semantic complexity analysis and strategy selection
- **Six Hats Integration**: White, Red, Black, Yellow, Green, Blue hat processing
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
    - Single Hat: Simple factual questions → quick single hat processing
    - Double/Triple Hat: Moderate complexity → focused hat sequences
    - Full Six Hats: Complex philosophical questions → comprehensive analysis

Configuration:
    Environment variables:
    - LLM_PROVIDER: Primary provider (deepseek, groq, openrouter, github, ollama)
    - {PROVIDER}_API_KEY: API keys for providers
    - {PROVIDER}_{TEAM|AGENT}_MODEL_ID: Model selection
    - AI_CONFIDENCE_THRESHOLD: Routing confidence threshold (default: 0.7)

Performance Benefits:
    - Intelligent complexity assessment using AI understanding
    - Automated Blue Hat synthesis for coherent responses
    - Semantic depth recognition for philosophical questions
    - Efficient hat sequence selection based on content analysis
"""

__version__ = "0.6.0-ai-routing"


def get_version() -> str:
    """Get package version."""
    return __version__
