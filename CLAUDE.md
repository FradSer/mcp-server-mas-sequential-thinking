# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Setup & Installation
uv pip install -e ".[dev]"                              # Install all dependencies
uv run python -c "import agno; print('Agno imported successfully')"  # Verify setup

# Development Workflow
uv run mcp-server-mas-sequential-thinking               # Run server
ruff check . --fix && ruff format . && mypy .           # Code quality (ruff replaces black + isort)
pytest --cov=. --cov-report=html                        # Test with coverage

# Monitoring & Debugging
tail -f ~/.sequential_thinking/logs/sequential_thinking.log  # Live logs
grep "ERROR\|WARNING" ~/.sequential_thinking/logs/sequential_thinking.log  # Error search
```

### Additional Commands
- **Upgrade agno**: `uv pip install --upgrade agno`
- **Single test**: `pytest tests/unit/test_models.py::test_thought_data_validation -v`
- **Alternative server runs**: `uvx mcp-server-mas-sequential-thinking` or `uv run python src/mcp_server_mas_sequential_thinking/main.py`
- **MCP Inspector**: `npx @modelcontextprotocol/inspector uv run python src/mcp_server_mas_sequential_thinking/main.py`

## Project Overview

Multi-Agent System (MAS) for sequential thinking built with **Agno v2.0** framework and served via MCP. Features **AI-powered intelligent routing** with modern Python packaging (src layout, Python 3.10+).

### Architecture

**Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py`
- FastMCP application with server lifespan management
- Delegates to specialized modules: `server_core.py`, `models.py`, `team.py`, `agents.py`

**Agent System (Agno v2.0):**
- **Coordinator:** Team leader with v2 attributes (`respond_directly=False`, `delegate_task_to_all_members=False`)
- **Specialists:** Planner, Researcher, Analyzer, Critic, Synthesizer using ReasoningTools
- **Performance:** ~10,000x faster agent creation, ~50x less memory vs LangGraph

**Core Modules:**
- `ThoughtProcessor`: Central async processing logic
- `SessionMemory`: In-memory state with branch support
- `ai_routing.py`: Complexity analysis and routing decisions
- `adaptive_routing.py`: Performance-based route optimization

### Configuration & Data Flow

**Environment Variables:**
- `LLM_PROVIDER`: Provider selection (deepseek, groq, openrouter, ollama, github)
- `{PROVIDER}_API_KEY`: API keys (e.g., `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`)
- `{PROVIDER}_{TEAM|AGENT}_MODEL_ID`: Model selection for coordinator vs specialists
- `EXA_API_KEY`: Research capabilities
- `AI_CONFIDENCE_THRESHOLD`: Routing confidence threshold (default: 0.7)

**Processing Flow:**
1. External LLM → `sequentialthinking` tool → ThoughtData validation
2. **AI Routing:** Complexity analysis selects strategy (`single_agent`, `hybrid`, `multi_agent`)
3. Coordinator delegates to specialists → synthesis → response
4. SessionMemory tracks history with branch support

### Testing & Key Notes

**Test Organization:**
- Unit tests: `tests/unit/` | Integration: `tests/` root
- Async configuration: `tests/pytest.ini` with `asyncio_mode = auto`
- Fixtures: `tests/conftest.py` | Factories: `tests/helpers/`

**Important Characteristics:**
- **High token usage**: 3-6x consumption due to multi-agent architecture
- **Modern Python**: Dataclasses, type hints, async/await, pattern matching
- **Environment-based config**: No config files, all via environment variables
- **Structured logging**: Rotation to `~/.sequential_thinking/logs/`

## Agno v2.0 Migration Notes

**Key Changes:**
- Team coordination: `respond_directly=False`, `delegate_task_to_all_members=False`
- Tools: `ThinkingTools` → `ReasoningTools` (`agno.tools.reasoning`)
- Memory: `enable_memory` → `enable_user_memories`
- Version: Requires `agno>=2.0.5`

**Performance Gains:**
- ~10,000x faster agent creation vs LangGraph
- ~50x less memory usage
- Microsecond-level initialization

**Compatibility:** Backward compatible, all APIs and environment variables unchanged
- **deepwiki MCP reference**: For agno framework documentation, use repoName: `agno-agi/agno`