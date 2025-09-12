# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies & Environment
- Use `uv` for dependency management (preferred over pip)
- Install dependencies: `uv pip install -e .` (uses pyproject.toml dependencies)
- Install dev dependencies: `uv pip install -e ".[dev]"` (pytest, black, isort, mypy)
- Upgrade agno: `uv pip install --upgrade agno`
- Test Python imports: `uv run python -c "import agno; print('Agno imported successfully')"`
- Project requires Python 3.10+ and uses modern packaging (PEP 621)

### Code Quality
- Linting: `ruff check . --fix`
- Formatting: `black .`
- Type checking: `mypy .`
- Testing: `pytest` (runs from tests/ directory with async support)
- Run single test: `pytest tests/unit/test_models.py::test_thought_data_validation -v`
- Run with coverage: `pytest --cov=src --cov-report=html`

### Running the Server
- Direct execution: `uv run python src/mcp_server_mas_sequential_thinking/main.py`
- Using uv: `uv run mcp-server-mas-sequential-thinking`  
- Package execution: `uvx mcp-server-mas-sequential-thinking`

## Project Architecture

This is a Multi-Agent System (MAS) for sequential thinking built with the Agno framework and served via MCP. The project follows modern Python packaging standards with **src layout** structure.

### Core Components

**Project Structure:**
- `src/mcp_server_mas_sequential_thinking/` contains all Python modules
- `tests/` directory with comprehensive unit and integration tests
- `docs/` directory for documentation organization

**Main Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py` serves as the FastMCP application entry point with refactored architecture:
- Server lifespan management and FastMCP setup
- Core logic delegated to specialized modules:
  - `server_core.py`: Server state, configuration, and thought processing
  - `models.py`: Pydantic models for data validation (ThoughtData)  
  - `team.py` / `unified_team.py`: Multi-agent team implementations
  - `agents.py`: Individual agent definitions and roles

**Agent Architecture (Agno v2.0):**
- **Team Coordinator:** Uses Agno's `Team` with v2 coordination attributes (respond_directly=False, delegate_task_to_all_members=False)
- **Specialist Agents:** Planner, Researcher, Analyzer, Critic, Synthesizer using ReasoningTools
- **Agent Flow:** Coordinator receives thoughts → delegates to specialists → synthesizes responses
- **Performance:** ~10,000x faster agent creation, ~50x less memory usage vs LangGraph

### Key Components

**Core Functions:**
- `create_sequential_thinking_team()`: Instantiates multi-agent team with specialized roles
- `sequentialthinking` tool (in main.py): Core MCP tool that processes ThoughtData objects  
- `get_model_config()` (in config.py): Configures LLM providers

**Architecture Modules:**
- `ThoughtProcessor` (server_core.py): Central processing logic with async team coordination
- `ServerState` & `ServerConfig`: State management and configuration containers
- `SessionMemory` (session.py): In-memory state tracking with branch support
- Multiple team implementations: `team.py`, `unified_team.py` for different coordination strategies

### Configuration

Environment variables control behavior:
- `LLM_PROVIDER`: Provider selection (deepseek, groq, openrouter, ollama, github)
- `{PROVIDER}_API_KEY`: API keys for each provider (e.g., `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`)
- `{PROVIDER}_{TEAM|AGENT}_MODEL_ID`: Model selection for coordinator vs specialists
- `EXA_API_KEY`: For research capabilities

**GitHub Models Support:**
- Enhanced GitHub token validation with format checking
- Supports PAT tokens and OAuth tokens
- Uses custom `GitHubOpenAI` class extending OpenAI for GitHub Models API

### Data Flow

1. External LLM calls `sequentialthinking` tool with ThoughtData
2. Tool validates input via Pydantic model
3. Coordinator analyzes thought and delegates to relevant specialists
4. Specialists process sub-tasks using their tools (ThinkingTools, ExaTools)
5. Coordinator synthesizes responses and returns guidance
6. Process continues with revisions/branches as needed

### Memory & State

- **SessionMemory:** In-memory storage for thought history and branches
- **Logging:** Structured logging to `~/.sequential_thinking/logs/`
- **Branch Management:** Supports non-linear thinking with branch tracking

### Testing Architecture

**Test Structure:**
- Unit tests in `tests/unit/` with comprehensive coverage
- Integration tests at `tests/` root level  
- Async test configuration in `tests/pytest.ini` with coverage reporting
- Well-organized fixtures in `tests/conftest.py`
- Factory pattern for mock data in `tests/helpers/factories.py`
- Mock utilities in `tests/helpers/mocks.py`

**Key Testing Patterns:**
- Async test configuration (`asyncio_mode = auto`) with proper event loop management
- Comprehensive mocking of external dependencies (Agno teams, API calls)
- Pydantic model validation testing with error handling scenarios
- Test-driven development approach with enhanced validation coverage
- Performance and integration test markers for categorized test runs

## Important Notes

- **High token usage**: Multi-agent architecture leads to 3-6x higher token consumption per thought
- **Modular design**: Clean separation allows independent agent development and testing
- **Modern Python practices**: Uses dataclasses, type hints, async/await, and pattern matching
- **Environment-based configuration**: No config files, all settings via environment variables
- **Comprehensive logging**: Structured logging with rotation to `~/.sequential_thinking/logs/`

## Agno v2.0 Migration

This project has been migrated to Agno v2.0 with the following key changes:

### Architecture Updates
- **Team coordination**: Replaced `mode="coordinate"` with explicit v2 attributes
  - `respond_directly=False` - Team leader processes member responses
  - `delegate_task_to_all_members=False` - Sequential task delegation  
  - `determine_input_for_members=True` - Team leader synthesizes inputs
- **Tool modules**: Migrated from `ThinkingTools` to `ReasoningTools` (`agno.tools.reasoning`)
- **Memory management**: Updated `enable_memory` to `enable_user_memories` parameter

### Performance Improvements
- **~10,000x faster** agent creation compared to LangGraph
- **~50x less memory** usage for agent instances
- **Microsecond-level** factory and configuration initialization
- **Optimized imports** and module loading

### Compatibility
- **Backward compatible**: All public APIs remain unchanged
- **Environment variables**: Same configuration approach maintained
- **Functionality preserved**: All existing features work identically