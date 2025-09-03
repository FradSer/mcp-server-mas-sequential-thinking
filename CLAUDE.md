# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Dependencies & Environment
- Use `uv` for dependency management (preferred over pip)
- Install dependencies: `uv pip install -e .` or `uv pip install -r requirements.txt`
- Install dev dependencies: `uv pip install -e ".[dev]"`
- Upgrade agno: `uv pip install --upgrade agno`
- Test Python imports: `uv run python -c "import agno; print('Agno imported successfully')"`

### Code Quality
- Linting: `ruff check . --fix`
- Formatting: `black .`
- Type checking: `mypy .`
- Testing: `pytest`

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

**Main Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py` contains all core logic:
- FastMCP server setup
- ThoughtData Pydantic model for input validation
- Multi-agent team creation and coordination
- Sequential thinking tool implementation

**Agent Architecture:**
- **Team Coordinator:** Uses Agno's `Team` object in `coordinate` mode
- **Specialist Agents:** Planner, Researcher, Analyzer, Critic, Synthesizer
- **Agent Flow:** Coordinator receives thoughts → delegates to specialists → synthesizes responses

### Key Functions

**`create_sequential_thinking_team()`:** Instantiates the multi-agent team with specialized roles
**`sequentialthinking` tool:** Core MCP tool that processes ThoughtData objects
**`get_model_config()`:** Configures LLM providers (DeepSeek, Groq, OpenRouter)

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
- Comprehensive test coverage with unit tests in `tests/unit/`
- Integration tests at `tests/` root level
- Well-organized fixtures in `tests/conftest.py`
- Factory pattern for creating mock test data in `tests/helpers/`

**Key Testing Patterns:**
- Async test configuration with proper event loop management
- Comprehensive mocking of external dependencies (Agno teams, API calls)
- Validation testing for Pydantic models and error handling
- Test-driven development approach evident in validation tests

## Important Notes

- **High token usage**: Multi-agent architecture leads to 3-6x higher token consumption per thought
- **Modular design**: Clean separation allows independent agent development and testing
- **Modern Python practices**: Uses dataclasses, type hints, async/await, and pattern matching
- **Environment-based configuration**: No config files, all settings via environment variables
- **Comprehensive logging**: Structured logging with rotation to `~/.sequential_thinking/logs/`