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

**Pure Multi-Thinking Implementation** built with **Agno v2.0** framework and served via MCP. Features **intelligent routing** with clean architecture (src layout, Python 3.10+). The system processes thoughts through multi-directional thinking methodology with AI-powered routing and model optimization.

### Core Architecture

**Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py`
- FastMCP application with `sequentialthinking` tool
- Uses refactored service-based architecture with dependency injection
- Global state management via `ServerState` and `ThoughtProcessor`

**Multi-Thinking Processing Flow:**
```
External LLM → sequentialthinking tool → ThoughtProcessor → WorkflowExecutor → MultiThinkingWorkflowRouter → MultiThinkingSequentialProcessor → Individual Thinking Agents → Synthesis
```

**Core Services (Dependency Injection):**
- **ThoughtProcessor**: Main orchestrator using specialized services
- **WorkflowExecutor**: Manages Multi-Thinking workflow execution
- **ContextBuilder**: Builds context-aware prompts
- **ResponseFormatter**: Formats final responses
- **SessionMemory**: Tracks thought history and branching (no Team dependency)

**Multi-Thinking Routing System:**
- **MultiThinkingIntelligentRouter**: Complexity analysis determines hat sequence
- **MultiThinkingSequentialProcessor**: Executes chosen hat sequence with model optimization
- **HatComplexity levels**: SINGLE, DOUBLE, TRIPLE, FULL sequences
- **Model Intelligence**: Enhanced model for Blue Hat synthesis, Standard model for individual hats

### Configuration & Data Flow

**Environment Variables:**
- `LLM_PROVIDER`: Provider selection (deepseek, groq, openrouter, ollama, github, anthropic)
- `{PROVIDER}_API_KEY`: API keys (e.g., `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`)
- `{PROVIDER}_ENHANCED_MODEL_ID`: Enhanced model for complex synthesis (Blue Hat)
- `{PROVIDER}_STANDARD_MODEL_ID`: Standard model for individual hat processing
- `EXA_API_KEY`: Research capabilities (if using research agents)

**Model Strategy:**
- **Enhanced Models**: Used for Blue Hat (metacognitive) thinking - complex synthesis, integration
- **Standard Models**: Used for individual hat processing (White, Red, Black, Yellow, Green)
- **Intelligent Selection**: System automatically chooses appropriate model based on hat type

**Processing Strategies:**
1. **Single Hat**: Simple focused thinking (White Hat facts, Red Hat emotions, etc.)
2. **Double Hat**: Two-step sequences (e.g., White→Blue for factual synthesis)
3. **Triple Hat**: Core philosophical thinking (White→Green→Blue)
4. **Full Sequence**: Complete Multi-Thinking methodology with Blue Hat orchestration

### Key Modules

**Core Framework:**
- `core/session.py`: SessionMemory for thought history (simplified, no Team dependency)
- `core/models.py`: ThoughtData validation and core data structures
- `config/modernized_config.py`: Provider strategies with Enhanced/Standard model configuration

**Multi-Thinking Implementation:**
- `processors/multi_thinking_processor.py`: Main Multi-Thinking sequential processor
- `processors/multi_thinking_core.py`: Hat definitions, agent factory, core logic
- `routing/multi_thinking_router.py`: Intelligent routing based on thought complexity
- `routing/agno_workflow_router.py`: Agno Workflow integration layer

**Service Layer:**
- `services/thought_processor_refactored.py`: Main thought processor with dependency injection
- `services/workflow_executor.py`: Multi-Thinking workflow execution
- `services/context_builder.py`: Context-aware prompt building
- `services/response_formatter.py`: Response formatting and extraction

### Testing & Key Notes

**Test Organization:**
- Unit tests: `tests/unit/` | Integration: `tests/` root
- Async configuration: `tests/pytest.ini` with `asyncio_mode = auto`
- Factories: `tests/helpers/` for test data creation
- Key test files: `test_agno_workflow.py`, `test_github_provider.py`

**Architecture Characteristics:**
- **Clean Architecture**: Dependency injection, separation of concerns, service-based design
- **Multi-Thinking Focus**: Pure implementation without legacy multi-agent complexity
- **Model Optimization**: Smart model selection (Enhanced for synthesis, Standard for processing)
- **Modern Python**: Dataclasses, type hints, async/await, pattern matching
- **Environment-based config**: No config files, all via environment variables
- **Structured logging**: Rotation to `~/.sequential_thinking/logs/`

## Enhanced/Standard Model Configuration

**Naming Convention (NEW):**
- `{PROVIDER}_ENHANCED_MODEL_ID`: For complex synthesis tasks (Blue Hat thinking)
- `{PROVIDER}_STANDARD_MODEL_ID`: For individual hat processing

**Examples:**
```bash
# GitHub Models
GITHUB_ENHANCED_MODEL_ID="openai/gpt-5"      # Blue Hat synthesis
GITHUB_STANDARD_MODEL_ID="openai/gpt-5-min"  # Individual hats

# DeepSeek
DEEPSEEK_ENHANCED_MODEL_ID="deepseek-chat"   # Both synthesis and processing
DEEPSEEK_STANDARD_MODEL_ID="deepseek-chat"

# Anthropic
ANTHROPIC_ENHANCED_MODEL_ID="claude-3-5-sonnet-20241022"  # Synthesis
ANTHROPIC_STANDARD_MODEL_ID="claude-3-5-haiku-20241022"   # Processing
```

**Usage Strategy:**
- **Enhanced Model**: Blue Hat (metacognitive orchestrator) uses enhanced model for final synthesis
- **Standard Model**: Individual hats (White, Red, Black, Yellow, Green) use standard model
- **Automatic Selection**: System chooses model based on hat type and complexity

## Agno v2.0 Integration

**Framework Features:**
- **Workflow Integration**: Uses Agno Workflow system for Multi-Thinking processing
- **Agent Factory**: Creates specialized hat agents with ReasoningTools
- **Performance**: ~10,000x faster agent creation, ~50x less memory vs LangGraph
- **Version**: Requires `agno>=2.0.5`

**Key Integration Points:**
- `MultiThinkingWorkflowRouter`: Bridges MCP and Agno Workflow systems
- `MultiThinkingAgentFactory`: Creates individual hat agents using Agno v2.0
- **StepOutput**: Workflow results converted to Agno StepOutput format

**For Agno Documentation**: Use deepwiki MCP reference with repoName: `agno-agi/agno`