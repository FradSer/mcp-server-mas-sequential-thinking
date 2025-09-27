# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Setup & Installation
uv pip install -e ".[dev]"                              # Install all dependencies
uv run python -c "import agno; print('Agno imported successfully')"  # Verify setup

# Development Workflow
uv run mcp-server-mas-sequential-thinking               # Run server
uv run ruff check . --fix && uv run ruff format . && uv run mypy .  # Code quality
uv run pytest --cov=. --cov-report=html                # Test with coverage (no tests currently)

# Monitoring & Debugging
tail -f ~/.sequential_thinking/logs/sequential_thinking.log  # Live logs
grep "ERROR\|WARNING" ~/.sequential_thinking/logs/sequential_thinking.log  # Error search
```

### Additional Commands
- **Upgrade agno**: `uv pip install --upgrade agno`
- **Alternative server runs**: `uvx mcp-server-mas-sequential-thinking` or `uv run python src/mcp_server_mas_sequential_thinking/main.py`
- **MCP Inspector**: `npx @modelcontextprotocol/inspector uv run python src/mcp_server_mas_sequential_thinking/main.py`

## Project Overview

**Pure Multi-Thinking Implementation** built with **Agno v2.0** framework and served via MCP. Features **AI-powered intelligent routing** with streamlined architecture (src layout, Python 3.10+). The system processes thoughts through multi-directional thinking methodology with AI-driven complexity analysis and optimized model selection.

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
- **SessionMemory**: Tracks thought history and branching

**AI-Powered Routing System:**
- **MultiThinkingIntelligentRouter**: AI-driven complexity analysis determines thinking sequence
- **AIComplexityAnalyzer**: Uses LLM to assess thought complexity, problem type, and required thinking modes
- **MultiThinkingSequentialProcessor**: Executes chosen sequence with model optimization
- **Thinking Complexity levels**: SINGLE, DOUBLE, TRIPLE, FULL sequences
- **Model Intelligence**: Enhanced model for Blue Hat synthesis, Standard model for individual hats

### Configuration & Data Flow

**Environment Variables:**
- `LLM_PROVIDER`: Provider selection (deepseek, groq, openrouter, ollama, github, anthropic)
- `{PROVIDER}_API_KEY`: API keys (e.g., `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`)
- `{PROVIDER}_ENHANCED_MODEL_ID`: Enhanced model for complex synthesis (Blue Hat)
- `{PROVIDER}_STANDARD_MODEL_ID`: Standard model for individual hat processing
- `EXA_API_KEY`: Research capabilities (if using research agents)

**AI-Driven Model Strategy:**
- **Enhanced Models**: Used for Blue Hat (metacognitive) thinking - complex synthesis, integration
- **Standard Models**: Used for individual hat processing (White, Red, Black, Yellow, Green)
- **Intelligent Selection**: System automatically chooses appropriate model based on hat type and AI-assessed complexity
- **AI Analysis**: Replaces rule-based pattern matching with semantic understanding

**Processing Strategies (AI-Determined):**
1. **Single Hat**: Simple focused thinking (White Hat facts, Red Hat emotions, etc.)
2. **Double Hat**: Two-step sequences (e.g., Optimistic→Critical for idea evaluation)
3. **Triple Hat**: Core philosophical thinking (Factual→Creative→Synthesis)
4. **Full Sequence**: Complete Multi-Thinking methodology with Blue Hat orchestration

### Streamlined Module Architecture

**Core Framework:**
- `core/session.py`: SessionMemory for thought history (simplified, no Team dependency)
- `core/models.py`: ThoughtData validation and core data structures
- `core/types.py`: Type definitions and protocols
- `config/modernized_config.py`: Provider strategies with Enhanced/Standard model configuration
- `config/constants.py`: All system constants and configuration values

**Multi-Thinking Implementation:**
- `processors/multi_thinking_processor.py`: Main Multi-Thinking sequential processor
- `processors/multi_thinking_core.py`: Hat definitions, agent factory, core logic
- `routing/multi_thinking_router.py`: AI-powered intelligent routing based on thought complexity
- `routing/ai_complexity_analyzer.py`: AI-driven complexity and problem type analysis
- `routing/agno_workflow_router.py`: Agno Workflow integration layer
- `routing/complexity_types.py`: Core complexity analysis types and enums

**Service Layer:**
- `services/thought_processor_refactored.py`: Main thought processor with dependency injection
- `services/workflow_executor.py`: Multi-Thinking workflow execution
- `services/context_builder.py`: Context-aware prompt building
- `services/response_formatter.py`: Response formatting and extraction
- `services/response_processor.py`: Response processing utilities
- `services/processing_orchestrator.py`: Processing orchestration logic
- `services/retry_handler.py`: Error handling and retry mechanisms
- `services/server_core.py`: Server lifecycle and state management

**Infrastructure:**
- `infrastructure/logging_config.py`: Structured logging with rotation
- `infrastructure/persistent_memory.py`: Memory persistence capabilities
- `utils/utils.py`: Logging utilities and helper functions

### Architecture Characteristics

- **Clean Architecture**: Dependency injection, separation of concerns, service-based design
- **AI-Driven Intelligence**: Pure AI-based complexity analysis replacing rule-based systems
- **Multi-Thinking Focus**: Streamlined implementation without legacy multi-agent complexity
- **Model Optimization**: Smart model selection (Enhanced for synthesis, Standard for processing)
- **Modern Python**: Dataclasses, type hints, async/await, pattern matching
- **Environment-based config**: No config files, all via environment variables
- **Structured logging**: Rotation to `~/.sequential_thinking/logs/`

## Enhanced/Standard Model Configuration

**Naming Convention:**
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
- **AI-Driven Selection**: System intelligently chooses model based on hat type and AI-assessed complexity

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

## AI-Powered Complexity Analysis

**Key Innovation**: The system uses AI instead of rule-based pattern matching for complexity analysis:

- **AIComplexityAnalyzer**: Uses LLM to assess thought complexity, semantic depth, and problem characteristics
- **Problem Type Detection**: AI identifies primary problem type (FACTUAL, EMOTIONAL, CREATIVE, PHILOSOPHICAL, etc.)
- **Thinking Modes Recommendation**: AI suggests required thinking modes for optimal processing
- **Semantic Understanding**: Replaces keyword matching with contextual analysis across languages

**Benefits over Rule-Based Systems:**
- Better handling of nuanced, philosophical, or cross-cultural content
- Adaptive to new problem types without code changes
- Semantic understanding vs simple pattern matching
- Reduced maintenance overhead (no keyword lists to maintain)

## Development Notes

**No Test Suite**: The project currently has no test files - all tests were removed during recent cleanup.

**Recent Architecture Changes**:
- Removed legacy multi-agent systems (agents/, optimization/, analysis/ modules)
- Consolidated configuration (removed processing_constants.py redundancy)
- Streamlined to 8 core modules focused on AI-driven Multi-Thinking
- Added comprehensive service layer with orchestration, retry handling, and response processing
- Enhanced error handling and resilience patterns

**Code Quality**: Uses ruff for linting/formatting, mypy for type checking. Run `uv run ruff check . --fix && uv run ruff format . && uv run mypy .` before committing.

## Critical Implementation Patterns

### Core Architecture Principles

**Dependency Injection**: Manual constructor injection throughout - services compose via constructors, Protocol-based interfaces in `core/types.py`, immutable configs with `@dataclass(frozen=True, slots=True)`.

**Two-Tier Model System**:
- **Enhanced Model**: Synthesis agents (Blue Hat) for complex integration
- **Standard Model**: Individual thinking agents (Factual, Emotional, etc.)
- Configuration: `{PROVIDER}_ENHANCED_MODEL_ID` and `{PROVIDER}_STANDARD_MODEL_ID`

**Parallel Processing**: Non-synthesis agents use `asyncio.gather` for simultaneous execution. Only Synthesis agents run sequentially.

### Critical Gotchas

**Import Management**: Use `TYPE_CHECKING` imports and lazy loading to avoid circular dependencies:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module import Class
```

**Thread Safety**: Global state uses async locks:
```python
_processor_lock = asyncio.Lock()
async with _processor_lock:
    # Safe initialization
```

**ExaTools Integration**: Graceful degradation pattern:
```python
try:
    from agno.tools.exa import ExaTools
    EXA_AVAILABLE = bool(os.environ.get("EXA_API_KEY"))
except ImportError:
    EXA_AVAILABLE = False
```

**Multi-Thinking Direction Setup**: Six cognitive perspectives with specific timing:
- `FACTUAL`/`CRITICAL`/`OPTIMISTIC`/`CREATIVE`: 120-240s, with ExaTools
- `EMOTIONAL`: 30s quick reaction mode
- `SYNTHESIS`: 60s, Enhanced Model, no ExaTools

### Essential Development Rules

**Error Handling**: Use `ThoughtProcessingError` hierarchy, never catch bare `Exception`, include `ProcessingMetadata` for debugging.

**Performance**: Use `@dataclass(frozen=True, slots=True)` for frequent objects, `TokenEstimates` constants for cost calculation.

**Security**: All inputs sanitized via `html.escape()`, injection patterns checked via `SecurityConstants.INJECTION_PATTERNS`.

**Logging**: Application-scoped `sequential_thinking` namespace, 5MB rotation, use `MetricsLogger.log_metrics_block()` for performance tracking.

## Debugging Essentials

**Common Issues**:
- Circular imports → Use `TYPE_CHECKING` or dynamic imports
- Empty Agno content → Check `StepOutput.success` and `session_state`
- API key errors → Ensure real tokens (GitHub needs 15+ unique chars, no fake patterns)

**Debug Commands**:
- Test server: `npx @modelcontextprotocol/inspector uv run python src/mcp_server_mas_sequential_thinking/main.py`
- Monitor: `tail -f ~/.sequential_thinking/logs/sequential_thinking.log`
- Debug level: Set `LOG_LEVEL=DEBUG`