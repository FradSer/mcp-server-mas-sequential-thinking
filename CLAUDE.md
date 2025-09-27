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

### Dependency Injection Architecture

The system uses **manual dependency injection** throughout, following clean architecture principles:

- **Constructor injection**: All services receive dependencies via constructors
- **Service composition**: Complex services compose simpler ones (e.g., `ThoughtProcessor` composes `ContextBuilder`, `WorkflowExecutor`, `ResponseFormatter`)
- **Protocol-based design**: Uses `Protocol` classes in `core/types.py` for flexible interfaces
- **Immutable configurations**: Uses `@dataclass(frozen=True, slots=True)` for configuration objects

**Example Pattern**:
```python
class ThoughtProcessor:
    def __init__(self, session: SessionMemory) -> None:
        self._context_builder = ContextBuilder(session)
        self._workflow_executor = WorkflowExecutor(session)
        self._response_formatter = ResponseFormatter()
```

### Error Handling Strategy

**Hierarchical Exception Design**: Custom exceptions inherit from base types in `core/types.py`:
- `ThoughtProcessingError` (base for processing errors)
  - `RoutingDecisionError`, `CostOptimizationError`, `PersistentStorageError`
- `ConfigurationError` (base for config errors)
  - `ModelConfigurationError`

**Retry Pattern**: `RetryHandler` uses configurable exponential backoff:
- Default: 3 attempts with 0.5s base sleep
- Comprehensive logging of retry attempts and failures
- Context-aware error reporting with operation metadata

### Multi-Model Intelligence

**Critical Pattern**: The system uses **two distinct model tiers**:
- **Enhanced Model**: For Synthesis agents (Blue Hat) - complex integration tasks
- **Standard Model**: For individual thinking agents (White, Red, Black, Yellow, Green)

**Provider Strategy Pattern**: Each provider implements both enhanced/standard model selection:
```python
{PROVIDER}_ENHANCED_MODEL_ID="model-for-synthesis"
{PROVIDER}_STANDARD_MODEL_ID="model-for-processing"
```

### Async/Parallel Processing Architecture

**Critical Performance Pattern**: Non-synthesis agents run in **parallel** using `asyncio.gather`:
- Double/Triple sequences: All non-synthesis agents execute simultaneously
- Full sequences: 3-step process (Initial Synthesis → Parallel Agents → Final Synthesis)
- Performance optimization: Reduces execution time while maintaining analysis quality

### Configuration Management

**Environment-Only Pattern**: No config files - all configuration via environment variables:
- Provider detection: `LLM_PROVIDER` determines which provider strategy to use
- Model selection: `{PROVIDER}_ENHANCED_MODEL_ID` and `{PROVIDER}_STANDARD_MODEL_ID`
- Feature flags: `ENABLE_ADAPTIVE_ROUTING`, `ENABLE_MULTI_THINKING`

**GitHub Token Validation**: Special validation for GitHub tokens with entropy checks and fake pattern detection.

### Memory and State Management

**Immutable Data Patterns**: All core data structures use Pydantic models with immutability:
- `ThoughtData`: Validated input with relationship constraints
- `ServerConfig`: Frozen dataclass for server configuration
- `ThinkingTimingConfig`: Frozen configuration for thinking direction timing

**Session State**: `SessionMemory` manages thought history and branching without team dependencies (simplified from legacy architecture).

### Import and Circular Dependency Management

**Critical Pattern**: The codebase uses **lazy imports** and `TYPE_CHECKING` to break circular dependencies:

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agno.agent import Agent
    from agno.models.base import Model
```

**Conditional Tool Imports**: ExaTools import is wrapped in try/catch with graceful degradation:
```python
try:
    from agno.tools.exa import ExaTools
    EXA_AVAILABLE = bool(os.environ.get("EXA_API_KEY"))
except ImportError:
    ExaTools = None
    EXA_AVAILABLE = False
```

### Logging Architecture

**Streamlined Approach**: Replaced complex 985-line logging with focused implementation:
- **File rotation**: 5MB files, 3 backups in `~/.sequential_thinking/logs/`
- **Environment-based**: Console logging only in non-production
- **Application-scoped**: Uses `sequential_thinking` logger namespace to avoid conflicts
- **Performance focus**: `MetricsLogger` mixin provides consistent performance logging patterns

### Multi-Thinking Direction Architecture

**Enum-Driven Design**: `ThinkingDirection` enum defines six distinct cognitive perspectives:
- `FACTUAL`: Facts and data (120s, with ExaTools)
- `EMOTIONAL`: Intuition (30s, quick reaction mode)
- `CRITICAL`: Risk assessment (120s, with ExaTools)
- `OPTIMISTIC`: Opportunities (120s, with ExaTools)
- `CREATIVE`: Innovation (240s, with ExaTools)
- `SYNTHESIS`: Integration (60s, Enhanced Model, no ExaTools)

**Timing Configuration**: `ThinkingTimingConfig` enforces different processing times per direction with min/max bounds.

## Parallel Processing Architecture

**Critical Implementation Detail**: The system uses `asyncio.gather` for parallel execution of thinking agents:

- **Non-Synthesis Agents**: Run in parallel for maximum efficiency (Factual, Emotional, Critical, Optimistic, Creative)
- **Synthesis Agent**: Runs sequentially after parallel agents complete, uses Enhanced Model for integration
- **Key Methods**:
  - `_process_double_direction_sequence()`: Parallel execution for 2-agent strategies
  - `_process_triple_direction_sequence()`: Parallel execution for 3-agent strategies
  - `_process_full_direction_sequence()`: 3-step process (Initial Synthesis → Parallel Agents → Final Synthesis)

**Performance Impact**: Parallel processing reduces total execution time while maintaining comprehensive analysis quality.

## Error Handling & Logging

**Structured Logging System**:
- **MetricsLogger**: Located in `infrastructure/logging_config.py`, provides `log_metrics_block()` and `log_separator()` methods
- **Log Location**: `~/.sequential_thinking/logs/sequential_thinking.log` with 5MB rotation, 3 backups
- **Performance Logging**: Uses `log_performance_metric()` for timing operations
- **Error Recovery**: Single-agent fallback when team processing fails

**Common Import Fix**: If you see `cannot import name 'MetricsLogger'`, ensure `infrastructure/__init__.py` exports it properly.

## Provider Configuration Patterns

**Model Selection Strategy**:
```bash
# Provider-specific model configuration
{PROVIDER}_ENHANCED_MODEL_ID="model-for-synthesis"    # Blue Hat/Synthesis Agent
{PROVIDER}_STANDARD_MODEL_ID="model-for-processing"   # Individual thinking agents
```

**Current Provider Models**:
- **Groq**: Enhanced=`openai/gpt-oss-120b`, Standard=`openai/gpt-oss-20b`
- **DeepSeek**: Both use `deepseek-chat`
- **Anthropic**: Enhanced=`claude-3-5-sonnet-20241022`, Standard=`claude-3-5-haiku-20241022`
- **GitHub Models**: Enhanced=`openai/gpt-5`, Standard=`openai/gpt-5-min`
- **OpenRouter**: Enhanced=`deepseek/deepseek-chat-v3-0324`, Standard=`deepseek/deepseek-r1`
- **Ollama**: Enhanced=`devstral:24b`, Standard=`devstral:24b`

## Configuration Management

**Environment File Template**: Use `.env.example` as a starting point for your configuration. The file includes comprehensive examples for all supported providers and advanced features including:

- **Adaptive Routing**: Automatic single vs multi-agent selection based on complexity
- **Budget Management**: Daily, monthly, and per-thought cost limits
- **Performance Monitoring**: Real-time performance tracking with baselines
- **Smart Logging**: Intelligent logging with adaptive verbosity levels
- **Memory Management**: Persistent storage with automatic pruning policies

## Development Gotchas and Critical Patterns

### Module Import Patterns
- **Always use lazy imports** in type annotations to avoid circular dependencies
- **Use `__slots__`** in performance-critical classes like `ThoughtProcessor`
- **Protocol-based typing**: Prefer `Protocol` classes over concrete inheritance for better flexibility

### Error Handling Requirements
- **Never catch bare `Exception`**: Always catch specific exceptions or use `ThoughtProcessingError` hierarchy
- **Include metadata**: Use `ProcessingMetadata` in exceptions for debugging context
- **Retry-aware design**: Operations should be idempotent to support retry logic

### Configuration Anti-Patterns
- **No hardcoded paths**: Always use `Path.home()` and environment variables
- **Provider validation**: All provider configurations must validate API keys before use
- **Model fallbacks**: Enhanced models must have Standard model fallbacks configured

### Performance Considerations
- **Parallel vs Sequential**: Only Synthesis agents run sequentially; all others use `asyncio.gather`
- **Token estimation**: Use `TokenEstimates` constants for cost calculation
- **Memory efficiency**: Use `@dataclass(frozen=True, slots=True)` for frequent objects

### Multi-Thinking Sequence Rules
- **Timing enforcement**: Each `ThinkingDirection` has strict min/max time bounds
- **Tool availability**: Only Factual, Critical, Optimistic, and Creative agents get ExaTools
- **Model selection**: Synthesis agents MUST use Enhanced models for quality integration
- **Parallel execution**: Non-synthesis agents in Double/Triple/Full sequences run in parallel for performance

### Logging Best Practices
- **Use structured logging**: Always include operation context in log messages
- **Performance metrics**: Log execution time and efficiency scores for optimization
- **Error context**: Include thought metadata and processing state in error logs
- **Log rotation**: Respect 5MB file limits and 3-backup rotation policy