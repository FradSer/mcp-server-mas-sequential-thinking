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
- `LLM_PROVIDER`: Provider selection (deepseek, groq, openrouter, ollama, github, anthropic, claude-agent-sdk)
- `{PROVIDER}_API_KEY`: API keys (e.g., `DEEPSEEK_API_KEY`, `GITHUB_TOKEN`) - **Not required for claude-agent-sdk**
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
# Claude Agent SDK (uses local Claude Code - no API key needed!)
LLM_PROVIDER="claude-agent-sdk"
# No API key required - uses locally installed Claude Code
# Model IDs are informational - Claude Code uses its internal models
CLAUDE_AGENT_SDK_ENHANCED_MODEL_ID="claude-sonnet-4-5"  # Both synthesis and processing
CLAUDE_AGENT_SDK_STANDARD_MODEL_ID="claude-sonnet-4-5"

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

## Claude Agent SDK Advanced Features

The Claude Agent SDK provider supports comprehensive Agno integration with advanced features:

### Structured Outputs Support

**Response Format Configuration:**
The SDK supports structured outputs through Agno's `response_format` parameter:

```python
from pydantic import BaseModel

class ThinkingResult(BaseModel):
    """Structured thinking output"""
    analysis: str
    confidence: float
    key_insights: list[str]

# Agent automatically uses this schema
agent = Agent(
    model=claude_sdk_model,
    output_schema=ThinkingResult  # Converted to system prompt instructions
)
```

**Supported Formats:**
- **Pydantic BaseModel**: Full schema with validation (converted to JSON schema in system prompt)
- **JSON Mode**: `{"type": "json_object"}` for generic JSON output
- **Custom Schemas**: Dict-based schemas with properties

**How it Works:**
- `supports_native_structured_outputs()` returns `True` for Agno compatibility
- Schema is converted to detailed system prompt instructions
- Claude follows the schema structure via system prompt guidance
- Reliable structured output without native SDK support

### Tool Choice Strategies

**Fine-grained Tool Control:**
```python
agent = Agent(
    model=claude_sdk_model,
    tools=[ReasoningTools],
    tool_choice="required"  # Force tool usage
)

# Available strategies:
# - "none": Disable all tools
# - "auto": Model decides (default)
# - "required" / "any": Model must use tools
# - {"type": "tool", "name": "Think"}: Specific tool only
```

**Automatic Mapping:**
- `tool_choice="none"` → All tools added to `disallowed_tools`
- `tool_choice="required"` → Keep `allowed_tools` as-is
- Specific tool selection → Only that tool in `allowed_tools`, rest in `disallowed_tools`

### Session Continuation and User Context

**Automatic Session Tracking:**
```python
# Agno automatically provides session_id and user_id
agent.arun(
    "Continue our previous discussion",
    session_id="abc123",  # Extracted automatically
    user_id="user_456"     # Used for personalization
)

# SDK automatically:
# - Sets continue_conversation=True
# - Passes user context: {"id": "user_456"}
# - Maintains conversation continuity
```

**Benefits:**
- Multi-Thinking sequences maintain context across hat switches
- User-specific personalization
- Session history preservation

### Usage and Cost Tracking

**Automatic Metadata Extraction:**
Every response includes comprehensive usage data:

```python
response = await agent.arun("Analyze this problem")

# Available in response.provider_data:
{
    "usage": {
        "input_tokens": 1500,
        "output_tokens": 800,
        "cache_creation_input_tokens": 200,
        "cache_read_input_tokens": 1000,
        "stop_reason": "end_turn",
        "model_used": "claude-sonnet-4-5"
    },
    "run_metadata": {
        "session_id": "abc123",
        "user_id": "user_456",
        "run_id": "run_789"
    },
    "response_format_used": True,
    "tool_choice_used": "auto",
    "session_continuation": True
}
```

**Use Cases:**
- Cost analysis and budget tracking
- Performance monitoring
- Cache efficiency optimization
- Debugging and troubleshooting

### Advanced Configuration

**All Supported ClaudeAgentOptions:**
```python
from pathlib import Path

model = ClaudeAgentSDKModel(
    model_id="claude-sonnet-4-5",

    # Permission control
    permission_mode="bypassPermissions",  # default, acceptEdits, plan, bypassPermissions

    # File system access
    cwd="/path/to/project",
    add_dirs=[Path("/extra/context")],

    # MCP servers
    mcp_servers={
        "filesystem": {"path": "/path/to/mcp/config"}
    },

    # Environment
    env={"DEBUG": "1", "CUSTOM_VAR": "value"},

    # Event hooks
    hooks={
        "PreToolUse": [lambda ctx: print(f"Using {ctx.tool_name}")],
        "PostToolUse": [lambda ctx: print(f"Completed {ctx.tool_name}")]
    },

    # Permission callback
    can_use_tool=async_permission_checker
)
```

**Automatic Agno Integration:**
When used with Agno Agent:
- ✅ `response_format` → System prompt schema instructions
- ✅ `tool_choice` → Allowed/disallowed tools mapping
- ✅ `tools` → Automatic tool name mapping (ReasoningTools → Think)
- ✅ `tool_call_limit` → max_turns parameter
- ✅ `run_response.session_id` → continue_conversation
- ✅ `run_response.user_id` → user context
- ✅ Usage metadata → Extracted and tracked

### Multi-Thinking Integration Benefits

**For Multi-Thinking Workflow:**
1. **Structured Outputs**: Each hat can return structured JSON for reliable parsing
2. **Tool Control**: Fine-tune which hats use Think tool vs direct responses
3. **Session Context**: Maintain context across all 6 thinking agents
4. **Cost Tracking**: Monitor token usage per hat for optimization
5. **User Personalization**: Context-aware responses based on user history

**Example Multi-Thinking Flow:**
```
Factual Hat → (session_id: abc123)
   ↓ usage: 500 input, 200 output tokens
Emotional Hat → (session continues, uses cache)
   ↓ usage: 100 input (400 cached), 150 output
Critical Hat → (session continues)
   ↓ usage: 100 input (400 cached), 250 output
...
Synthesis Hat → (full context, structured output)
   ✓ Total cost tracked across all hats
   ✓ Cache efficiency: 80% cache hit rate
```

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

**Code Quality**: Uses ruff for linting/formatting, mypy for type checking. Run `uv run ruff check . --fix && uv run ruff format . && uv run mypy .` before committing.