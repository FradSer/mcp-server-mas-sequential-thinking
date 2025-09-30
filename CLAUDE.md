# CLAUDE.md

## Essential Commands

```bash
# Setup & Development
uv pip install -e ".[dev]"                              # Install dependencies
uv run mcp-server-mas-sequential-thinking               # Run server
uv run ruff check . --fix && uv run ruff format . && uv run mypy .  # Code quality

# Testing Framework
python run_tests.py                                     # Run all tests with coverage
python run_tests.py --unit --security                   # Run unit and security tests
uv run pytest tests/ -v                                 # Direct pytest execution

# Debugging & Monitoring
tail -f ~/.sequential_thinking/logs/sequential_thinking.log          # Live logs
npx @modelcontextprotocol/inspector uv run python src/mcp_server_mas_sequential_thinking/main.py  # Test server
```

## Project Overview

**AI-powered Multi-Thinking implementation** using Agno v2.0 framework via MCP. Processes thoughts through six cognitive perspectives (Factual, Emotional, Critical, Optimistic, Creative, Synthesis) with intelligent complexity analysis determining execution strategy (Single/Double/Triple/Full sequences).

**Core Flow:** External LLM → `sequentialthinking` tool → AI complexity analysis → Multi-Thinking workflow → Individual hat agents → Synthesis

## Configuration

**Required Environment Variables:**
```bash
LLM_PROVIDER=deepseek                                    # Provider (deepseek, groq, openrouter, ollama, github, anthropic)
DEEPSEEK_API_KEY=your_key                               # Provider API key
DEEPSEEK_ENHANCED_MODEL_ID=deepseek-chat                # Synthesis model
DEEPSEEK_STANDARD_MODEL_ID=deepseek-chat                # Individual hats model
EXA_API_KEY=your_key                                    # Optional: Research capabilities
```

**Model Strategy:**
- **Enhanced Model**: Blue Hat (synthesis) for complex integration
- **Standard Model**: Individual hats (White, Red, Black, Yellow, Green) for focused thinking
- **AI Selection**: System automatically chooses model based on hat type and complexity

## Key Architecture

**Entry Point:** `src/mcp_server_mas_sequential_thinking/main.py`

**Core Services:**
- `ThoughtProcessor`: Main orchestrator with dependency injection
- `WorkflowExecutor`: Manages Multi-Thinking workflow execution
- `AIComplexityAnalyzer`: AI-driven complexity assessment (replaces rule-based patterns)
- `MultiThinkingSequentialProcessor`: Executes chosen thinking sequence

**Processing Strategies (AI-Determined):**
1. **Single Hat**: Simple focused thinking
2. **Double Hat**: Two-step sequences (e.g., Optimistic→Critical)
3. **Triple Hat**: Core philosophical thinking (Factual→Creative→Synthesis)
4. **Full Sequence**: Complete Multi-Thinking with Blue Hat orchestration

## Critical Development Patterns

**Dependency Injection:** Manual constructor injection, Protocol-based interfaces in `core/types.py`

**Import Safety:** Avoid circular dependencies:
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from module import Class
```

**Thread Safety:** Global state uses async locks:
```python
_processor_lock = asyncio.Lock()
async with _processor_lock:
    # Safe initialization
```

**Error Handling:** Use `ThoughtProcessingError` hierarchy, include `ProcessingMetadata` for debugging

**Parallel Processing:** Non-synthesis agents use `asyncio.gather` for simultaneous execution

**Security & Rate Limiting:**
- Prompt injection protection with regex patterns and Shannon entropy
- Request size validation (50KB max)
- Token bucket algorithm for burst protection (30 req/min, 500 req/hour)
- Concurrent request limiting (5 max)
- Comprehensive input sanitization with HTML escaping

## Common Issues

- **Circular imports** → Use `TYPE_CHECKING` or dynamic imports
- **Empty Agno content** → Check `StepOutput.success` and `session_state`
- **API key errors** → Ensure real tokens (GitHub needs 15+ unique chars)
- **ExaTools import errors** → Optional dependency, graceful degradation built-in