# Sequential Thinking Multi-Agent System (MAS) ![](https://img.shields.io/badge/A%20FRAD%20PRODUCT-WIP-yellow)

[![smithery badge](https://smithery.ai/badge/@FradSer/mcp-server-mas-sequential-thinking)](https://smithery.ai/server/@FradSer/mcp-server-mas-sequential-thinking) [![Twitter Follow](https://img.shields.io/twitter/follow/FradSer?style=social)](https://twitter.com/FradSer) [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Framework](https://img.shields.io/badge/Framework-Agno-orange.svg)](https://github.com/cognitivecomputations/agno)

English | [简体中文](README.zh-CN.md)

This project implements an advanced sequential thinking process using a **Multi-Agent System (MAS)** built with the **Agno** framework and served via **MCP**. It represents a significant evolution from simpler state-tracking approaches by leveraging coordinated, specialized agents for deeper analysis and problem decomposition.

[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/fradser-mcp-server-mas-sequential-thinking-badge.png)](https://mseep.ai/app/fradser-mcp-server-mas-sequential-thinking)

## Overview

This server provides a sophisticated `sequentialthinking` tool designed for complex problem-solving. Unlike [its predecessor](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking), this version utilizes a true Multi-Agent System (MAS) architecture where:

- **A Coordinating Agent** (the `Team` object in `coordinate` mode) manages the workflow.
- **Specialized Agents** (Planner, Researcher, Analyzer, Critic, Synthesizer) handle specific sub-tasks based on their defined roles and expertise.
- Incoming thoughts are actively **processed, analyzed, and synthesized** by the agent team, not just logged.
- The system supports complex thought patterns, including **revisions** of previous steps and **branching** to explore alternative paths.
- Integration with external tools like **Exa** (via the Researcher agent) allows for dynamic information gathering.
- Robust **Pydantic** validation ensures data integrity for thought steps.
- Detailed **logging** tracks the process, including agent interactions (handled by the coordinator).

The goal is to achieve a higher quality of analysis and a more nuanced thinking process than possible with a single agent or simple state tracking by harnessing the power of specialized roles working collaboratively.

## Key Differences from Original Version (TypeScript)

This Python/Agno implementation marks a fundamental shift from the original TypeScript version:

| Feature/Aspect      | Python/Agno Version (Current)                                        | TypeScript Version (Original)                        |
| :------------------ | :------------------------------------------------------------------- | :--------------------------------------------------- |
| **Architecture**    | **Multi-Agent System (MAS)**; Active processing by a team of agents. | **Single Class State Tracker**; Simple logging/storing. |
| **Intelligence**    | **Distributed Agent Logic**; Embedded in specialized agents & Coordinator. | **External LLM Only**; No internal intelligence.     |
| **Processing**      | **Active Analysis & Synthesis**; Agents *act* on the thought.      | **Passive Logging**; Merely recorded the thought.    |
| **Frameworks**      | **Agno (MAS) + FastMCP (Server)**; Uses dedicated MAS library.     | **MCP SDK only**.                                    |
| **Coordination**    | **Explicit Team Coordination Logic** (`Team` in `coordinate` mode).  | **None**; No coordination concept.                   |
| **Validation**      | **Pydantic Schema Validation**; Robust data validation.            | **Basic Type Checks**; Less reliable.              |
| **External Tools**  | **Integrated (Exa via Researcher)**; Can perform research tasks.   | **None**.                                            |
| **Logging**         | **Structured Python Logging (File + Console)**; Configurable.      | **Console Logging with Chalk**; Basic.             |
| **Language & Ecosystem** | **Python**; Leverages Python AI/ML ecosystem.                    | **TypeScript/Node.js**.                              |

In essence, the system evolved from a passive thought *recorder* to an active thought *processor* powered by a collaborative team of AI agents.

## How it Works (Coordinate Mode)

1.  **Initiation:** An external LLM uses the `sequential-thinking-starter` prompt to define the problem and initiate the process.
2.  **Tool Call:** The LLM calls the `sequentialthinking` tool with the first (or subsequent) thought, structured according to the `ThoughtData` Pydantic model.
3.  **Validation & Logging:** The tool receives the call, validates the input using Pydantic, logs the incoming thought, and updates the history/branch state via `AppContext`.
4.  **Coordinator Invocation:** The core thought content (along with context about revisions/branches) is passed to the `SequentialThinkingTeam`'s `arun` method.
5.  **Coordinator Analysis & Delegation:** The `Team` (acting as Coordinator) analyzes the input thought, breaks it down into sub-tasks, and delegates these sub-tasks to the *most relevant* specialist agents (e.g., Analyzer for analysis tasks, Researcher for information needs).
6.  **Specialist Execution:** Delegated agents execute their specific sub-tasks using their instructions, models, and tools (like `ThinkingTools` or `ExaTools`).
7.  **Response Collection:** Specialists return their results to the Coordinator.
8.  **Synthesis & Guidance:** The Coordinator synthesizes the specialists' responses into a single, cohesive output. This output may include recommendations for revision or branching based on the specialists' findings (especially from the Critic and Analyzer). It also provides guidance for the LLM on formulating the next thought.
9.  **Return Value:** The tool returns a JSON string containing the Coordinator's synthesized response, status, and updated context (branches, history length).
10. **Iteration:** The calling LLM uses the Coordinator's response and guidance to formulate the next `sequentialthinking` tool call, potentially triggering revisions or branches as suggested.

## Token Consumption Warning

⚠️ **High Token Usage:** Due to the Multi-Agent System architecture, this tool consumes significantly **more tokens** than single-agent alternatives or the previous TypeScript version. Each `sequentialthinking` call invokes:

- The Coordinator agent (the `Team` itself).
- Multiple specialist agents (potentially Planner, Researcher, Analyzer, Critic, Synthesizer, depending on the Coordinator's delegation).

This parallel processing leads to substantially higher token usage (potentially 3-6x or more per thought step) compared to single-agent or state-tracking approaches. Budget and plan accordingly. This tool prioritizes **analysis depth and quality** over token efficiency.

## Prerequisites

- Python 3.10+
- Access to a compatible LLM API (configured for `agno`). The system currently supports:
    - **Groq:** Requires `GROQ_API_KEY`.
    - **DeepSeek:** Requires `DEEPSEEK_API_KEY`.
    - **OpenRouter:** Requires `OPENROUTER_API_KEY`.
    - **GitHub Models:** Requires `GITHUB_TOKEN`. Access to OpenAI GPT models through GitHub's API.
    - **Kimi K2:** Uses OpenRouter with `OPENROUTER_API_KEY`. Configure using `LLM_PROVIDER=kimi`.
    - Configure the desired provider using the `LLM_PROVIDER` environment variable (defaults to `deepseek`).
- Exa API Key (required only if using the Researcher agent's capabilities)
    - Set via the `EXA_API_KEY` environment variable.
- `uv` package manager (recommended) or `pip`.

## MCP Server Configuration (Client-Side)

This server runs as a standard executable script that communicates via stdio, as expected by MCP. The exact configuration method depends on your specific MCP client implementation. Consult your client's documentation for details on integrating external tool servers.

The `env` section within your MCP client configuration should include the API key for your chosen `LLM_PROVIDER`.

```json
{
  "mcpServers": {
      "mas-sequential-thinking": {
         "command": "uvx", // Or "python", "path/to/venv/bin/python" etc.
         "args": [
            "mcp-server-mas-sequential-thinking" // Or the path to your main script, e.g., "main.py"
         ],
         "env": {
            "LLM_PROVIDER": "deepseek", // "ollama", "groq", "openrouter", "github", "kimi"
            // "GROQ_API_KEY": "your_groq_api_key", // Only if LLM_PROVIDER="groq"
            "DEEPSEEK_API_KEY": "your_deepseek_api_key", // Default provider
            // "OPENROUTER_API_KEY": "your_openrouter_api_key", // Only if LLM_PROVIDER="openrouter" or "kimi"
            // "GITHUB_TOKEN": "your_github_token", // Only if LLM_PROVIDER="github"
            "LLM_BASE_URL": "your_base_url_if_needed", // Optional: If using a custom endpoint for DeepSeek
            "EXA_API_KEY": "your_exa_api_key" // Only if using Exa
         }
      }
   }
}
```

### GitHub Models Configuration Example

For using GitHub Models specifically:

```json
{
  "mcpServers": {
      "mas-sequential-thinking": {
         "command": "uvx",
         "args": [
            "mcp-server-mas-sequential-thinking"
         ],
         "env": {
            "LLM_PROVIDER": "github",
            "GITHUB_TOKEN": "ghp_your_github_personal_access_token",
            "GITHUB_ENHANCED_MODEL_ID": "openai/gpt-5", // Optional: Enhanced model for complex synthesis (default: openai/gpt-5)
            "GITHUB_STANDARD_MODEL_ID": "openai/gpt-5-min", // Optional: Standard model for individual processing (default: openai/gpt-5-min)
            "EXA_API_KEY": "your_exa_api_key" // Only if using Exa for research tasks
         }
      }
   }
}
```

## Installation & Setup

### Installing via Smithery

To install Sequential Thinking Multi-Agent System for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@FradSer/mcp-server-mas-sequential-thinking):

```bash
npx -y @smithery/cli install @FradSer/mcp-server-mas-sequential-thinking --client claude
```

### Manual Installation
1.  **Clone the repository:**
    ```bash
    git clone git@github.com:FradSer/mcp-server-mas-sequential-thinking.git
    cd mcp-server-mas-sequential-thinking
    ```

2.  **Set Environment Variables:**
    Create a `.env` file in the project root directory or export the variables directly into your environment:
    ```dotenv
    # --- LLM Configuration ---
    # Select the LLM provider: "deepseek" (default), "groq", "openrouter", "github", "ollama", or "kimi"
    LLM_PROVIDER="deepseek"

    # Provide the API key for the chosen provider:
    # GROQ_API_KEY="your_groq_api_key"
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    # OPENROUTER_API_KEY="your_openrouter_api_key" # Required for both "openrouter" and "kimi" providers
    # GITHUB_TOKEN="ghp_your_github_personal_access_token" # Required for LLM_PROVIDER="github"
    # Note: Ollama requires no API key but needs local installation
    # Note: Kimi K2 uses OpenRouter API, so OPENROUTER_API_KEY is required for LLM_PROVIDER="kimi"
    # Note: GitHub Models requires a GitHub Personal Access Token with appropriate scopes

    # Optional: Base URL override (e.g., for custom endpoints)
    # LLM_BASE_URL="your_base_url_if_needed"

    # Optional: Specify different models for Enhanced (Complex Synthesis) and Standard (Individual Processing)
    # Defaults are set within the code based on the provider if these are not set.
    # Example for Groq:
    # GROQ_ENHANCED_MODEL_ID="deepseek-r1-distill-llama-70b"  # For complex synthesis (Blue Hat)
    # GROQ_STANDARD_MODEL_ID="qwen/qwen3-32b"  # For individual hat processing
    # Example for DeepSeek:
    # DEEPSEEK_ENHANCED_MODEL_ID="deepseek-chat"  # For complex synthesis
    # DEEPSEEK_STANDARD_MODEL_ID="deepseek-chat"  # For individual processing
    # Example for GitHub Models:
    # GITHUB_ENHANCED_MODEL_ID="openai/gpt-5"  # Enhanced model for synthesis
    # GITHUB_STANDARD_MODEL_ID="openai/gpt-5-min"  # Standard model for processing
    # Example for OpenRouter:
    # OPENROUTER_ENHANCED_MODEL_ID="deepseek/deepseek-r1"  # Example, adjust as needed
    # OPENROUTER_STANDARD_MODEL_ID="deepseek/deepseek-chat"  # Example, adjust as needed
    # Example for Kimi K2:
    # KIMI_ENHANCED_MODEL_ID="moonshotai/kimi-k2"  # Uses Moonshot's Kimi K2 via OpenRouter
    # KIMI_STANDARD_MODEL_ID="moonshotai/kimi-k2"  # Same model for both, optimized for tool use

    # --- External Tools ---
    # Required ONLY if the Researcher agent is used and needs Exa
    EXA_API_KEY="your_exa_api_key"
    ```

    **Note on Model Selection:**
    - The `ENHANCED_MODEL_ID` is used for complex synthesis tasks (like Blue Hat thinking). This role benefits from strong reasoning, synthesis, and integration capabilities. Consider using a more powerful model (e.g., `deepseek-chat`, `claude-3-opus`, `gpt-5`) here, potentially balancing capability with cost/speed.
    - The `STANDARD_MODEL_ID` is used for individual hat processing (White, Red, Black, Yellow, Green hats). These handle focused thinking perspectives. A faster or more cost-effective model (e.g., `deepseek-chat`, `claude-3-sonnet`, `qwen3-32b`) might be suitable, depending on task complexity and budget/performance needs.
    - Defaults are provided in the code (e.g., in the configuration files) if these environment variables are not set. Experimentation is encouraged to find the optimal balance for your use case.

    **GitHub Models Setup and Available Models:**
    
    GitHub Models provides access to OpenAI's GPT models through GitHub's API infrastructure. To use GitHub Models:
    
    1. **Get a GitHub Personal Access Token:**
       - Go to GitHub Settings > Developer settings > Personal access tokens > Tokens (classic)
       - Generate a new token with appropriate scopes (typically `repo` and `read:user`)
       - The token format will be `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`
    
    2. **Available Models and Use Cases:**
       - **`openai/gpt-5`** (Default for Enhanced Model):
         - Latest GPT-5 model
         - Best for complex reasoning, coordination, and synthesis tasks
         - Higher token cost but superior performance for synthesis and integration
         - Excellent function calling capabilities

       - **`openai/gpt-5-min`** (Default for Standard Model):
         - Lightweight version of GPT-5
         - Optimized for speed and cost-efficiency
         - Ideal for focused individual hat processing (White, Red, Black, Yellow, Green)
         - Good balance of performance and cost for high-volume operations
       
       - **`openai/gpt-4o`**:
         - Previous generation GPT-4 Omni model
         - Strong reasoning capabilities but slower than gpt-5
         - Can be used for either role depending on requirements
       
       - **`openai/gpt-3.5-turbo`**:
         - Most cost-effective option
         - Suitable for simpler tasks or budget-conscious deployments
         - May have limitations with complex reasoning tasks
    
    3. **Recommended Configurations:**
       - **High Performance:** `GITHUB_ENHANCED_MODEL_ID="openai/gpt-5"`, `GITHUB_STANDARD_MODEL_ID="openai/gpt-5-min"`
       - **Balanced:** `GITHUB_ENHANCED_MODEL_ID="openai/gpt-5"`, `GITHUB_STANDARD_MODEL_ID="openai/gpt-3.5-turbo"`
       - **Budget-Conscious:** `GITHUB_ENHANCED_MODEL_ID="openai/gpt-5-min"`, `GITHUB_STANDARD_MODEL_ID="openai/gpt-3.5-turbo"`
    
    4. **Token Usage Considerations with GitHub Models:**
       - GitHub Models usage counts toward GitHub's rate limits and pricing
       - Monitor your usage through GitHub's billing dashboard
       - Consider the Multi-Agent System's high token consumption when budgeting
       - The default configuration (openai/gpt-5 + openai/gpt-5-min) provides the best balance of performance and cost for Six Thinking Hats methodology

3.  **Install Dependencies:**
    It's highly recommended to use a virtual environment.

    - **Using `uv` (Recommended):**
        ```bash
        # Install uv if you don't have it:
        # curl -LsSf https://astral.sh/uv/install.sh | sh
        # source $HOME/.cargo/env # Or restart your shell

        # Create and activate a virtual environment (optional but recommended)
        python -m venv .venv
        source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`

        # Install dependencies
        uv pip install -r requirements.txt
        # Or if a pyproject.toml exists with dependencies defined:
        # uv pip install .
        ```
    - **Using `pip`:**
        ```bash
        # Create and activate a virtual environment (optional but recommended)
        python -m venv .venv
        source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`

        # Install dependencies
        pip install -r requirements.txt
        # Or if a pyproject.toml exists with dependencies defined:
        # pip install .
        ```

## Usage

Ensure your environment variables are set and the virtual environment (if used) is active.

Run the server. Choose one of the following methods:

1.  **Using `uv run` (Recommended):**
    ```bash
    uv --directory /path/to/mcp-server-mas-sequential-thinking run mcp-server-mas-sequential-thinking
    ```
2.  **Directly using Python:**

    ```bash
    python main.py
    ```

The server will start and listen for requests via stdio, making the `sequentialthinking` tool available to compatible MCP clients configured to use it.

### `sequentialthinking` Tool Parameters

The tool expects arguments matching the `ThoughtData` Pydantic model:

```python
# Simplified representation from src/models.py
class ThoughtData(BaseModel):
    thought: str                 # Content of the current thought/step
    thoughtNumber: int           # Sequence number (>=1)
    totalThoughts: int           # Estimated total steps (>=1, suggest >=5)
    nextThoughtNeeded: bool      # Is another step required after this?
    isRevision: bool = False     # Is this revising a previous thought?
    revisesThought: Optional[int] = None # If isRevision, which thought number?
    branchFromThought: Optional[int] = None # If branching, from which thought?
    branchId: Optional[str] = None # Unique ID for the new branch being created
    needsMoreThoughts: bool = False # Signal if estimate is too low before last step
```

### Interacting with the Tool (Conceptual Example)

An LLM would interact with this tool iteratively:

1.  **LLM:** Uses a starter prompt (like `sequential-thinking-starter`) with the problem definition.
2.  **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 1`, the initial `thought` (e.g., "Plan the analysis..."), an estimated `totalThoughts`, and `nextThoughtNeeded: True`.
3.  **Server:** MAS processes the thought. The Coordinator synthesizes responses from specialists and provides guidance (e.g., "Analysis plan complete. Suggest researching X next. No revisions recommended yet.").
4.  **LLM:** Receives the JSON response containing `coordinatorResponse`.
5.  **LLM:** Formulates the next thought based on the `coordinatorResponse` (e.g., "Research X using available tools...").
6.  **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 2`, the new `thought`, potentially updated `totalThoughts`, `nextThoughtNeeded: True`.
7.  **Server:** MAS processes. The Coordinator synthesizes (e.g., "Research complete. Findings suggest a flaw in thought #1's assumption. RECOMMENDATION: Revise thought #1...").
8.  **LLM:** Receives the response, notes the recommendation.
9.  **LLM:** Formulates a revision thought.
10. **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 3`, the revision `thought`, `isRevision: True`, `revisesThought: 1`, `nextThoughtNeeded: True`.
11. **... and so on, potentially branching or extending the process as needed.**

### Tool Response Format

The tool returns a JSON string containing:

```json
{
  "processedThoughtNumber": int,          // The thought number that was just processed
  "estimatedTotalThoughts": int,          // The current estimate of total thoughts
  "nextThoughtNeeded": bool,              // Whether the process indicates more steps are needed
  "coordinatorResponse": "...",           // Synthesized output from the agent team, including analysis, findings, and guidance for the next step.
  "branches": ["main", "branch-id-1"],  // List of active branch IDs
  "thoughtHistoryLength": int,          // Total number of thoughts processed so far (across all branches)
  "branchDetails": {
    "currentBranchId": "main",            // The ID of the branch the processed thought belongs to
    "branchOriginThought": null | int,    // The thought number where the current branch diverged (null for 'main')
    "allBranches": {                      // Count of thoughts in each active branch
      "main": 5,
      "branch-id-1": 2
     }
  },
  "isRevision": bool,                     // Was the processed thought a revision?
  "revisesThought": null | int,           // Which thought number was revised (if isRevision is true)
  "isBranch": bool,                       // Did this thought start a new branch?
  "status": "success | validation_error | failed", // Outcome status
  "error": null | "Error message..."     // Error details if status is not 'success'
}
```

## Logging

- Logs are written to `~/.sequential_thinking/logs/sequential_thinking.log` by default. (Configuration might be adjustable in the logging setup code).
- Uses Python's standard `logging` module.
- Includes a rotating file handler (e.g., 10MB limit, 5 backups) and a console handler (typically INFO level).
- Logs include timestamps, levels, logger names, and messages, including structured representations of thoughts being processed.

## Development

1.  **Clone the repository:** (As in Installation)
    ```bash
    git clone git@github.com:FradSer/mcp-server-mas-sequential-thinking.git
    cd mcp-server-mas-sequential-thinking
    ```
2.  **Set up Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\\Scripts\\activate`
    ```
3.  **Install Dependencies (including dev):**
    Ensure your `requirements-dev.txt` or `pyproject.toml` specifies development tools (like `pytest`, `ruff`, `black`, `mypy`).
    ```bash
    # Using uv
    uv pip install -r requirements.txt
    uv pip install -r requirements-dev.txt # Or install extras if defined in pyproject.toml: uv pip install -e ".[dev]"

    # Using pip
    pip install -r requirements.txt
    pip install -r requirements-dev.txt # Or install extras if defined in pyproject.toml: pip install -e ".[dev]"
    ```
4.  **Run Checks:**
    Execute linters, formatters, and tests (adjust commands based on your project setup).
    ```bash
    # Example commands (replace with actual commands used in the project)
    ruff check . --fix
    black .
    mypy .
    pytest
    ```

5.  **Testing the Application:**
6.  Testing: Test your MCP server locally before deploying using MCP Inspector. Please ensure your Dockerfile builds locally first before deploying.
7.  ```bash
    npx @modelcontextprotocol/inspector uv run main.py
    ```
    Open the url http://127.0.0.1:6274/, then click on the "Run" button to test your MCP server locally. Then click on the "Tools" button to see the tools that are available in the MCP server, and verify that the tool "sequentialthinking" is available.
    Copy "I need to test mcp-server-mas-sequential-thinking, I just hope mcp inspector will help me" in the "thought" field, "1" in the "thoughtNumber"and in the "totalThoughts" field, and check the "nextThoughtNeeded" checkbox. Then click on the "Run Tool" button to test your MCP server locally.
    The model should return a response with a new thought, a new thought number, a new total thoughts, and Tool Result: Success. If you see this, your MCP server is working correctly.
8.  **Contribution:**
    (Consider adding contribution guidelines: branching strategy, pull request process, code style).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## License

MIT
