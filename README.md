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
    - **DeepSeek:** Requires `DEEPSEEK_API_KEY` (default).
    - **Groq:** Requires `GROQ_API_KEY`.
    - **OpenRouter:** Requires `OPENROUTER_API_KEY`.
    - **GitHub Models:** Requires `GITHUB_TOKEN`.
    - **Anthropic:** Requires `ANTHROPIC_API_KEY`.
    - **Ollama:** No API key needed, but requires a local Ollama installation.
    - Configure the desired provider using the `LLM_PROVIDER` environment variable.
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
            "LLM_PROVIDER": "deepseek", // "groq", "openrouter", "github", "anthropic", "ollama"
            "DEEPSEEK_API_KEY": "your_deepseek_api_key", // Default provider
            // "GROQ_API_KEY": "your_groq_api_key",
            // "OPENROUTER_API_KEY": "your_openrouter_api_key",
            // "GITHUB_TOKEN": "your_github_token",
            // "ANTHROPIC_API_KEY": "your_anthropic_api_key",
            "LLM_BASE_URL": "your_base_url_if_needed", // Optional: For custom endpoints
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
    # Select the LLM provider: "deepseek" (default), "groq", "openrouter", "github", "anthropic", or "ollama"
    LLM_PROVIDER="deepseek"

    # Provide the API key for the chosen provider:
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    # GROQ_API_KEY="your_groq_api_key"
    # OPENROUTER_API_KEY="your_openrouter_api_key"
    # GITHUB_TOKEN="ghp_your_github_personal_access_token"
    # ANTHROPIC_API_KEY="your_anthropic_api_key"
    # Note: Ollama requires no API key but needs a local Ollama installation.

    # Optional: Base URL override (e.g., for custom endpoints)
    # LLM_BASE_URL="your_base_url_if_needed"

    # Optional: Specify different models for Enhanced (Complex Synthesis) and Standard (Individual Processing)
    # Defaults are set within the code based on the provider if these are not set.
    # Example for Groq:
    # GROQ_ENHANCED_MODEL_ID="deepseek-r1-distill-llama-70b"  # For complex synthesis
    # GROQ_STANDARD_MODEL_ID="qwen/qwen3-32b"  # For individual processing
    # Example for DeepSeek:
    # DEEPSEEK_ENHANCED_MODEL_ID="deepseek-chat"  # For complex synthesis
    # DEEPSEEK_STANDARD_MODEL_ID="deepseek-chat"  # For individual processing
    # Example for GitHub Models:
    # GITHUB_ENHANCED_MODEL_ID="openai/gpt-5"  # Enhanced model for synthesis
    # GITHUB_STANDARD_MODEL_ID="openai/gpt-5-min"  # Standard model for processing
    # Example for OpenRouter:
    # OPENROUTER_ENHANCED_MODEL_ID="deepseek/deepseek-r1"  # Example, adjust as needed
    # OPENROUTER_STANDARD_MODEL_ID="deepseek/deepseek-chat"  # Example, adjust as needed
    # Example for Anthropic:
    # ANTHROPIC_ENHANCED_MODEL_ID="claude-3-5-sonnet-20241022"
    # ANTHROPIC_STANDARD_MODEL_ID="claude-3-5-haiku-20241022"

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
       - The default configuration (openai/gpt-5 + openai/gpt-5-min) provides the best balance of performance and cost for Multi-Thinking methodology

3.  **Install Dependencies:**
    It's highly recommended to use a virtual environment.

    - **Using `uv` (Recommended):**
        ```bash
        # Install uv if you don't have it:
        # curl -LsSf https://astral.sh/uv/install.sh | sh
        # source $HOME/.cargo/env # Or restart your shell

        # Create and activate a virtual environment
        python -m venv .venv
        source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

        # Install dependencies from pyproject.toml
        uv pip install .
        ```
    - **Using `pip`:**
        ```bash
        # Create and activate a virtual environment
        python -m venv .venv
        source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

        # Install dependencies from pyproject.toml
        pip install .
        ```

## Usage

Ensure your environment variables are set and the virtual environment (if used) is active.

Run the server. Choose one of the following methods:

1.  **Using the installed script (Recommended):**
    After installation, you can run the server directly.
    ```bash
    mcp-server-mas-sequential-thinking
    ```

2.  **Using `uv run`:**
    This is useful for running without activating the virtual environment.
    ```bash
    uv run mcp-server-mas-sequential-thinking
    ```

3.  **Directly using Python:**
    ```bash
    python src/mcp_server_mas_sequential_thinking/main.py
    ```

The server will start and listen for requests via stdio, making the `sequentialthinking` tool available to compatible MCP clients configured to use it.

### `sequentialthinking` Tool Parameters

## Development

1.  **Clone the repository:** (As in Installation)
    ```bash
    git clone git@github.com:FradSer/mcp-server-mas-sequential-thinking.git
    cd mcp-server-mas-sequential-thinking
    ```
2.  **Set up Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`
    ```
3.  **Install Dependencies (including dev):**
    Install the project in editable mode with development dependencies.
    ```bash
    # Using uv
    uv pip install -e .[dev]

    # Using pip
    pip install -e .[dev]
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

5.  **Contribution:**
    (Consider adding contribution guidelines: branching strategy, pull request process, code style).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## License

MIT
