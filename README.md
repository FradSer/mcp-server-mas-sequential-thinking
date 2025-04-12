# Sequential Thinking Multi-Agent System (MAS) ![](https://img.shields.io/badge/A%20FRAD%20PRODUCT-WIP-yellow)

[![smithery badge](https://smithery.ai/badge/@FradSer/mcp-server-mas-sequential-thinking)](https://smithery.ai/server/@FradSer/mcp-server-mas-sequential-thinking)
[![Twitter Follow](https://img.shields.io/twitter/follow/FradSer?style=social)](https://twitter.com/FradSer) [![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/) [![Framework](https://img.shields.io/badge/Framework-Agno-orange.svg)](https://github.com/cognitivecomputations/agno) 

English | [简体中文](README.zh-CN.md)

This project implements an advanced sequential thinking process using a **Multi-Agent System (MAS)** built with the **Agno** framework and served via **MCP**. It represents a significant evolution from simpler state-tracking approaches, leveraging coordinated specialized agents for deeper analysis and problem decomposition.

## Overview

This server provides a sophisticated `sequentialthinking` tool designed for complex problem-solving. Unlike [its predecessor](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking), this version utilizes a true Multi-Agent System (MAS) architecture where:

* **A Coordinating Agent** (the `Team` object in `coordinate` mode) manages the workflow.
* **Specialized Agents** (Planner, Researcher, Analyzer, Critic, Synthesizer) handle specific sub-tasks based on their defined roles and expertise.
* Incoming thoughts are actively **processed, analyzed, and synthesized** by the agent team, not just logged.
* The system supports complex thought patterns including **revisions** of previous steps and **branching** to explore alternative paths.
* Integration with external tools like **Exa** (via the Researcher agent) allows for dynamic information gathering.
* Robust **Pydantic** validation ensures data integrity for thought steps.
* Detailed **logging** tracks the process, including agent interactions (handled by the coordinator).

The goal is to achieve a higher quality of analysis and a more nuanced thinking process than possible with a single agent or simple state tracking, by harnessing the power of specialized roles working collaboratively.

## Key Differences from Original Version (TypeScript)

This Python/Agno implementation marks a fundamental shift from the original TypeScript version:

| Feature/Aspect      | Python/Agno Version (Current)                                        | TypeScript Version (Original)                        |
| :------------------ | :------------------------------------------------------------------- | :--------------------------------------------------- |
| **Architecture** | **Multi-Agent System (MAS)**; Active processing by a team of agents. | **Single Class State Tracker**; Simple logging/storing. |
| **Intelligence** | **Distributed Agent Logic**; Embedded in specialized agents & Coordinator. | **External LLM Only**; No internal intelligence.     |
| **Processing** | **Active Analysis & Synthesis**; Agents *act* on the thought.      | **Passive Logging**; Merely recorded the thought.    |
| **Frameworks** | **Agno (MAS) + FastMCP (Server)**; Uses dedicated MAS library.     | **MCP SDK only**.                                    |
| **Coordination** | **Explicit Team Coordination Logic** (`Team` in `coordinate` mode).  | **None**; No coordination concept.                   |
| **Validation** | **Pydantic Schema Validation**; Robust data validation.            | **Basic Type Checks**; Less reliable.              |
| **External Tools** | **Integrated (Exa via Researcher)**; Can perform research tasks.   | **None**.                                            |
| **Logging** | **Structured Python Logging (File + Console)**; Configurable.      | **Console Logging with Chalk**; Basic.             |
| **Language & Ecosystem** | **Python**; Leverages Python AI/ML ecosystem.                    | **TypeScript/Node.js**.                              |

In essence, the system evolved from a passive thought *recorder* to an active thought *processor* powered by a collaborative team of AI agents.

## How it Works (Coordinate Mode)

1.  **Initiation:** An external LLM uses the `sequential-thinking-starter` prompt to define the problem and initiate the process.
2.  **Tool Call:** The LLM calls the `sequentialthinking` tool with the first (or subsequent) thought, structured according to the `ThoughtData` model.
3.  **Validation & Logging:** The tool receives the call, validates the input using Pydantic, logs the incoming thought, and updates the history/branch state via `AppContext`.
4.  **Coordinator Invocation:** The core thought content (with context about revisions/branches) is passed to the `SequentialThinkingTeam`'s `arun` method.
5.  **Coordinator Analysis & Delegation:** The `Team` (acting as Coordinator) analyzes the input thought, breaks it into sub-tasks, and delegates these sub-tasks to the *most relevant* specialist agents (e.g., Analyzer for analysis tasks, Researcher for information needs).
6.  **Specialist Execution:** Delegated agents execute their specific sub-tasks using their instructions, models, and tools (like `ThinkingTools` or `ExaTools`).
7.  **Response Collection:** Specialists return their results to the Coordinator.
8.  **Synthesis & Guidance:** The Coordinator synthesizes the specialists' responses into a single, cohesive output. It may include recommendations for revision or branching based on the specialists' findings (especially the Critic and Analyzer). It also adds guidance for the LLM on formulating the next thought.
9.  **Return Value:** The tool returns a JSON string containing the Coordinator's synthesized response, status, and updated context (branches, history length).
10. **Iteration:** The calling LLM uses the Coordinator's response and guidance to formulate the next `sequentialthinking` tool call, potentially triggering revisions or branches as suggested.

## Token Consumption Warning

⚠️ **High Token Usage:** Due to the Multi-Agent System architecture, this tool consumes significantly **more tokens** than single-agent alternatives or the previous TypeScript version. Each `sequentialthinking` call invokes:
    * The Coordinator agent (the `Team` itself).
    * Multiple specialist agents (potentially Planner, Researcher, Analyzer, Critic, Synthesizer, depending on the Coordinator's delegation).

This parallel processing leads to substantially higher token usage (potentially 3-6x or more per thought step) compared to single-agent or state-tracking approaches. Budget and plan accordingly. This tool prioritizes **analysis depth and quality** over token efficiency.

## Prerequisites

* Python 3.10+
* Access to a compatible LLM API (configured for `agno`). The system now supports:
    * **Groq:** Requires `GROQ_API_KEY`.
    * **DeepSeek:** Requires `DEEPSEEK_API_KEY`.
    * **OpenRouter:** Requires `OPENROUTER_API_KEY`.
    * Configure the desired provider using the `LLM_PROVIDER` environment variable (defaults to `deepseek`).
* Exa API Key (if using the Researcher agent's capabilities)
    * `EXA_API_KEY` environment variable.
* `uv` package manager (recommended) or `pip`.

## MCP Server Configuration (Client-Side)

This server runs as a standard executable script that communicates via stdio, as expected by MCP. The exact configuration method depends on your specific MCP client implementation. Consult your client's documentation for details.

The `env` section should include the API key for your chosen `LLM_PROVIDER`.

```json
{
  "mcpServers": {
      "mas-sequential-thinking": {
         "command": "uvx",
         "args": [
            "mcp-server-mas-sequential-thinking"
         ],
         "env": {
            "LLM_PROVIDER": "deepseek", // Or "groq", "openrouter"
            // "GROQ_API_KEY": "your_groq_api_key", // Only if LLM_PROVIDER="groq"
            "DEEPSEEK_API_KEY": "your_deepseek_api_key", // Default provider
            // "OPENROUTER_API_KEY": "your_openrouter_api_key", // Only if LLM_PROVIDER="openrouter"
            "DEEPSEEK_BASE_URL": "your_base_url_if_needed", // Optional: If using a custom endpoint for DeepSeek
            "EXA_API_KEY": "your_exa_api_key" // Only if using Exa
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
    Create a `.env` file in the root directory or export the variables:
    ```dotenv
    # --- LLM Configuration ---
    # Select the LLM provider: "deepseek" (default), "groq", or "openrouter"
    LLM_PROVIDER="deepseek"

    # Provide the API key for the chosen provider:
    # GROQ_API_KEY="your_groq_api_key"
    DEEPSEEK_API_KEY="your_deepseek_api_key"
    # OPENROUTER_API_KEY="your_openrouter_api_key"

    # Optional: Base URL override (e.g., for custom DeepSeek endpoints)
    DEEPSEEK_BASE_URL="your_base_url_if_needed"

    # Optional: Specify different models for Team Coordinator and Specialist Agents
    # Defaults are set within the code based on the provider if these are not set.
    # Example for Groq:
    # GROQ_TEAM_MODEL_ID="llama3-70b-8192"
    # GROQ_AGENT_MODEL_ID="llama3-8b-8192"
    # Example for DeepSeek:
    # DEEPSEEK_TEAM_MODEL_ID="deepseek-chat"  # Note: `deepseek-reasoner` is not recommended as it doesn't support function calling
    # DEEPSEEK_AGENT_MODEL_ID="deepseek-chat"  # Recommended for specialists
    # Example for OpenRouter:
    # OPENROUTER_TEAM_MODEL_ID="deepseek/deepseek-r1"
    # OPENROUTER_AGENT_MODEL_ID="deepseek/deepseek-chat-v3-0324"

    # --- External Tools ---
    # Required ONLY if the Researcher agent is used and needs Exa
    EXA_API_KEY="your_exa_api_key"
    ```

    **Note on Model Selection:**

    *   The `TEAM_MODEL_ID` is used by the Coordinator (the `Team` object itself). This role requires strong reasoning, synthesis, and delegation capabilities. Using a more powerful model (like `deepseek-r1`, `claude-3-opus`, or `gpt-4-turbo`) is often beneficial here, even if it's slower or more expensive.
    *   The `AGENT_MODEL_ID` is used by the specialist agents (Planner, Researcher, etc.). These agents handle more focused sub-tasks. You might choose a faster or more cost-effective model (like `deepseek-v3`, `claude-3-sonnet`, `llama3-70b`) for specialists, depending on the complexity of the tasks they typically handle and your budget/performance requirements.
    *   The defaults provided in `main.py` (e.g., `deepseek-reasoner` for agents when using DeepSeek) are starting points. Experimentation is encouraged to find the optimal balance for your specific use case.

3.  **Install Dependencies:**

    * **Using `uv` (Recommended):**
        ```bash
        # Install uv if you don't have it:
        # curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
        # source $HOME/.cargo/env # Or restart your shell

        uv pip install -r requirements.txt
        # Or if a pyproject.toml exists with dependencies:
        # uv pip install .
        ```
    * **Using `pip`:**
        ```bash
        pip install -r requirements.txt
        # Or if a pyproject.toml exists with dependencies:
        # pip install .
        ```

## Usage

Run the server script (assuming the main script is named `main.py` or similar based on your file structure):

```bash
python your_main_script_name.py
```

The server will start and listen for requests via stdio, making the `sequentialthinking` tool available to compatible MCP clients (like certain LLMs or testing frameworks).

### `sequentialthinking` Tool Parameters

The tool expects arguments matching the `ThoughtData` Pydantic model:

```python
# Simplified representation
{
    "thought": str,              # Content of the current thought/step
    "thoughtNumber": int,        # Sequence number (>=1)
    "totalThoughts": int,        # Estimated total steps (>=1, suggest >=5)
    "nextThoughtNeeded": bool,   # Is another step required after this?
    "isRevision": bool = False,  # Is this revising a previous thought?
    "revisesThought": Optional[int] = None, # If isRevision, which thought number?
    "branchFromThought": Optional[int] = None, # If branching, from which thought?
    "branchId": Optional[str] = None, # Unique ID for the branch
    "needsMoreThoughts": bool = False # Signal if estimate is too low before last step
}
```

### Interacting with the Tool (Conceptual Example)

An LLM would interact with this tool iteratively:

1.  **LLM:** Uses `sequential-thinking-starter` prompt with the problem.
2.  **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 1`, initial `thought` (e.g., "Plan the analysis..."), `totalThoughts` estimate, `nextThoughtNeeded: True`.
3.  **Server:** MAS processes the thought -> Coordinator synthesizes response & provides guidance (e.g., "Analysis plan complete. Suggest researching X next. No revisions recommended yet.").
4.  **LLM:** Receives JSON response containing `coordinatorResponse`.
5.  **LLM:** Formulates the next thought (e.g., "Research X using Exa...") based on the `coordinatorResponse`.
6.  **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 2`, the new `thought`, updated `totalThoughts` (if needed), `nextThoughtNeeded: True`.
7.  **Server:** MAS processes -> Coordinator synthesizes (e.g., "Research complete. Findings suggest a flaw in thought #1's assumption. RECOMMENDATION: Revise thought #1...").
8.  **LLM:** Receives response, sees the recommendation.
9.  **LLM:** Formulates a revision thought.
10. **LLM:** Calls `sequentialthinking` tool with `thoughtNumber: 3`, the revision `thought`, `isRevision: True`, `revisesThought: 1`, `nextThoughtNeeded: True`.
11. **... and so on, potentially branching or extending as needed.**

### Tool Response Format

The tool returns a JSON string containing:

```json
{
  "processedThoughtNumber": int,
  "estimatedTotalThoughts": int,
  "nextThoughtNeeded": bool,
  "coordinatorResponse": "Synthesized output from the agent team, including analysis, findings, and guidance for the next step...",
  "branches": ["list", "of", "branch", "ids"],
  "thoughtHistoryLength": int,
  "branchDetails": {
    "currentBranchId": "main | branchId",
    "branchOriginThought": null | int,
    "allBranches": {"main": count, "branchId": count, ...}
  },
  "isRevision": bool,
  "revisesThought": null | int,
  "isBranch": bool,
  "status": "success | validation_error | failed",
  "error": "Error message if status is not success" // Optional
}
```

## Logging

* Logs are written to `~/.sequential_thinking/logs/sequential_thinking.log`.
* Uses Python's standard `logging` module.
* Includes rotating file handler (10MB limit, 5 backups) and console handler (INFO level).
* Logs include timestamps, levels, logger names, and messages, including formatted thought representations.

## Development

(Add development guidelines here if applicable, e.g., setting up dev environments, running tests, linting.)

1.  Clone the repository.
2.  Set up a virtual environment.
3.  Install dependencies, potentially including development extras:
    ```bash
    # Using uv
    uv pip install -e ".[dev]"
    # Using pip
    pip install -e ".[dev]"
    ```
4.  Run linters/formatters/tests.

## License

MIT
