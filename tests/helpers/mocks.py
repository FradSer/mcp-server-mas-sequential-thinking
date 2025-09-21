"""Mock objects and utilities for testing the MCP Sequential Thinking Server."""

from unittest.mock import MagicMock


class MockLLMResponse:
    """Standardized mock LLM response."""

    def __init__(self, content: str = "Mock response"):
        self.content = content

    def __str__(self):
        return self.content


class MockAgentTools:
    """Mock agent tools for testing."""

    @staticmethod
    async def mock_thinking_tool(query: str) -> str:
        """Mock thinking tool response."""
        return f"Thinking about: {query}"

    @staticmethod
    async def mock_research_tool(query: str) -> str:
        """Mock research tool response."""
        return f"Research results for: {query}"


class MockTeam:
    """Mock team with configurable responses."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ["Default mock response"]
        self.call_count = 0
        self.call_history = []

    async def arun(self, prompt: str) -> str:
        """Mock team run with response cycling."""
        self.call_history.append(prompt)

        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = self.responses[-1]  # Use last response if we run out

        self.call_count += 1
        return MockLLMResponse(response)


class ConfigurableMockTeam:
    """Mock team with conditional response logic."""

    def __init__(self):
        self.response_map = {}
        self.default_response = "Default mock response"
        self.call_history = []

    def configure_response(self, keyword: str, response: str):
        """Configure response for prompts containing keyword."""
        self.response_map[keyword] = response

    async def arun(self, prompt: str) -> str:
        """Return configured response based on prompt content."""
        self.call_history.append(prompt)

        for keyword, response in self.response_map.items():
            if keyword in prompt:
                return MockLLMResponse(response)

        return MockLLMResponse(self.default_response)


class MockModelConfig:
    """Mock model configuration for testing."""

    def __init__(
        self,
        provider_class=None,
        team_model_id: str = "test-team-model",
        agent_model_id: str = "test-agent-model",
        api_key: str = "test-key",
    ):
        self.provider_class = provider_class or MagicMock
        self.team_model_id = team_model_id
        self.agent_model_id = agent_model_id
        self.api_key = api_key


class MockEnvironment:
    """Mock environment for testing configuration."""

    def __init__(self, env_vars: dict[str, str]):
        self.env_vars = env_vars

    def __enter__(self):
        import os

        self.original_env = os.environ.copy()
        os.environ.clear()
        os.environ.update(self.env_vars)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import os

        os.environ.clear()
        os.environ.update(self.original_env)
