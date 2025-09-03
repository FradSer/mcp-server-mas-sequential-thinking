"""Global test configuration and fixtures for the MCP Sequential Thinking Server.

This file provides shared fixtures and configuration for all tests across the project.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from models import ThoughtData
from session import SessionMemory


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_log_dir():
    """Temporary directory for test logging."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Clean environment for testing."""
    test_env = {
        "LLM_PROVIDER": "deepseek",
        "DEEPSEEK_API_KEY": "test-key",
        "EXA_API_KEY": "test-exa-key",
        "LOG_LEVEL": "DEBUG",
    }
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    return test_env


@pytest.fixture
def mock_team():
    """Mock team instance for testing."""
    team = MagicMock()
    team.arun = AsyncMock(return_value="Mock team response")
    return team


@pytest.fixture
def sample_session(mock_team):
    """Sample session with mock team."""
    return SessionMemory(team=mock_team)


@pytest.fixture
def basic_thought_data():
    """Basic thought data for testing."""
    return ThoughtData(
        thought="Test thought content",
        thought_number=1,
        total_thoughts=5,
        next_needed=True,
    )


@pytest.fixture
def revision_thought_data():
    """Sample revision thought data."""
    return ThoughtData(
        thought="Revised test thought content",
        thought_number=2,
        total_thoughts=5,
        next_needed=True,
        is_revision=True,
        revises_thought=1,
    )


@pytest.fixture
def branch_thought_data():
    """Sample branch thought data."""
    return ThoughtData(
        thought="Branch test thought content",
        thought_number=2,
        total_thoughts=5,
        next_needed=True,
        branch_from=1,
        branch_id="test-branch-1",
    )


@pytest.fixture
def mock_server_config():
    """Mock server configuration for testing."""
    from main import ServerConfig

    return ServerConfig(
        provider="deepseek", log_level="DEBUG", max_retries=3, timeout=30.0
    )


@pytest.fixture
def mock_server_state(mock_server_config, sample_session):
    """Mock server state for testing."""
    from main import _server_state

    original_config = _server_state._config
    original_session = _server_state._session

    _server_state.initialize(mock_server_config, sample_session)

    yield _server_state

    # Restore original state
    _server_state._config = original_config
    _server_state._session = original_session
