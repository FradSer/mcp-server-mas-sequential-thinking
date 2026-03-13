"""Unit tests for services/retry_handler.py."""

from unittest.mock import AsyncMock, patch

import pytest

from mcp_server_mas_sequential_thinking.core.types import ThoughtProcessingError
from mcp_server_mas_sequential_thinking.services.retry_handler import (
    RetryConfig,
    RetryHandler,
    TeamProcessingRetryHandler,
)


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        config = RetryConfig()
        assert config.max_attempts >= 1
        assert config.sleep_duration >= 0
        assert config.use_exponential_backoff is False

    def test_custom_values(self):
        config = RetryConfig(max_attempts=5, sleep_duration=0.1, use_exponential_backoff=True)
        assert config.max_attempts == 5
        assert config.sleep_duration == 0.1
        assert config.use_exponential_backoff is True


class TestRetryHandler:
    """Tests for RetryHandler."""

    def make_handler(self, max_attempts: int = 2, sleep: float = 0.0) -> RetryHandler:
        config = RetryConfig(max_attempts=max_attempts, sleep_duration=sleep)
        return RetryHandler(config)

    @pytest.mark.asyncio
    async def test_successful_on_first_attempt(self):
        handler = self.make_handler()
        operation = AsyncMock(return_value="success")
        result = await handler.execute_with_retry(operation, "test_op")
        assert result == "success"
        operation.assert_called_once()

    @pytest.mark.asyncio
    async def test_succeeds_after_retry(self):
        handler = self.make_handler(max_attempts=2, sleep=0.0)
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("transient error")
            return "recovered"

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await handler.execute_with_retry(operation, "test_op")
        assert result == "recovered"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        handler = self.make_handler(max_attempts=1, sleep=0.0)
        operation = AsyncMock(side_effect=RuntimeError("always fails"))

        with patch("asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ThoughtProcessingError, match="always fails"):
                await handler.execute_with_retry(operation, "failing_op")

    @pytest.mark.asyncio
    async def test_with_context_info(self):
        handler = self.make_handler()
        operation = AsyncMock(return_value="ok")
        result = await handler.execute_with_retry(
            operation, "test_op", context_info={"key": "value", "count": 3}
        )
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        config = RetryConfig(max_attempts=2, sleep_duration=0.01, use_exponential_backoff=True)
        handler = RetryHandler(config)
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("fail")
            return "done"

        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            result = await handler.execute_with_retry(operation, "op")
        assert result == "done"
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_config_when_none_provided(self):
        handler = RetryHandler(None)
        assert handler.config is not None


class TestTeamProcessingRetryHandler:
    """Tests for TeamProcessingRetryHandler."""

    @pytest.mark.asyncio
    async def test_execute_team_processing_success(self):
        handler = TeamProcessingRetryHandler()
        team_op = AsyncMock(return_value="team result")
        team_info = {"name": "TestTeam", "member_count": 3, "leader_model": "gpt-4"}

        with patch("asyncio.sleep", new=AsyncMock()):
            result = await handler.execute_team_processing(team_op, team_info, "high")
        assert result == "team result"

    @pytest.mark.asyncio
    async def test_execute_team_processing_missing_info_keys(self):
        """Test with minimal team info dict."""
        handler = TeamProcessingRetryHandler()
        team_op = AsyncMock(return_value="ok")
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await handler.execute_team_processing(team_op, {}, "low")
        assert result == "ok"

    def test_uses_exponential_backoff(self):
        handler = TeamProcessingRetryHandler()
        assert handler.config.use_exponential_backoff is True
