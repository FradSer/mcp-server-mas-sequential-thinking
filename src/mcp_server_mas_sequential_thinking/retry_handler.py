"""Retry handling utilities for robust processing."""

import asyncio
import time
import logging
from typing import Callable, TypeVar, Any, Optional
from dataclasses import dataclass

from .constants import DefaultTimeouts, PerformanceMetrics
from .types import ThoughtProcessingError

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry operations."""

    max_attempts: int = DefaultTimeouts.MAX_RETRY_ATTEMPTS
    sleep_duration: float = PerformanceMetrics.RETRY_SLEEP_DURATION
    exponential_base: float = DefaultTimeouts.RETRY_EXPONENTIAL_BASE
    use_exponential_backoff: bool = False


class RetryHandler:
    """Handles retry logic with configurable strategies."""

    def __init__(self, config: RetryConfig = None):
        """Initialize retry handler with configuration."""
        self.config = config or RetryConfig()

    async def execute_with_retry(
        self,
        operation: Callable[[], Any],
        operation_name: str,
        context_info: Optional[dict] = None,
    ) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        max_retries = self.config.max_attempts

        for retry_count in range(max_retries + 1):
            try:
                self._log_attempt(
                    retry_count, max_retries, operation_name, context_info
                )

                start_time = time.time()
                result = await operation()
                processing_time = time.time() - start_time

                self._log_success(operation_name, processing_time)
                return result

            except Exception as e:
                last_exception = e
                self._log_error(retry_count, max_retries, operation_name, e)

                if retry_count < max_retries:
                    await self._wait_before_retry(retry_count)
                else:
                    self._log_exhaustion(max_retries, operation_name)
                    raise ThoughtProcessingError(
                        f"{operation_name} failed after {max_retries + 1} attempts: {e}"
                    ) from e

        # Should never reach here, but provide safety
        raise ThoughtProcessingError(
            f"Unexpected error in retry logic for {operation_name}"
        ) from last_exception

    def _log_attempt(
        self,
        retry_count: int,
        max_retries: int,
        operation_name: str,
        context_info: Optional[dict],
    ) -> None:
        """Log retry attempt information."""
        logger.info(
            f"Processing attempt {retry_count + 1}/{max_retries + 1}: {operation_name}"
        )

        if context_info:
            for key, value in context_info.items():
                logger.info(f"  {key}: {value}")

    def _log_success(self, operation_name: str, processing_time: float) -> None:
        """Log successful operation completion."""
        logger.info(f"✅ {operation_name} completed in {processing_time:.3f}s")

    def _log_error(
        self, retry_count: int, max_retries: int, operation_name: str, error: Exception
    ) -> None:
        """Log error information."""
        logger.error(f"{operation_name} error on attempt {retry_count + 1}: {error}")

        if retry_count < max_retries:
            logger.info(f"Retrying... ({retry_count + 1}/{max_retries})")

    def _log_exhaustion(self, max_retries: int, operation_name: str) -> None:
        """Log retry exhaustion."""
        logger.error(f"All retry attempts exhausted for {operation_name}")

    async def _wait_before_retry(self, retry_count: int) -> None:
        """Wait before retry with optional exponential backoff."""
        if self.config.use_exponential_backoff:
            wait_time = self.config.sleep_duration * (
                self.config.exponential_base**retry_count
            )
        else:
            wait_time = self.config.sleep_duration

        await asyncio.sleep(wait_time)


class TeamProcessingRetryHandler(RetryHandler):
    """Specialized retry handler for team processing operations."""

    def __init__(self):
        """Initialize with team processing specific configuration."""
        config = RetryConfig(
            max_attempts=DefaultTimeouts.MAX_RETRY_ATTEMPTS,
            sleep_duration=PerformanceMetrics.RETRY_SLEEP_DURATION,
            use_exponential_backoff=True,
        )
        super().__init__(config)

    async def execute_team_processing(
        self, team_operation: Callable[[], Any], team_info: dict, complexity_level: str
    ) -> Any:
        """Execute team processing with specialized retry logic."""
        context_info = {
            "complexity": complexity_level,
            "team": team_info.get("name", "unknown"),
            "agents": team_info.get("member_count", 0),
            "leader": team_info.get("leader_model", "unknown"),
        }

        return await self.execute_with_retry(
            team_operation, "MULTI-AGENT TEAM PROCESSING", context_info
        )
