"""Rate limiting and DoS protection implementation.

Implements token bucket algorithm with sliding window for accurate rate limiting
and protection against resource exhaustion attacks.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import ClassVar

from mcp_server_mas_sequential_thinking.utils import setup_logging

logger = setup_logging()


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, limit_type: str, retry_after: float):
        self.limit_type = limit_type
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for {limit_type}. "
            f"Retry after {retry_after:.1f} seconds."
        )


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Token bucket settings
    max_requests_per_minute: int = 30
    max_requests_per_hour: int = 500
    max_concurrent_requests: int = 5

    # Request size limits
    max_request_size: int = 50_000  # 50KB
    max_thought_length: int = 10_000  # 10K characters

    # Burst allowance
    burst_multiplier: float = 1.5

    # Cleanup intervals
    cleanup_interval: int = 300  # 5 minutes


class TokenBucket:
    """Token bucket implementation for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """Initialize token bucket.

        Args:
            capacity: Maximum number of tokens
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill

        # Add tokens based on elapsed time
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Seconds to wait
        """
        self._refill()

        if self.tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self.tokens
        return tokens_needed / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for accurate request tracking."""

    def __init__(self, window_seconds: int):
        """Initialize sliding window counter.

        Args:
            window_seconds: Size of the sliding window
        """
        self.window_seconds = window_seconds
        self.requests: deque = deque()
        self._lock = asyncio.Lock()

    async def add_request(self) -> None:
        """Add a request to the window."""
        async with self._lock:
            now = time.monotonic()
            self.requests.append(now)
            self._cleanup(now)

    async def get_count(self) -> int:
        """Get current request count in the window."""
        async with self._lock:
            now = time.monotonic()
            self._cleanup(now)
            return len(self.requests)

    def _cleanup(self, now: float) -> None:
        """Remove old requests outside the window."""
        cutoff = now - self.window_seconds

        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()

    async def get_oldest_request_age(self) -> float:
        """Get age of oldest request in window."""
        async with self._lock:
            if not self.requests:
                return 0.0

            now = time.monotonic()
            return now - self.requests[0]


class RateLimiter:
    """Comprehensive rate limiter with DoS protection."""

    # Class-level rate limiting (shared across all instances)
    _global_token_bucket: ClassVar[TokenBucket | None] = None
    _global_minute_counter: ClassVar[SlidingWindowCounter | None] = None
    _global_hour_counter: ClassVar[SlidingWindowCounter | None] = None
    _concurrent_requests: ClassVar[int] = 0
    _concurrent_lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _client_buckets: ClassVar[dict] = defaultdict(lambda: None)
    _cleanup_task: ClassVar[asyncio.Task | None] = None

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Initialize global rate limiters on first instance
        if RateLimiter._global_token_bucket is None:
            self._initialize_global_limiters()

    def _initialize_global_limiters(self) -> None:
        """Initialize global rate limiting structures."""
        # Token bucket for burst protection
        burst_capacity = int(
            self.config.max_requests_per_minute * self.config.burst_multiplier
        )
        refill_rate = self.config.max_requests_per_minute / 60.0

        RateLimiter._global_token_bucket = TokenBucket(burst_capacity, refill_rate)

        # Sliding windows for time-based limits
        RateLimiter._global_minute_counter = SlidingWindowCounter(60)
        RateLimiter._global_hour_counter = SlidingWindowCounter(3600)

        # Cleanup task will be started lazily when first request is made
        # (to avoid "no running event loop" error during initialization)

        logger.info(
            f"Rate limiter initialized: "
            f"{self.config.max_requests_per_minute} req/min, "
            f"{self.config.max_requests_per_hour} req/hour, "
            f"{self.config.max_concurrent_requests} concurrent"
        )

    async def check_rate_limit(self, client_id: str = "default") -> None:
        """Check if request is within rate limits.

        Args:
            client_id: Identifier for the client

        Raises:
            RateLimitExceeded: If any rate limit is exceeded
        """
        # Start cleanup task on first request (lazy initialization)
        if RateLimiter._cleanup_task is None:
            try:
                RateLimiter._cleanup_task = asyncio.create_task(self._periodic_cleanup())
                logger.debug("Started rate limiter cleanup task")
            except RuntimeError:
                # If no event loop is running, that's okay - we'll try again next time
                pass

        # Check concurrent request limit
        await self._check_concurrent_limit()

        # Check token bucket (burst protection)
        if not await RateLimiter._global_token_bucket.consume():
            wait_time = RateLimiter._global_token_bucket.get_wait_time()
            raise RateLimitExceeded("burst", wait_time)

        # Check per-minute limit
        minute_count = await RateLimiter._global_minute_counter.get_count()
        if minute_count >= self.config.max_requests_per_minute:
            oldest_age = await RateLimiter._global_minute_counter.get_oldest_request_age()
            wait_time = max(0.0, 60.0 - oldest_age)
            raise RateLimitExceeded("per_minute", wait_time)

        # Check per-hour limit
        hour_count = await RateLimiter._global_hour_counter.get_count()
        if hour_count >= self.config.max_requests_per_hour:
            oldest_age = await RateLimiter._global_hour_counter.get_oldest_request_age()
            wait_time = max(0.0, 3600.0 - oldest_age)
            raise RateLimitExceeded("per_hour", wait_time)

        # Record the request
        await RateLimiter._global_minute_counter.add_request()
        await RateLimiter._global_hour_counter.add_request()

        logger.debug(
            f"Rate limit check passed for {client_id}: "
            f"{minute_count + 1}/{self.config.max_requests_per_minute} per minute, "
            f"{hour_count + 1}/{self.config.max_requests_per_hour} per hour"
        )

    async def _check_concurrent_limit(self) -> None:
        """Check concurrent request limit."""
        async with RateLimiter._concurrent_lock:
            if RateLimiter._concurrent_requests >= self.config.max_concurrent_requests:
                raise RateLimitExceeded(
                    "concurrent",
                    retry_after=1.0,
                )

            RateLimiter._concurrent_requests += 1

    async def release_concurrent_slot(self) -> None:
        """Release a concurrent request slot."""
        async with RateLimiter._concurrent_lock:
            RateLimiter._concurrent_requests = max(
                0, RateLimiter._concurrent_requests - 1
            )

    def validate_request_size(self, thought: str, field_name: str = "thought") -> None:
        """Validate request size to prevent resource exhaustion.

        Args:
            thought: The thought text to validate
            field_name: Name of the field being validated

        Raises:
            ValueError: If request size exceeds limits
        """
        thought_size = len(thought.encode("utf-8"))

        if thought_size > self.config.max_request_size:
            raise ValueError(
                f"{field_name} size ({thought_size} bytes) exceeds maximum "
                f"({self.config.max_request_size} bytes). "
                "Split your request into smaller parts."
            )

        if len(thought) > self.config.max_thought_length:
            raise ValueError(
                f"{field_name} length ({len(thought)} chars) exceeds maximum "
                f"({self.config.max_thought_length} chars). "
                "Split your thought into multiple sequential thoughts."
            )

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of rate limiting data."""
        while True:
            try:
                await asyncio.sleep(self.config.cleanup_interval)

                # Cleanup old client buckets
                if RateLimiter._client_buckets:
                    logger.debug("Performing rate limiter cleanup")

            except asyncio.CancelledError:
                logger.info("Rate limiter cleanup task cancelled")
                break
            except Exception as e:
                logger.error(f"Error in rate limiter cleanup: {e}")

    @classmethod
    def get_current_stats(cls) -> dict:
        """Get current rate limiting statistics.

        Returns:
            Dictionary with current statistics
        """
        return {
            "concurrent_requests": cls._concurrent_requests,
            "tokens_available": (
                cls._global_token_bucket.tokens if cls._global_token_bucket else 0
            ),
        }

    @classmethod
    async def reset_limits(cls) -> None:
        """Reset all rate limits (for testing purposes)."""
        cls._concurrent_requests = 0

        if cls._global_token_bucket:
            cls._global_token_bucket.tokens = cls._global_token_bucket.capacity

        if cls._global_minute_counter:
            cls._global_minute_counter.requests.clear()

        if cls._global_hour_counter:
            cls._global_hour_counter.requests.clear()

        logger.info("Rate limits reset")