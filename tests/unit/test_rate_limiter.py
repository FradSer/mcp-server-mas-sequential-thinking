"""Unit tests for rate limiter and DoS protection.

Tests the rate limiting functionality to ensure proper protection
against resource exhaustion and denial of service attacks.
"""

import asyncio
import pytest

from mcp_server_mas_sequential_thinking.security.rate_limiter import (
    RateLimiter,
    RateLimitExceeded,
    TokenBucket,
    SlidingWindowCounter,
    RateLimitConfig,
)


class TestTokenBucket:
    """Test suite for token bucket implementation."""

    @pytest.mark.asyncio
    async def test_token_consumption(self):
        """Test that tokens are consumed correctly."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Should be able to consume tokens
        assert await bucket.consume(5) is True
        assert 4.9 <= bucket.tokens <= 5.1  # Allow for small refill during execution

        # Should be able to consume more
        assert await bucket.consume(3) is True
        assert 1.9 <= bucket.tokens <= 2.1  # Allow for small refill during execution

    @pytest.mark.asyncio
    async def test_insufficient_tokens(self):
        """Test behavior when insufficient tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0)

        # Consume most tokens
        await bucket.consume(9)

        # Should fail to consume more than available
        assert await bucket.consume(5) is False
        assert 0.9 <= bucket.tokens <= 1.1  # Still has ~1 token left

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test that tokens refill over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens per second

        # Consume all tokens
        await bucket.consume(10)
        assert bucket.tokens == 0.0

        # Wait for refill
        await asyncio.sleep(0.5)  # 0.5 seconds = 5 tokens

        # Should have refilled
        assert await bucket.consume(4) is True

    def test_wait_time_calculation(self):
        """Test wait time calculation."""
        bucket = TokenBucket(capacity=10, refill_rate=2.0)  # 2 tokens per second
        bucket.tokens = 1.0

        # Need 5 tokens, have 1, need 4 more
        # At 2 tokens/sec, need 2 seconds
        wait_time = bucket.get_wait_time(5)
        assert 1.9 <= wait_time <= 2.1  # Allow for small timing variations

    def test_refill_cap(self):
        """Test that tokens don't exceed capacity."""
        bucket = TokenBucket(capacity=10, refill_rate=100.0)

        # Wait a long time
        import time
        time.sleep(0.1)

        # Refill should not exceed capacity
        bucket._refill()
        assert bucket.tokens <= 10.0


class TestSlidingWindowCounter:
    """Test suite for sliding window counter."""

    @pytest.mark.asyncio
    async def test_add_and_count_requests(self):
        """Test adding and counting requests."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Add some requests
        await counter.add_request()
        await counter.add_request()
        await counter.add_request()

        # Should have 3 requests
        count = await counter.get_count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_window_expiration(self):
        """Test that old requests expire from window."""
        counter = SlidingWindowCounter(window_seconds=1)

        # Add requests
        await counter.add_request()
        await counter.add_request()

        assert await counter.get_count() == 2

        # Wait for window to expire
        await asyncio.sleep(1.1)

        # Requests should have expired
        assert await counter.get_count() == 0

    @pytest.mark.asyncio
    async def test_oldest_request_age(self):
        """Test oldest request age calculation."""
        counter = SlidingWindowCounter(window_seconds=60)

        # Initially no requests
        assert await counter.get_oldest_request_age() == 0.0

        # Add a request
        await counter.add_request()

        # Should have some age
        await asyncio.sleep(0.1)
        age = await counter.get_oldest_request_age()
        assert 0.05 <= age <= 0.15


class TestRateLimiter:
    """Test suite for rate limiter."""

    def setup_method(self):
        """Set up test environment."""
        self.config = RateLimitConfig(
            max_requests_per_minute=10,
            max_requests_per_hour=100,
            max_concurrent_requests=3,
            max_request_size=1000,
            max_thought_length=500,
        )

    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limiter = RateLimiter(self.config)

        # Reset limits for clean test
        await RateLimiter.reset_limits()

        # Should allow requests within limit
        for i in range(5):
            await limiter.check_rate_limit("test_client")
            await limiter.release_concurrent_slot()

    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self):
        """Test concurrent request limiting."""
        limiter = RateLimiter(self.config)
        await RateLimiter.reset_limits()

        # Fill up concurrent slots
        for i in range(self.config.max_concurrent_requests):
            await limiter.check_rate_limit("test_client")

        # Next request should fail
        with pytest.raises(RateLimitExceeded, match="concurrent"):
            await limiter.check_rate_limit("test_client")

        # Release a slot
        await limiter.release_concurrent_slot()

        # Should now succeed
        await limiter.check_rate_limit("test_client")
        await limiter.release_concurrent_slot()

    @pytest.mark.asyncio
    async def test_per_minute_limit(self):
        """Test per-minute rate limiting."""
        limiter = RateLimiter(self.config)
        await RateLimiter.reset_limits()

        # Make requests up to the limit
        for i in range(self.config.max_requests_per_minute):
            await limiter.check_rate_limit("test_client")
            await limiter.release_concurrent_slot()

        # Next request should exceed per-minute limit
        with pytest.raises(RateLimitExceeded, match="per_minute"):
            await limiter.check_rate_limit("test_client")

    @pytest.mark.asyncio
    async def test_burst_protection(self):
        """Test burst protection with token bucket."""
        limiter = RateLimiter(self.config)
        await RateLimiter.reset_limits()

        # Make many rapid requests
        burst_size = int(self.config.max_requests_per_minute * 1.5)

        successful_requests = 0
        for i in range(burst_size):
            try:
                await limiter.check_rate_limit("test_client")
                await limiter.release_concurrent_slot()
                successful_requests += 1
            except RateLimitExceeded:
                break

        # Should have been limited before hitting burst_size
        assert successful_requests < burst_size

    def test_request_size_validation(self):
        """Test request size validation."""
        limiter = RateLimiter(self.config)

        # Valid size should pass
        valid_text = "a" * 500
        limiter.validate_request_size(valid_text, "test_field")

        # Oversized request should fail
        oversized_text = "a" * (self.config.max_request_size + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            limiter.validate_request_size(oversized_text, "test_field")

    def test_thought_length_validation(self):
        """Test thought length validation."""
        limiter = RateLimiter(self.config)

        # Valid length should pass
        valid_text = "a" * 400
        limiter.validate_request_size(valid_text, "test_field")

        # Excessive length should fail
        long_text = "a" * (self.config.max_thought_length + 1)
        with pytest.raises(ValueError, match="exceeds maximum"):
            limiter.validate_request_size(long_text, "test_field")

    @pytest.mark.asyncio
    async def test_rate_limit_statistics(self):
        """Test rate limit statistics tracking."""
        limiter = RateLimiter(self.config)
        await RateLimiter.reset_limits()

        stats = RateLimiter.get_current_stats()
        assert "concurrent_requests" in stats
        assert "tokens_available" in stats

        # Make a request
        await limiter.check_rate_limit("test_client")

        stats = RateLimiter.get_current_stats()
        assert stats["concurrent_requests"] >= 1

        await limiter.release_concurrent_slot()

    @pytest.mark.asyncio
    async def test_rate_limit_reset(self):
        """Test rate limit reset functionality."""
        limiter = RateLimiter(self.config)

        # Make some requests
        for i in range(5):
            await limiter.check_rate_limit("test_client")
            await limiter.release_concurrent_slot()

        # Reset limits
        await RateLimiter.reset_limits()

        stats = RateLimiter.get_current_stats()
        assert stats["concurrent_requests"] == 0


class TestRateLimitEdgeCases:
    """Test edge cases for rate limiting."""

    @pytest.mark.asyncio
    async def test_zero_concurrent_requests(self):
        """Test behavior with zero concurrent requests allowed."""
        config = RateLimitConfig(max_concurrent_requests=0)
        limiter = RateLimiter(config)
        await RateLimiter.reset_limits()

        # Should immediately fail
        with pytest.raises(RateLimitExceeded, match="concurrent"):
            await limiter.check_rate_limit("test_client")

    @pytest.mark.asyncio
    async def test_very_high_limits(self):
        """Test behavior with very high limits."""
        # Reset global state first
        await RateLimiter.reset_limits()

        config = RateLimitConfig(
            max_requests_per_minute=10000,
            max_concurrent_requests=100,
        )
        limiter = RateLimiter(config)

        # Need to reinitialize with new config
        RateLimiter._global_token_bucket = None
        limiter._initialize_global_limiters()

        # Should allow many requests
        for i in range(50):
            await limiter.check_rate_limit("test_client")
            await limiter.release_concurrent_slot()

    def test_unicode_in_request(self):
        """Test request size validation with unicode characters."""
        config = RateLimitConfig(max_request_size=100)
        limiter = RateLimiter(config)

        # Unicode characters take multiple bytes
        unicode_text = "🔥" * 50  # Each emoji is 4 bytes

        # Should fail due to byte size
        with pytest.raises(ValueError, match="exceeds maximum"):
            limiter.validate_request_size(unicode_text, "test_field")

    @pytest.mark.asyncio
    async def test_concurrent_client_requests(self):
        """Test multiple clients making concurrent requests."""
        config = RateLimitConfig(max_concurrent_requests=5)
        limiter = RateLimiter(config)
        await RateLimiter.reset_limits()

        # Simulate multiple clients
        clients = ["client1", "client2", "client3"]

        async def make_request(client_id: str):
            await limiter.check_rate_limit(client_id)
            await asyncio.sleep(0.1)
            await limiter.release_concurrent_slot()

        # Run concurrent requests
        tasks = [make_request(client) for client in clients for _ in range(2)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Some should succeed
        successful = sum(1 for r in results if r is None)
        assert successful > 0