"""Tests for the token bucket rate limiter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_serving.queue.rate_limiter import TokenBucketRateLimiter
from llm_serving.queue.redis_client import RedisClient


@pytest.fixture
def mock_redis_for_limiter() -> MagicMock:
    """Create a mock RedisClient for rate limiter tests."""
    client = MagicMock(spec=RedisClient)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.script_load = AsyncMock(return_value="fake_sha")
    mock_redis_instance.evalsha = AsyncMock(return_value=[1, 9, 0])
    # Make the .redis property return our mock
    type(client).redis = property(lambda self: mock_redis_instance)
    client._mock_redis = mock_redis_instance  # for easy test access
    return client


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_init_stores_config(self, mock_redis_for_limiter: MagicMock) -> None:
        """Limiter should store bucket_size and refill_rate."""
        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=20, refill_rate=5.0)
        assert limiter._bucket_size == 20
        assert limiter._refill_rate == 5.0

    def test_bucket_key_format(self, mock_redis_for_limiter: MagicMock) -> None:
        """_bucket_key should return a namespaced key."""
        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, key_prefix="rl")
        assert limiter._bucket_key("user-123") == "rl:user-123"

    async def test_try_consume_allowed(self, mock_redis_for_limiter: MagicMock) -> None:
        """try_consume should return (True, info) when tokens are available."""
        # Lua returns [allowed=1, remaining=9, retry_after_ms=0]
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(return_value=[1, 9, 0])

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=10, refill_rate=2.0)
        allowed, info = await limiter.try_consume("user-123")

        assert allowed is True
        assert info["remaining"] == 9.0
        assert info["limit"] == 10.0
        assert info["retry_after"] == 0.0

    async def test_try_consume_denied(self, mock_redis_for_limiter: MagicMock) -> None:
        """try_consume should return (False, info) when bucket is empty."""
        # Lua returns [allowed=0, remaining=0, retry_after_ms=500]
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(return_value=[0, 0, 500])

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=10, refill_rate=2.0)
        allowed, info = await limiter.try_consume("user-123")

        assert allowed is False
        assert info["remaining"] == 0.0
        assert info["limit"] == 10.0
        assert info["retry_after"] == 0.5  # 500ms = 0.5s

    async def test_try_consume_loads_script_once(self, mock_redis_for_limiter: MagicMock) -> None:
        """The Lua script should be loaded only once (cached SHA)."""
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(return_value=[1, 8, 0])

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=10, refill_rate=2.0)

        await limiter.try_consume("user-1")
        await limiter.try_consume("user-2")
        await limiter.try_consume("user-3")

        # script_load called only once
        mock_redis_for_limiter._mock_redis.script_load.assert_awaited_once()
        # evalsha called 3 times
        assert mock_redis_for_limiter._mock_redis.evalsha.await_count == 3

    async def test_try_consume_passes_correct_args(self, mock_redis_for_limiter: MagicMock) -> None:
        """evalsha should be called with the correct key and arguments."""
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(return_value=[1, 9, 0])

        limiter = TokenBucketRateLimiter(
            mock_redis_for_limiter,
            bucket_size=10,
            refill_rate=2.0,
            key_prefix="rate_limit",
        )

        with patch("llm_serving.queue.rate_limiter.time.time", return_value=1000000.0):
            await limiter.try_consume("api-key-abc")

        call_args = mock_redis_for_limiter._mock_redis.evalsha.call_args
        assert call_args[0][0] == "fake_sha"  # SHA
        assert call_args[0][1] == 1  # num keys
        assert call_args[0][2] == "rate_limit:api-key-abc"  # key
        assert call_args[0][3] == "10"  # bucket_size
        assert call_args[0][4] == "2.0"  # refill_rate
        assert call_args[0][5] == "1000000.0"  # now

    async def test_try_consume_fails_open_on_redis_error(
        self, mock_redis_for_limiter: MagicMock
    ) -> None:
        """On Redis failure, try_consume should allow the request (fail open)."""
        mock_redis_for_limiter._mock_redis.script_load = AsyncMock(
            side_effect=ConnectionError("Redis down")
        )

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=10, refill_rate=2.0)
        allowed, info = await limiter.try_consume("user-123")

        assert allowed is True
        assert info["remaining"] == 10.0
        assert info["limit"] == 10.0
        assert info["retry_after"] == 0.0

    async def test_try_consume_fails_open_on_evalsha_error(
        self, mock_redis_for_limiter: MagicMock
    ) -> None:
        """On evalsha failure, try_consume should allow the request (fail open)."""
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(side_effect=Exception("timeout"))

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=5, refill_rate=1.0)
        allowed, info = await limiter.try_consume("user-456")

        assert allowed is True
        assert info["remaining"] == 5.0
        assert info["limit"] == 5.0
        assert info["retry_after"] == 0.0

    async def test_different_api_keys_get_different_buckets(
        self, mock_redis_for_limiter: MagicMock
    ) -> None:
        """Each API key should have an independent bucket (different Redis key)."""
        mock_redis_for_limiter._mock_redis.evalsha = AsyncMock(return_value=[1, 9, 0])

        limiter = TokenBucketRateLimiter(mock_redis_for_limiter, bucket_size=10, refill_rate=2.0)

        await limiter.try_consume("user-a")
        await limiter.try_consume("user-b")

        calls = mock_redis_for_limiter._mock_redis.evalsha.call_args_list
        key_a = calls[0][0][2]
        key_b = calls[1][0][2]
        assert key_a == "rate_limit:user-a"
        assert key_b == "rate_limit:user-b"
        assert key_a != key_b
