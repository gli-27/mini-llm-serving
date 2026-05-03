"""Tests for the async Redis client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_serving.queue.redis_client import RedisClient


class TestRedisClient:
    """Tests for RedisClient lifecycle and health checking."""

    def test_init_sets_url(self) -> None:
        """RedisClient stores the URL but does not connect immediately."""
        client = RedisClient("redis://localhost:6379/0")
        assert client._redis_url == "redis://localhost:6379/0"
        assert client._redis is None

    def test_redis_property_raises_when_not_connected(self) -> None:
        """Accessing .redis before connect() should raise RuntimeError."""
        client = RedisClient("redis://localhost:6379/0")
        with pytest.raises(RuntimeError, match="not connected"):
            _ = client.redis

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_connect_creates_redis_and_pings(self, mock_from_url: MagicMock) -> None:
        """connect() should create a Redis instance and verify with PING."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()

        mock_from_url.assert_called_once_with(
            "redis://localhost:6379/0",
            decode_responses=True,
        )
        mock_redis.ping.assert_awaited_once()
        assert client._redis is mock_redis

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_connect_raises_on_unreachable(self, mock_from_url: MagicMock) -> None:
        """connect() should propagate ConnectionError if Redis is unreachable."""
        import redis.exceptions

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=redis.exceptions.ConnectionError("refused"))
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        with pytest.raises(redis.exceptions.ConnectionError):
            await client.connect()

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_close_cleans_up(self, mock_from_url: MagicMock) -> None:
        """close() should call aclose() and set _redis to None."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_redis.aclose = AsyncMock()
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()
        await client.close()

        mock_redis.aclose.assert_awaited_once()
        assert client._redis is None

    async def test_close_is_safe_when_not_connected(self) -> None:
        """close() should be a no-op when not connected."""
        client = RedisClient("redis://localhost:6379/0")
        await client.close()  # Should not raise

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_health_check_returns_true_when_healthy(
        self, mock_from_url: MagicMock
    ) -> None:
        """health_check() should return True when Redis responds to PING."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()

        assert await client.health_check() is True

    async def test_health_check_returns_false_when_not_connected(self) -> None:
        """health_check() should return False when not connected."""
        client = RedisClient("redis://localhost:6379/0")
        assert await client.health_check() is False

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_health_check_returns_false_on_error(
        self, mock_from_url: MagicMock
    ) -> None:
        """health_check() should return False when PING raises an exception."""
        mock_redis = AsyncMock()
        # First ping succeeds (connect), second fails (health_check)
        mock_redis.ping = AsyncMock(side_effect=[True, Exception("connection lost")])
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()

        assert await client.health_check() is False

    @patch("llm_serving.queue.redis_client.aioredis.from_url")
    async def test_redis_property_returns_instance_after_connect(
        self, mock_from_url: MagicMock
    ) -> None:
        """The .redis property should return the Redis instance after connect()."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        mock_from_url.return_value = mock_redis

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()

        assert client.redis is mock_redis
