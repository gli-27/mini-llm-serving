"""Async Redis client with connection pooling and health checking.

Provides a singleton RedisClient that manages an async Redis connection pool
using ``redis.asyncio``. Designed to be initialized at app startup and closed
at shutdown via the FastAPI lifespan.

The connection pool is created lazily from the configured ``redis_url`` and
uses hiredis for fast protocol parsing when available.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client with connection pool lifecycle management.

    Manages a ``redis.asyncio.Redis`` connection pool. Must be explicitly
    connected via :meth:`connect` and disconnected via :meth:`close`.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379/0``).

    Example::

        client = RedisClient("redis://localhost:6379/0")
        await client.connect()
        healthy = await client.health_check()
        await client.close()
    """

    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._redis: aioredis.Redis | None = None

    @property
    def redis(self) -> aioredis.Redis:
        """Return the underlying async Redis instance.

        Raises:
            RuntimeError: If :meth:`connect` has not been called yet.
        """
        if self._redis is None:
            raise RuntimeError("Redis client is not connected. Call connect() first.")
        return self._redis

    async def connect(self) -> None:
        """Create the async Redis connection pool and verify connectivity.

        Creates a ``redis.asyncio.Redis`` instance from the configured URL
        with connection pooling enabled, then issues a PING to verify the
        connection is alive.

        Raises:
            redis.ConnectionError: If Redis is unreachable.
        """
        logger.info("Connecting to Redis", redis_url=self._redis_url)
        self._redis = aioredis.from_url(
            self._redis_url,
            decode_responses=True,
        )
        # Verify connectivity on startup
        await self._redis.ping()
        logger.info("Redis connection established")

    async def close(self) -> None:
        """Close the Redis connection pool and release all connections.

        Safe to call even if not connected (no-op in that case).
        """
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed")

    async def health_check(self) -> bool:
        """Check if Redis is reachable by issuing a PING command.

        Returns:
            True if Redis responds to PING, False otherwise.
        """
        if self._redis is None:
            return False
        try:
            return bool(await self._redis.ping())
        except Exception:
            logger.warning("Redis health check failed", exc_info=True)
            return False
