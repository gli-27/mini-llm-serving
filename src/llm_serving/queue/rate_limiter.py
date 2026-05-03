"""Token bucket rate limiter backed by Redis with atomic Lua scripting.

Implements a per-API-key token bucket algorithm using a single Redis Lua
script for atomicity. The bucket refills at a configurable rate and has
a configurable maximum size.

On Redis failure, the limiter fails open (allows the request) and logs
a warning — availability is prioritized over strict enforcement.
"""

from __future__ import annotations

import time

from llm_serving.logging import get_logger
from llm_serving.queue.redis_client import RedisClient

logger = get_logger(__name__)

# Lua script for atomic token bucket rate limiting.
# Keys: [bucket_key]
# Args: [bucket_size, refill_rate, now]
#
# Algorithm:
#   1. Read current tokens and last_refill timestamp from Redis hash
#   2. If key doesn't exist, initialize with full bucket
#   3. Calculate elapsed time and refill tokens (capped at bucket_size)
#   4. Attempt to consume 1 token
#   5. Write updated state back to Redis with TTL
#   6. Return [allowed (0/1), remaining_tokens, retry_after_ms]
_TOKEN_BUCKET_LUA = """
local bucket_key = KEYS[1]
local bucket_size = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

-- Read current state
local data = redis.call('HMGET', bucket_key, 'tokens', 'last_refill')
local tokens = tonumber(data[1])
local last_refill = tonumber(data[2])

-- Initialize if bucket doesn't exist
if tokens == nil then
    tokens = bucket_size
    last_refill = now
end

-- Refill tokens based on elapsed time
local elapsed = now - last_refill
local refill = elapsed * refill_rate
tokens = math.min(bucket_size, tokens + refill)
last_refill = now

-- Try to consume 1 token
local allowed = 0
if tokens >= 1 then
    tokens = tokens - 1
    allowed = 1
end

-- Write state back with TTL (bucket_size / refill_rate * 2 seconds)
local ttl = math.ceil(bucket_size / refill_rate * 2)
redis.call('HSET', bucket_key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
redis.call('EXPIRE', bucket_key, ttl)

-- Calculate retry_after in milliseconds (time until 1 token is available)
local retry_after_ms = 0
if allowed == 0 then
    retry_after_ms = math.ceil((1 - tokens) / refill_rate * 1000)
end

return {allowed, math.floor(tokens), retry_after_ms}
"""


class TokenBucketRateLimiter:
    """Per-API-key token bucket rate limiter using Redis.

    Each API key gets an independent bucket that refills at a constant rate.
    All state is stored in Redis and operations are atomic via a Lua script.

    On Redis failure, the limiter fails open (allows the request) to
    prioritize availability.

    Args:
        redis_client: The shared RedisClient instance.
        bucket_size: Maximum tokens in the bucket (burst capacity).
        refill_rate: Tokens added per second.
        key_prefix: Redis key prefix for bucket state.

    Example::

        limiter = TokenBucketRateLimiter(redis_client, bucket_size=10, refill_rate=2.0)
        allowed, info = await limiter.try_consume("user-123")
        if not allowed:
            # Return 429 with Retry-After header
            retry_after = info["retry_after"]
    """

    def __init__(
        self,
        redis_client: RedisClient,
        bucket_size: int = 10,
        refill_rate: float = 2.0,
        key_prefix: str = "rate_limit",
    ) -> None:
        self._redis_client = redis_client
        self._bucket_size = bucket_size
        self._refill_rate = refill_rate
        self._key_prefix = key_prefix
        self._script_sha: str | None = None

    def _bucket_key(self, api_key: str) -> str:
        """Build the Redis key for an API key's token bucket.

        Args:
            api_key: The client's API key identifier.

        Returns:
            The namespaced Redis key string.
        """
        return f"{self._key_prefix}:{api_key}"

    async def _ensure_script_loaded(self) -> str:
        """Load the Lua script into Redis if not already cached.

        Uses SCRIPT LOAD + EVALSHA for efficiency — the script body is
        sent once, then only the SHA is used for subsequent calls.

        Returns:
            The SHA1 hash of the loaded script.
        """
        if self._script_sha is None:
            redis = self._redis_client.redis
            self._script_sha = await redis.script_load(_TOKEN_BUCKET_LUA)
        return self._script_sha

    async def try_consume(self, api_key: str) -> tuple[bool, dict[str, float]]:
        """Attempt to consume one token from the bucket for the given API key.

        Returns whether the request is allowed and rate limit metadata
        (remaining tokens, limit, and retry-after if denied).

        On Redis failure, fails open (allows the request) and logs a warning.

        Args:
            api_key: The client's API key identifier.

        Returns:
            A tuple of (allowed, info_dict) where info_dict contains:
                - remaining: Tokens remaining in the bucket.
                - limit: Maximum bucket size.
                - retry_after: Seconds until a token is available (0 if allowed).
        """
        try:
            sha = await self._ensure_script_loaded()
            redis = self._redis_client.redis
            now = time.time()

            result = await redis.evalsha(
                sha,
                1,
                self._bucket_key(api_key),
                str(self._bucket_size),
                str(self._refill_rate),
                str(now),
            )

            allowed = bool(result[0])
            remaining = int(result[1])
            retry_after_ms = int(result[2])

            info = {
                "remaining": float(remaining),
                "limit": float(self._bucket_size),
                "retry_after": retry_after_ms / 1000.0,
            }

            if not allowed:
                logger.warning(
                    "Rate limit exceeded",
                    api_key=api_key,
                    remaining=remaining,
                    retry_after_s=info["retry_after"],
                )

            return allowed, info

        except Exception:
            # Fail open: allow the request if Redis is unavailable
            logger.warning(
                "Rate limiter Redis error — failing open",
                api_key=api_key,
                exc_info=True,
            )
            return True, {
                "remaining": float(self._bucket_size),
                "limit": float(self._bucket_size),
                "retry_after": 0.0,
            }
