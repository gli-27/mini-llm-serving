"""Integration tests using fakeredis for rate limiter and priority queue.

These tests use fakeredis (in-memory Redis emulator with Lua support)
to test the actual Redis interactions without requiring a running Redis server.

NOTE: These are **unit-level** integration tests. While fakeredis supports
EVALSHA and Lua scripting, its behavior may differ subtly from real Redis
(e.g., floating-point precision, edge cases in Lua number handling). For
true Lua script validation against a production Redis instance, run the
integration test suite with ``--redis-url=redis://...`` in a real environment
(CI with redis-service or docker-compose).
"""

import asyncio
import time

import fakeredis.aioredis
import pytest

from llm_serving.queue.priority_queue import Priority, PriorityQueue
from llm_serving.queue.rate_limiter import TokenBucketRateLimiter
from llm_serving.queue.redis_client import RedisClient


@pytest.fixture
async def fake_redis_client():
    """Create a RedisClient backed by fakeredis for integration testing."""
    client = RedisClient("redis://fake:6379/0")
    # Monkey-patch the connect to use fakeredis instead of real Redis
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    client._redis = fake_redis
    yield client
    await fake_redis.aclose()


class TestRateLimiterIntegration:
    """Integration tests for TokenBucketRateLimiter with fakeredis."""

    async def test_burst_allows_up_to_bucket_size(self, fake_redis_client: RedisClient) -> None:
        """Burst of requests up to bucket_size should all be allowed."""
        limiter = TokenBucketRateLimiter(
            fake_redis_client, bucket_size=5, refill_rate=1.0
        )

        results = []
        for _ in range(5):
            allowed, info = await limiter.try_consume("user-burst")
            results.append(allowed)

        assert all(results), "All 5 requests within bucket should be allowed"

    async def test_rejects_when_bucket_empty(self, fake_redis_client: RedisClient) -> None:
        """Requests beyond bucket_size should be rejected."""
        limiter = TokenBucketRateLimiter(
            fake_redis_client, bucket_size=3, refill_rate=1.0
        )

        # Exhaust the bucket
        for _ in range(3):
            await limiter.try_consume("user-exhaust")

        # Next request should be denied
        allowed, info = await limiter.try_consume("user-exhaust")
        assert allowed is False
        assert info["remaining"] == 0.0
        assert info["retry_after"] > 0

    async def test_refill_allows_again(self, fake_redis_client: RedisClient) -> None:
        """After enough time passes, tokens refill and requests are allowed."""
        limiter = TokenBucketRateLimiter(
            fake_redis_client, bucket_size=2, refill_rate=100.0  # 100 tokens/sec = fast refill
        )

        # Exhaust bucket
        await limiter.try_consume("user-refill")
        await limiter.try_consume("user-refill")

        # Should be denied now
        allowed, _ = await limiter.try_consume("user-refill")
        assert allowed is False

        # Wait for refill (100 tokens/sec → 10ms for 1 token)
        await asyncio.sleep(0.05)

        # Should be allowed again
        allowed, info = await limiter.try_consume("user-refill")
        assert allowed is True

    async def test_different_keys_independent_buckets(
        self, fake_redis_client: RedisClient
    ) -> None:
        """Different API keys should have independent buckets."""
        limiter = TokenBucketRateLimiter(
            fake_redis_client, bucket_size=2, refill_rate=0.1
        )

        # Exhaust user-a
        await limiter.try_consume("user-a")
        await limiter.try_consume("user-a")
        allowed_a, _ = await limiter.try_consume("user-a")
        assert allowed_a is False

        # user-b should still have full bucket
        allowed_b, info = await limiter.try_consume("user-b")
        assert allowed_b is True

    async def test_remaining_decrements(self, fake_redis_client: RedisClient) -> None:
        """Remaining count should decrement with each consume."""
        limiter = TokenBucketRateLimiter(
            fake_redis_client, bucket_size=5, refill_rate=0.01
        )

        _, info1 = await limiter.try_consume("user-dec")
        _, info2 = await limiter.try_consume("user-dec")
        _, info3 = await limiter.try_consume("user-dec")

        # Remaining should decrease (approximately, accounting for tiny refill)
        assert info1["remaining"] >= info2["remaining"] >= info3["remaining"]
        assert info1["remaining"] == 4.0  # Started with 5, consumed 1


class TestPriorityQueueIntegration:
    """Integration tests for PriorityQueue with fakeredis."""

    async def test_dequeue_respects_priority_order(
        self, fake_redis_client: RedisClient
    ) -> None:
        """Items should be dequeued in priority order (CRITICAL first)."""
        queue = PriorityQueue(fake_redis_client, queue_key="test_pq_order")

        # Enqueue in reverse order: BATCH first, CRITICAL last
        await queue.enqueue("req-batch", Priority.BATCH, {"prompt": "batch"})
        await queue.enqueue("req-standard", Priority.STANDARD, {"prompt": "standard"})
        await queue.enqueue("req-critical", Priority.CRITICAL, {"prompt": "critical"})

        # Dequeue should return CRITICAL first, then STANDARD, then BATCH
        item1 = await queue.dequeue()
        item2 = await queue.dequeue()
        item3 = await queue.dequeue()

        assert item1["request_id"] == "req-critical"
        assert item2["request_id"] == "req-standard"
        assert item3["request_id"] == "req-batch"

    async def test_fifo_within_same_priority(
        self, fake_redis_client: RedisClient
    ) -> None:
        """Items with the same priority should be dequeued FIFO."""
        queue = PriorityQueue(fake_redis_client, queue_key="test_pq_fifo")

        await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "first"})
        await asyncio.sleep(0.001)  # Ensure different timestamps
        await queue.enqueue("req-2", Priority.STANDARD, {"prompt": "second"})
        await asyncio.sleep(0.001)
        await queue.enqueue("req-3", Priority.STANDARD, {"prompt": "third"})

        item1 = await queue.dequeue()
        item2 = await queue.dequeue()
        item3 = await queue.dequeue()

        assert item1["request_id"] == "req-1"
        assert item2["request_id"] == "req-2"
        assert item3["request_id"] == "req-3"

    async def test_dequeue_empty_returns_none(
        self, fake_redis_client: RedisClient
    ) -> None:
        """Dequeue from empty queue should return None."""
        queue = PriorityQueue(fake_redis_client, queue_key="test_pq_empty")
        item = await queue.dequeue()
        assert item is None

    async def test_queue_depth_tracks_size(
        self, fake_redis_client: RedisClient
    ) -> None:
        """queue_depth should reflect the current number of items."""
        queue = PriorityQueue(fake_redis_client, queue_key="test_pq_depth")

        assert await queue.queue_depth() == 0

        await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "a"})
        assert await queue.queue_depth() == 1

        await queue.enqueue("req-2", Priority.CRITICAL, {"prompt": "b"})
        assert await queue.queue_depth() == 2

        await queue.dequeue()
        assert await queue.queue_depth() == 1

        await queue.dequeue()
        assert await queue.queue_depth() == 0

    async def test_critical_always_before_batch(
        self, fake_redis_client: RedisClient
    ) -> None:
        """A CRITICAL request enqueued after BATCH should still dequeue first."""
        queue = PriorityQueue(fake_redis_client, queue_key="test_pq_prio_beat")

        # Enqueue BATCH first
        await queue.enqueue("req-batch-1", Priority.BATCH, {"prompt": "batch1"})
        await queue.enqueue("req-batch-2", Priority.BATCH, {"prompt": "batch2"})

        # Then enqueue CRITICAL
        await queue.enqueue("req-critical", Priority.CRITICAL, {"prompt": "urgent"})

        # CRITICAL should come out first despite being enqueued last
        item = await queue.dequeue()
        assert item["request_id"] == "req-critical"
        assert item["priority"] == Priority.CRITICAL


class TestFailOpen:
    """Test that the rate limiter fails open when Redis is unavailable."""

    async def test_rate_limiter_fails_open_on_connection_error(self) -> None:
        """Rate limiter should allow requests when Redis connection fails."""
        # Create a client that's not connected (will raise RuntimeError)
        broken_client = RedisClient("redis://nonexistent:6379/0")

        limiter = TokenBucketRateLimiter(
            broken_client, bucket_size=10, refill_rate=2.0
        )

        # Should fail open (allow) instead of crashing
        allowed, info = await limiter.try_consume("user-123")
        assert allowed is True
        assert info["remaining"] == 10.0
        assert info["retry_after"] == 0.0
