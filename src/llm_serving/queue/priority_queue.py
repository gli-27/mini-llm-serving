"""Priority queue backed by a Redis Sorted Set.

Uses Redis ZADD/ZPOPMIN/ZCARD for a priority-aware request queue.
Requests are scored by ``priority_level * 1e10 + timestamp`` so that
higher-priority items (lower numeric level) are always dequeued first,
with FIFO ordering within the same priority tier.

Priority tiers:
    - CRITICAL = 1 (dequeued first)
    - STANDARD = 2
    - BATCH    = 3 (dequeued last)
"""

from __future__ import annotations

import json
import time
from enum import IntEnum

from llm_serving.logging import get_logger
from llm_serving.queue.redis_client import RedisClient

logger = get_logger(__name__)


class Priority(IntEnum):
    """Priority tiers for the request queue.

    Lower numeric value = higher priority = dequeued first.

    Attributes:
        CRITICAL: Highest priority. Dequeued before all others.
        STANDARD: Default priority for normal requests.
        BATCH: Lowest priority. Dequeued only when no higher items exist.
    """

    CRITICAL = 1
    STANDARD = 2
    BATCH = 3


# Score multiplier to separate priority tiers.
# With 1e10, each priority tier has ~317 years of timestamp space
# before colliding with the next tier — effectively infinite separation.
_PRIORITY_MULTIPLIER = 1e10


class PriorityQueue:
    """Redis Sorted Set-backed priority queue for inference requests.

    Each enqueued request gets a score of ``priority * 1e10 + timestamp``,
    ensuring lower-priority-number items are always dequeued first (ZPOPMIN),
    with FIFO ordering within the same tier.

    The payload is stored as a JSON string in the sorted set member field.

    Args:
        redis_client: The shared RedisClient instance.
        queue_key: Redis key for the sorted set.

    Example::

        queue = PriorityQueue(redis_client)
        position = await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "Hi"})
        item = await queue.dequeue()  # Returns the highest-priority item
        depth = await queue.queue_depth()
    """

    def __init__(
        self,
        redis_client: RedisClient,
        queue_key: str = "inference_queue",
    ) -> None:
        self._redis_client = redis_client
        self._queue_key = queue_key

    def _compute_score(self, priority: int, timestamp: float) -> float:
        """Compute the sorted set score for a request.

        Score = priority_level * 1e10 + timestamp. Lower scores are
        dequeued first (ZPOPMIN), so CRITICAL (1) beats STANDARD (2)
        beats BATCH (3). Within the same tier, earlier timestamps win.

        Args:
            priority: The priority tier (1=CRITICAL, 2=STANDARD, 3=BATCH).
            timestamp: Unix timestamp of the request.

        Returns:
            The computed score for Redis ZADD.
        """
        return priority * _PRIORITY_MULTIPLIER + timestamp

    async def enqueue(
        self,
        request_id: str,
        priority: int,
        payload: dict[str, object],
    ) -> int:
        """Add a request to the priority queue.

        The payload is serialized to JSON and stored as the sorted set
        member, with the priority-based score determining dequeue order.

        Args:
            request_id: Unique identifier for the request.
            priority: Priority tier (use ``Priority`` enum values).
            payload: Request data to store (must be JSON-serializable).

        Returns:
            The current queue depth after insertion.
        """
        redis = self._redis_client.redis
        now = time.time()
        score = self._compute_score(priority, now)

        # Store request_id + payload as the member
        member = json.dumps(
            {
                "request_id": request_id,
                "priority": priority,
                "enqueued_at": now,
                **payload,
            }
        )

        await redis.zadd(self._queue_key, {member: score})
        depth = await redis.zcard(self._queue_key)

        logger.info(
            "Enqueued request",
            request_id=request_id,
            priority=priority,
            score=score,
            queue_depth=depth,
        )

        return depth

    async def dequeue(self) -> dict[str, object] | None:
        """Remove and return the highest-priority (lowest-score) item.

        Uses ZPOPMIN for atomic pop of the lowest-scored member.

        Returns:
            The deserialized request payload dict, or None if the queue
            is empty.
        """
        redis = self._redis_client.redis
        result = await redis.zpopmin(self._queue_key, count=1)

        if not result:
            return None

        # zpopmin returns [(member, score), ...]
        member, score = result[0]
        item: dict[str, object] = json.loads(member)

        logger.info(
            "Dequeued request",
            request_id=item.get("request_id"),
            priority=item.get("priority"),
            score=score,
        )

        return item

    async def queue_depth(self) -> int:
        """Return the number of items currently in the queue.

        Uses ZCARD for O(1) cardinality check.

        Returns:
            The number of pending requests in the queue.
        """
        redis = self._redis_client.redis
        return await redis.zcard(self._queue_key)
