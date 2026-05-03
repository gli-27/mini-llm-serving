"""Tests for the Redis-backed priority queue."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_serving.queue.priority_queue import Priority, PriorityQueue, _PRIORITY_MULTIPLIER
from llm_serving.queue.redis_client import RedisClient


@pytest.fixture
def mock_redis_for_queue() -> MagicMock:
    """Create a mock RedisClient for priority queue tests."""
    client = MagicMock(spec=RedisClient)
    mock_redis_instance = AsyncMock()
    mock_redis_instance.zadd = AsyncMock(return_value=1)
    mock_redis_instance.zcard = AsyncMock(return_value=1)
    mock_redis_instance.zpopmin = AsyncMock(return_value=[])
    type(client).redis = property(lambda self: mock_redis_instance)
    client._mock_redis = mock_redis_instance
    return client


class TestPriorityEnum:
    """Tests for the Priority enum."""

    def test_priority_values(self) -> None:
        """Priority tiers should have correct numeric values."""
        assert Priority.CRITICAL == 1
        assert Priority.STANDARD == 2
        assert Priority.BATCH == 3

    def test_priority_ordering(self) -> None:
        """CRITICAL < STANDARD < BATCH for correct dequeue order."""
        assert Priority.CRITICAL < Priority.STANDARD < Priority.BATCH


class TestPriorityQueue:
    """Tests for PriorityQueue operations."""

    def test_compute_score_critical_lower_than_standard(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """CRITICAL score should be lower than STANDARD at the same timestamp."""
        queue = PriorityQueue(mock_redis_for_queue)
        now = 1000000.0
        critical_score = queue._compute_score(Priority.CRITICAL, now)
        standard_score = queue._compute_score(Priority.STANDARD, now)
        batch_score = queue._compute_score(Priority.BATCH, now)

        assert critical_score < standard_score < batch_score

    def test_compute_score_fifo_within_tier(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """Earlier timestamps should produce lower scores within the same tier."""
        queue = PriorityQueue(mock_redis_for_queue)
        earlier = queue._compute_score(Priority.STANDARD, 1000000.0)
        later = queue._compute_score(Priority.STANDARD, 1000001.0)

        assert earlier < later

    def test_compute_score_priority_beats_timestamp(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """A CRITICAL request should always score lower than a STANDARD one,
        even if the STANDARD was enqueued much earlier."""
        queue = PriorityQueue(mock_redis_for_queue)
        # Standard enqueued very early
        standard_early = queue._compute_score(Priority.STANDARD, 0.0)
        # Critical enqueued much later
        critical_late = queue._compute_score(Priority.CRITICAL, 9999999999.0)

        assert critical_late < standard_early

    async def test_enqueue_calls_zadd_and_zcard(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """enqueue should call ZADD to add and ZCARD to get depth."""
        mock_redis_for_queue._mock_redis.zcard = AsyncMock(return_value=3)

        queue = PriorityQueue(mock_redis_for_queue, queue_key="test_queue")
        depth = await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "Hi"})

        assert depth == 3
        mock_redis_for_queue._mock_redis.zadd.assert_awaited_once()
        mock_redis_for_queue._mock_redis.zcard.assert_awaited_once()

        # Verify ZADD was called with correct key
        zadd_call = mock_redis_for_queue._mock_redis.zadd.call_args
        assert zadd_call[0][0] == "test_queue"

    async def test_enqueue_stores_payload_as_json(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """enqueue should serialize the payload to JSON as the member."""
        queue = PriorityQueue(mock_redis_for_queue)

        with patch("llm_serving.queue.priority_queue.time.time", return_value=1000000.0):
            await queue.enqueue("req-42", Priority.CRITICAL, {"prompt": "Hello"})

        zadd_call = mock_redis_for_queue._mock_redis.zadd.call_args
        member_dict = zadd_call[0][1]  # {member_json: score}
        member_json = list(member_dict.keys())[0]
        member = json.loads(member_json)

        assert member["request_id"] == "req-42"
        assert member["priority"] == Priority.CRITICAL
        assert member["enqueued_at"] == 1000000.0
        assert member["prompt"] == "Hello"

    async def test_enqueue_computes_correct_score(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """enqueue should use priority * 1e10 + timestamp as the score."""
        queue = PriorityQueue(mock_redis_for_queue)

        with patch("llm_serving.queue.priority_queue.time.time", return_value=1000000.0):
            await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "Hi"})

        zadd_call = mock_redis_for_queue._mock_redis.zadd.call_args
        member_dict = zadd_call[0][1]
        score = list(member_dict.values())[0]
        expected = Priority.STANDARD * _PRIORITY_MULTIPLIER + 1000000.0

        assert score == expected

    async def test_dequeue_returns_item_on_success(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """dequeue should return the deserialized payload dict."""
        payload = json.dumps({
            "request_id": "req-1",
            "priority": 2,
            "enqueued_at": 1000000.0,
            "prompt": "Hello",
        })
        mock_redis_for_queue._mock_redis.zpopmin = AsyncMock(
            return_value=[(payload, 20000001000000.0)]
        )

        queue = PriorityQueue(mock_redis_for_queue)
        item = await queue.dequeue()

        assert item is not None
        assert item["request_id"] == "req-1"
        assert item["priority"] == 2
        assert item["prompt"] == "Hello"

    async def test_dequeue_returns_none_when_empty(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """dequeue should return None when the queue is empty."""
        mock_redis_for_queue._mock_redis.zpopmin = AsyncMock(return_value=[])

        queue = PriorityQueue(mock_redis_for_queue)
        item = await queue.dequeue()

        assert item is None

    async def test_dequeue_calls_zpopmin(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """dequeue should use ZPOPMIN with count=1."""
        mock_redis_for_queue._mock_redis.zpopmin = AsyncMock(return_value=[])

        queue = PriorityQueue(mock_redis_for_queue, queue_key="my_queue")
        await queue.dequeue()

        mock_redis_for_queue._mock_redis.zpopmin.assert_awaited_once_with(
            "my_queue", count=1
        )

    async def test_queue_depth_calls_zcard(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """queue_depth should use ZCARD to get cardinality."""
        mock_redis_for_queue._mock_redis.zcard = AsyncMock(return_value=5)

        queue = PriorityQueue(mock_redis_for_queue, queue_key="my_queue")
        depth = await queue.queue_depth()

        assert depth == 5
        mock_redis_for_queue._mock_redis.zcard.assert_awaited_once_with("my_queue")

    async def test_custom_queue_key(
        self, mock_redis_for_queue: MagicMock
    ) -> None:
        """PriorityQueue should use the configured queue_key for all operations."""
        queue = PriorityQueue(mock_redis_for_queue, queue_key="custom_q")

        await queue.enqueue("req-1", Priority.STANDARD, {"prompt": "test"})
        zadd_key = mock_redis_for_queue._mock_redis.zadd.call_args[0][0]
        assert zadd_key == "custom_q"

        zcard_key = mock_redis_for_queue._mock_redis.zcard.call_args[0][0]
        assert zcard_key == "custom_q"
