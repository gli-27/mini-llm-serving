"""Background worker pool for priority-aware inference processing.

Implements a consumer loop that pulls requests from the Redis priority
queue (ZPOPMIN) and dispatches them to the ThreadPoolExecutor for
inference. Multiple instances can safely pull from the same queue —
ZPOPMIN is atomic (each item popped exactly once).

Request handlers await an asyncio.Future that the worker resolves
after inference completes, enabling true priority-aware scheduling:
CRITICAL requests are dequeued and processed before STANDARD/BATCH
even under high concurrency.

Architecture:
    Request → Router → enqueue(priority) → Redis Sorted Set
                                                ↓
                        Worker (asyncio.Task) ← ZPOPMIN (poll loop)
                               ↓
                        ThreadPoolExecutor.submit(generate)
                               ↓
                        Result → Future.set_result() → response to client
"""

from __future__ import annotations

import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from llm_serving.core.batcher import BatchScheduler
from llm_serving.core.circuit_breaker import CircuitBreaker
from llm_serving.core.inference import generate
from llm_serving.exceptions import CircuitOpenError
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager
from llm_serving.queue.priority_queue import PriorityQueue

logger = get_logger(__name__)


class InferenceWorkerPool:
    """Background worker that consumes from the priority queue and runs inference.

    Manages a registry of pending request Futures. The router enqueues a
    request and registers a Future, then awaits it. The worker pulls items
    from Redis via ZPOPMIN and resolves the corresponding Future with the
    inference result.

    Args:
        priority_queue: The shared PriorityQueue instance.
        model_manager: The ModelManager with loaded model/tokenizer.
        executor: ThreadPoolExecutor for running sync inference.
        num_workers: Number of concurrent worker tasks (defaults to executor max_workers).
        poll_interval_s: Seconds to sleep when the queue is empty.

    Example::

        pool = InferenceWorkerPool(queue, model_mgr, executor, num_workers=2)
        await pool.start()

        # In request handler:
        future = pool.register_request("req-123")
        await queue.enqueue("req-123", priority, payload)
        result = await future  # blocks until worker processes it

        await pool.stop()
    """

    def __init__(
        self,
        priority_queue: PriorityQueue,
        model_manager: ModelManager,
        executor: ThreadPoolExecutor,
        circuit_breaker: CircuitBreaker | None = None,
        batch_scheduler: BatchScheduler | None = None,
        num_workers: int = 1,
        poll_interval_s: float = 0.01,
    ) -> None:
        self._priority_queue = priority_queue
        self._model_manager = model_manager
        self._executor = executor
        self._circuit_breaker = circuit_breaker
        self._batch_scheduler = batch_scheduler
        self._num_workers = num_workers
        self._poll_interval_s = poll_interval_s
        self._tasks: list[asyncio.Task[None]] = []
        self._running = False

        # Registry: request_id → Future that the request handler awaits
        self._pending: dict[str, asyncio.Future[Any]] = {}

    def register_request(self, request_id: str) -> asyncio.Future[Any]:
        """Register a request and return a Future the handler can await.

        Must be called BEFORE enqueue so the worker can find the Future
        when it dequeues the item.

        Args:
            request_id: Unique request identifier (matches the one in the queue).

        Returns:
            An asyncio.Future that will be resolved with the inference result
            (generated_text, prompt_tokens, completion_tokens) or an exception.
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        return future

    def cancel_request(self, request_id: str) -> None:
        """Cancel a pending request (e.g., on client disconnect).

        Args:
            request_id: The request to cancel.
        """
        future = self._pending.pop(request_id, None)
        if future and not future.done():
            future.cancel()

    async def start(self) -> None:
        """Start the worker tasks.

        Spawns ``num_workers`` asyncio tasks that each run the
        consume-and-process loop.
        """
        self._running = True
        for i in range(self._num_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"inference-worker-{i}",
            )
            self._tasks.append(task)
        logger.info("Worker pool started", num_workers=self._num_workers)

    async def stop(self) -> None:
        """Stop all worker tasks gracefully.

        Sets the running flag to False and waits for all tasks to finish
        their current iteration.
        """
        self._running = False
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        # Cancel any remaining pending futures
        for _request_id, future in self._pending.items():
            if not future.done():
                future.cancel()
        self._pending.clear()

        logger.info("Worker pool stopped")

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop: poll queue → run inference → resolve future.

        Continuously polls the priority queue. When an item is available,
        runs inference in the executor and resolves the corresponding Future.
        When the queue is empty, sleeps briefly to avoid busy-waiting.

        Args:
            worker_id: Numeric ID for logging/debugging.
        """
        logger.info("Worker started", worker_id=worker_id)

        while self._running:
            try:
                item = await self._priority_queue.dequeue()

                if item is None:
                    # Queue empty — sleep briefly to avoid busy-wait
                    await asyncio.sleep(self._poll_interval_s)
                    continue

                request_id = item.get("request_id")
                if not request_id or request_id not in self._pending:
                    logger.warning(
                        "Dequeued orphan request (no pending future)",
                        request_id=request_id,
                        worker_id=worker_id,
                    )
                    continue

                # Extract inference parameters from the payload
                prompt = item.get("prompt", "")
                max_tokens = item.get("max_tokens", 256)
                temperature = item.get("temperature", 0.7)
                seed = item.get("seed")

                logger.info(
                    "Worker processing request",
                    request_id=request_id,
                    priority=item.get("priority"),
                    worker_id=worker_id,
                )

                # Check circuit breaker before inference
                if self._circuit_breaker and not await self._circuit_breaker.allow_request():
                    future = self._pending.pop(request_id, None)
                    if future and not future.done():
                        future.set_exception(CircuitOpenError("Circuit breaker is OPEN"))
                    continue

                # Run inference — batched or direct depending on config
                try:
                    if self._batch_scheduler:
                        # Batching enabled: submit to batch scheduler
                        # The scheduler collects, pads, and calls
                        # generate_batch() when batch is full or timeout
                        result = await self._batch_scheduler.submit(
                            request_id=request_id,
                            prompt=prompt,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            seed=seed,
                        )
                    else:
                        # Batching disabled: direct single-request inference
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self._executor,
                            functools.partial(
                                generate,
                                model_manager=self._model_manager,
                                prompt=prompt,
                                max_new_tokens=max_tokens,
                                temperature=temperature,
                                seed=seed,
                            ),
                        )

                    # Record success with circuit breaker
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_success()

                    # Resolve the future with the result
                    future = self._pending.pop(request_id, None)
                    if future and not future.done():
                        future.set_result(result)

                except Exception as exc:
                    # Record failure with circuit breaker
                    if self._circuit_breaker:
                        await self._circuit_breaker.record_failure()

                    logger.exception(
                        "Worker inference failed",
                        request_id=request_id,
                        worker_id=worker_id,
                    )
                    future = self._pending.pop(request_id, None)
                    if future and not future.done():
                        future.set_exception(exc)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Worker loop error", worker_id=worker_id)
                await asyncio.sleep(self._poll_interval_s)

        logger.info("Worker stopped", worker_id=worker_id)
