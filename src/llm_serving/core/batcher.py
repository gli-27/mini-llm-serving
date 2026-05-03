"""Timeout-based dynamic batch scheduler for inference.

Collects individual inference requests into batches, then delegates to
``generate_batch()`` in ``inference.py`` for the actual model forward pass.
This maximizes GPU/CPU utilization by amortizing the fixed overhead.

Batch triggers (whichever comes first):
    1. Batch is full (``len == max_batch_size``)
    2. ``max_batch_wait_ms`` elapsed since the first request arrived

Architecture::

    Worker dequeues request → BatchScheduler.submit(request)
        │
    Collect phase (wait up to max_batch_wait_ms OR max_batch_size reached)
        │
    Tokenize + left-pad → call generate_batch() from inference.py
        │
    Split outputs → resolve each request's Future individually

Separation of concerns:
    - BatchScheduler: collection, timeout, padding, output splitting
    - inference.generate_batch(): model.generate() call + decoding
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import torch

from llm_serving.core.inference import generate_batch
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager

logger = get_logger(__name__)


@dataclass
class BatchItem:
    """A single request waiting to be batched.

    Attributes:
        request_id: Unique identifier for the request.
        prompt: The input text prompt.
        max_tokens: Maximum new tokens to generate.
        temperature: Sampling temperature.
        seed: Optional random seed.
        future: asyncio.Future to resolve with the result.
    """

    request_id: str
    prompt: str
    max_tokens: int
    temperature: float
    seed: int | None
    future: asyncio.Future[tuple[str, int, int]] = field(repr=False)


class BatchScheduler:
    """Timeout-based dynamic batch scheduler.

    Collects individual requests into a batch. Triggers forward when:
    1. Batch is full (``len == max_batch_size``), OR
    2. ``max_batch_wait_ms`` elapsed since first request arrived

    Whichever happens first.

    Trade-offs documented:
    - Temperature: uses ``max(temperatures)`` in the batch for simplicity.
      Production systems group requests by temperature to avoid this.
    - max_tokens: uses ``max(all_max_tokens)`` so all sequences generate
      to the longest requested length. Shorter requests may over-generate
      (truncated by generate_batch).
    - Left-padding: correct for decoder-only models (generation continues
      from the right side of the sequence).

    Args:
        model_manager: The ModelManager with loaded model/tokenizer.
        executor: ThreadPoolExecutor for running sync batch inference.
        max_batch_size: Maximum requests per batch (default 8).
        max_batch_wait_ms: Max ms to wait before flushing partial batch.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        executor: ThreadPoolExecutor,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 50,
    ) -> None:
        self._model_manager = model_manager
        self._executor = executor
        self._max_batch_size = max_batch_size
        self._max_batch_wait_ms = max_batch_wait_ms
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background collector loop."""
        self._running = True
        self._task = asyncio.create_task(self._collector_loop(), name="batch-collector")
        logger.info(
            "BatchScheduler started",
            max_batch_size=self._max_batch_size,
            max_batch_wait_ms=self._max_batch_wait_ms,
        )

    async def stop(self) -> None:
        """Stop the collector loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("BatchScheduler stopped")

    async def submit(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        seed: int | None,
    ) -> tuple[str, int, int]:
        """Submit a single request. Returns when the batch completes.

        The caller awaits this coroutine, which blocks until the batch
        containing this request is forwarded through the model and the
        result is available.

        Args:
            request_id: Unique request identifier.
            prompt: Input text prompt.
            max_tokens: Max new tokens to generate.
            temperature: Sampling temperature.
            seed: Optional random seed (best-effort in batched mode).

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens).
        """
        loop = asyncio.get_event_loop()
        future: asyncio.Future[tuple[str, int, int]] = loop.create_future()

        item = BatchItem(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            future=future,
        )

        await self._queue.put(item)
        return await future

    async def _collector_loop(self) -> None:
        """Background task: collect requests, trigger batch on size/timeout.

        Waits for either the batch to fill up OR the timeout to expire,
        then takes all pending items and processes them as a batch.
        """
        while self._running:
            try:
                # Wait for the first item to arrive
                first_item = await self._queue.get()
                batch: list[BatchItem] = [first_item]

                # Collect more items until batch full or timeout
                timeout_s = self._max_batch_wait_ms / 1000.0

                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._wait_for_batch(batch),
                        timeout=timeout_s,
                    )

                # Drain any remaining items from queue (non-blocking)
                while not self._queue.empty() and len(batch) < self._max_batch_size:
                    try:
                        item = self._queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                logger.info(
                    "Batch collected",
                    batch_size=len(batch),
                    request_ids=[item.request_id for item in batch],
                )

                # Forward the batch through the model
                await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Batch collector error")

    async def _wait_for_batch(self, batch: list[BatchItem]) -> None:
        """Wait until the batch is full by consuming from the queue.

        Args:
            batch: The current batch being built (mutated in place).
        """
        while len(batch) < self._max_batch_size:
            item = await self._queue.get()
            batch.append(item)
            if len(batch) >= self._max_batch_size:
                return

    async def _process_batch(self, batch: list[BatchItem]) -> None:
        """Tokenize, left-pad, delegate to generate_batch, resolve futures.

        This method handles:
        1. Tokenization of all prompts
        2. Left-padding to uniform length
        3. Delegating to generate_batch() for the model forward pass
        4. Resolving each item's Future with its result

        Args:
            batch: List of BatchItems to process together.
        """
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                self._executor,
                functools.partial(self._prepare_and_generate, batch),
            )

            # Resolve each future with its result
            for item, result in zip(batch, results, strict=True):
                if not item.future.done():
                    item.future.set_result(result)

        except Exception as exc:
            logger.exception("Batch forward failed")
            # Fail all items in the batch
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)

    def _prepare_and_generate(self, batch: list[BatchItem]) -> list[tuple[str, int, int]]:
        """Synchronous: tokenize, left-pad, call generate_batch().

        Left-padding is correct for decoder-only models because generation
        continues from the rightmost token. The attention_mask ensures
        padding tokens are ignored.

        Args:
            batch: List of BatchItems to process.

        Returns:
            List of (generated_text, prompt_tokens, completion_tokens)
            tuples, one per batch item.
        """
        assert self._model_manager.tokenizer is not None

        tokenizer = self._model_manager.tokenizer
        device = self._model_manager.settings.device

        # Tokenize all prompts individually
        tokenized = [tokenizer(item.prompt, return_tensors="pt")["input_ids"][0] for item in batch]
        prompt_lengths = [ids.shape[0] for ids in tokenized]

        # Left-pad to max length in batch
        max_prompt_len = max(prompt_lengths)
        pad_token_id = tokenizer.pad_token_id or 0

        padded_ids = []
        attention_masks = []
        for ids in tokenized:
            pad_len = max_prompt_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_token_id, dtype=ids.dtype)
                padded = torch.cat([padding, ids])
                mask = torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.long),
                        torch.ones(ids.shape[0], dtype=torch.long),
                    ]
                )
            else:
                padded = ids
                mask = torch.ones(ids.shape[0], dtype=torch.long)
            padded_ids.append(padded)
            attention_masks.append(mask)

        input_ids = torch.stack(padded_ids).to(device)
        attention_mask = torch.stack(attention_masks).to(device)

        # Use max tokens and max temperature across the batch
        # Trade-off: simpler but less optimal than grouping by params
        max_new_tokens = max(item.max_tokens for item in batch)
        batch_temperature = max(item.temperature for item in batch)
        per_request_max_tokens = [item.max_tokens for item in batch]

        # Delegate to inference.generate_batch() for the model forward pass
        return generate_batch(
            model_manager=self._model_manager,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
            max_new_tokens=max_new_tokens,
            temperature=batch_temperature,
            per_request_max_tokens=per_request_max_tokens,
        )
