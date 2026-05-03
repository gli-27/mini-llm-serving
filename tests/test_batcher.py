"""Tests for the timeout-based dynamic batch scheduler."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import torch

from llm_serving.config import Settings
from llm_serving.core.batcher import BatchScheduler
from llm_serving.models.loader import ModelManager


def _make_model_manager() -> ModelManager:
    """Create a ModelManager with mocked model and tokenizer for batch tests."""
    settings = Settings(
        app_env="testing",
        model_name="test-model",
        device="cpu",
        max_new_tokens=32,
        generation_timeout_s=10.0,
        max_concurrent_requests=1,
        log_level="debug",
    )
    manager = ModelManager(settings)

    # Mock tokenizer
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0

    def _tokenize(text: str, return_tensors: str = "pt") -> dict:
        # Simulate different token lengths for different prompts
        if text == "short":
            ids = torch.tensor([[1, 2, 3]])
        elif text == "medium length":
            ids = torch.tensor([[1, 2, 3, 4, 5]])
        elif text == "this is a longer prompt":
            ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        else:
            # Default: hash-based length for variety
            n = max(3, len(text.split()))
            ids = torch.tensor([[i + 1 for i in range(n)]])
        return {"input_ids": ids}

    tokenizer.side_effect = _tokenize
    tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    def _decode(ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        return f"generated_{len(ids)}_tokens"

    tokenizer.decode = MagicMock(side_effect=_decode)

    # Mock model
    model = MagicMock()

    def _generate(*args, **kwargs) -> torch.Tensor:
        # model.generate(input_ids, ...) — first positional arg is input_ids
        input_ids = args[0] if args else kwargs.get("input_ids")
        batch_size = input_ids.shape[0]
        max_new = kwargs.get("max_new_tokens", 10)
        # Generate dummy output: input + new tokens
        new_tokens = torch.full((batch_size, max_new), 99, dtype=input_ids.dtype)
        return torch.cat([input_ids, new_tokens], dim=1)

    model.generate = MagicMock(side_effect=_generate)

    manager.tokenizer = tokenizer
    manager.model = model
    return manager


class TestPaddingCorrectness:
    """Test left-padding + attention mask for different length prompts."""

    def test_left_padding_aligns_prompts(self) -> None:
        """Shorter prompts should be left-padded with pad_token_id."""
        manager = _make_model_manager()

        # Simulate the padding logic directly
        tokenizer = manager.tokenizer
        prompts = ["short", "medium length", "this is a longer prompt"]
        tokenized = [tokenizer(p, return_tensors="pt")["input_ids"][0] for p in prompts]

        # Verify different lengths
        assert tokenized[0].shape[0] == 3  # "short" = 3 tokens
        assert tokenized[1].shape[0] == 5  # "medium" = 5 tokens
        assert tokenized[2].shape[0] == 7  # "longer" = 7 tokens

        # Left-pad to max length (7)
        max_len = max(t.shape[0] for t in tokenized)
        assert max_len == 7

        pad_token_id = 0
        for _i, ids in enumerate(tokenized):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padded = torch.cat(
                    [
                        torch.full((pad_len,), pad_token_id, dtype=ids.dtype),
                        ids,
                    ]
                )
                # Verify padding is on the LEFT
                assert padded[:pad_len].tolist() == [0] * pad_len
                # Verify original tokens are on the RIGHT
                assert padded[pad_len:].tolist() == ids.tolist()
                assert padded.shape[0] == max_len

    def test_attention_mask_zeros_for_padding(self) -> None:
        """Attention mask should be 0 for padding positions, 1 for real tokens."""
        manager = _make_model_manager()
        tokenizer = manager.tokenizer

        ids_short = tokenizer("short", return_tensors="pt")["input_ids"][0]  # 3 tokens

        max_len = 7
        pad_len = max_len - ids_short.shape[0]  # 4

        mask = torch.cat(
            [
                torch.zeros(pad_len, dtype=torch.long),
                torch.ones(ids_short.shape[0], dtype=torch.long),
            ]
        )

        # First 4 positions should be 0 (padding)
        assert mask[:4].tolist() == [0, 0, 0, 0]
        # Last 3 positions should be 1 (real tokens)
        assert mask[4:].tolist() == [1, 1, 1]

    def test_no_padding_for_same_length(self) -> None:
        """Prompts of the same length should not need padding."""
        manager = _make_model_manager()
        tokenizer = manager.tokenizer

        ids1 = tokenizer("short", return_tensors="pt")["input_ids"][0]
        ids2 = tokenizer("short", return_tensors="pt")["input_ids"][0]

        max_len = max(ids1.shape[0], ids2.shape[0])
        assert ids1.shape[0] == max_len  # No padding needed
        mask = torch.ones(ids1.shape[0], dtype=torch.long)
        assert mask.sum().item() == max_len  # All 1s


class TestBatchSchedulerStop:
    """Test that pending requests resolve or get cancelled on stop."""

    async def test_stop_cancels_collector_task(self) -> None:
        """stop() should cancel the collector task cleanly."""
        manager = _make_model_manager()
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=ThreadPoolExecutor(max_workers=1),
            max_batch_size=4,
            max_batch_wait_ms=50,
        )

        await scheduler.start()
        assert scheduler._task is not None
        assert not scheduler._task.done()

        await scheduler.stop()
        assert scheduler._task.done()

    async def test_stop_when_not_started(self) -> None:
        """stop() should be safe to call even if never started."""
        manager = _make_model_manager()
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=ThreadPoolExecutor(max_workers=1),
        )
        await scheduler.stop()  # Should not raise


class TestSingleRequestLatency:
    """Test that a single request doesn't wait for a full batch."""

    async def test_single_request_fires_on_timeout(self) -> None:
        """A single request should be processed after max_batch_wait_ms, not wait for full batch."""
        manager = _make_model_manager()
        executor = ThreadPoolExecutor(max_workers=1)
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=executor,
            max_batch_size=8,  # Large batch — won't fill with 1 request
            max_batch_wait_ms=20,  # Short timeout — should fire quickly
        )

        await scheduler.start()

        try:
            # Submit a single request — should complete after ~20ms timeout, not wait for 8
            result = await asyncio.wait_for(
                scheduler.submit(
                    request_id="req-single",
                    prompt="short",
                    max_tokens=10,
                    temperature=0.7,
                    seed=None,
                ),
                timeout=2.0,  # 2s overall timeout (should complete in ~50ms)
            )

            # Should get a result (not timeout)
            assert result is not None
            generated_text, prompt_tokens, completion_tokens = result
            assert isinstance(generated_text, str)
            assert prompt_tokens > 0
            assert completion_tokens > 0
        finally:
            await scheduler.stop()
            executor.shutdown(wait=False)

    async def test_full_batch_fires_immediately(self) -> None:
        """A full batch should fire immediately without waiting for timeout."""
        manager = _make_model_manager()
        executor = ThreadPoolExecutor(max_workers=1)
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=executor,
            max_batch_size=2,  # Small batch — fills quickly
            max_batch_wait_ms=5000,  # Long timeout — should NOT wait this long
        )

        await scheduler.start()

        try:
            # Submit 2 requests concurrently — should trigger immediately
            results = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("req-1", "short", 10, 0.7, None),
                    scheduler.submit("req-2", "short", 10, 0.7, None),
                ),
                timeout=2.0,  # Should complete fast (batch full, no timeout wait)
            )

            assert len(results) == 2
            for text, prompt_tokens, _completion_tokens in results:
                assert isinstance(text, str)
                assert prompt_tokens > 0
        finally:
            await scheduler.stop()
            executor.shutdown(wait=False)


class TestBatchForward:
    """Test the full _prepare_and_generate flow with mocked model."""

    async def test_batch_produces_correct_number_of_results(self) -> None:
        """Batch forward should produce one result per input item."""
        manager = _make_model_manager()
        executor = ThreadPoolExecutor(max_workers=1)
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=executor,
            max_batch_size=3,
            max_batch_wait_ms=20,
        )

        await scheduler.start()

        try:
            results = await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("r1", "short", 5, 0.7, None),
                    scheduler.submit("r2", "medium length", 10, 0.7, None),
                    scheduler.submit("r3", "short", 8, 0.5, None),
                ),
                timeout=2.0,
            )

            assert len(results) == 3
            # Each result is (text, prompt_tokens, completion_tokens)
            for text, pt, ct in results:
                assert isinstance(text, str)
                assert pt > 0
                assert ct > 0
        finally:
            await scheduler.stop()
            executor.shutdown(wait=False)

    async def test_generate_batch_called_with_correct_shapes(self) -> None:
        """generate_batch should receive correctly shaped tensors."""
        manager = _make_model_manager()
        executor = ThreadPoolExecutor(max_workers=1)
        scheduler = BatchScheduler(
            model_manager=manager,
            executor=executor,
            max_batch_size=2,
            max_batch_wait_ms=20,
        )

        await scheduler.start()

        try:
            await asyncio.wait_for(
                asyncio.gather(
                    scheduler.submit("r1", "short", 5, 0.7, None),
                    scheduler.submit("r2", "medium length", 10, 0.7, None),
                ),
                timeout=2.0,
            )

            # Verify model.generate was called with batched input
            call_kwargs = manager.model.generate.call_args
            input_ids = call_kwargs.kwargs.get(
                "input_ids", call_kwargs[0][0] if call_kwargs[0] else None
            )
            attention_mask = call_kwargs.kwargs.get("attention_mask")

            # Should be batch_size=2
            assert input_ids.shape[0] == 2
            assert attention_mask.shape[0] == 2
            # Both should have same sequence length (padded)
            assert input_ids.shape[1] == attention_mask.shape[1]
        finally:
            await scheduler.stop()
            executor.shutdown(wait=False)
