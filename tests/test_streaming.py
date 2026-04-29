"""Tests for SSE streaming inference."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_serving.core.streaming import _generate_in_thread, generate_stream
from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.models.loader import ModelManager


class TestGenerateStream:
    """Tests for the generate_stream() generator."""

    def test_raises_when_model_not_loaded(
        self, unloaded_model_manager: ModelManager
    ) -> None:
        """generate_stream() should raise ModelNotLoadedError if model isn't loaded."""
        with pytest.raises(ModelNotLoadedError):
            # Must consume the generator to trigger the check
            list(
                generate_stream(
                    model_manager=unloaded_model_manager,
                    prompt="Hello",
                    max_new_tokens=32,
                )
            )

    @patch("llm_serving.core.streaming.TextIteratorStreamer")
    @patch("llm_serving.core.streaming.threading.Thread")
    def test_stream_yields_tokens(
        self,
        mock_thread_cls: MagicMock,
        mock_streamer_cls: MagicMock,
        model_manager: ModelManager,
    ) -> None:
        """generate_stream() should yield tokens from the TextIteratorStreamer."""
        # Mock streamer to yield specific tokens
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter(["Hello", ", ", "world", "!"]))
        mock_streamer_cls.return_value = mock_streamer

        # Mock thread to start and join without actually running
        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        tokens = list(
            generate_stream(
                model_manager=model_manager,
                prompt="Test",
                max_new_tokens=32,
            )
        )

        assert tokens == ["Hello", ", ", "world", "!"]
        mock_thread.start.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=5.0)

    @patch("llm_serving.core.streaming.TextIteratorStreamer")
    @patch("llm_serving.core.streaming.threading.Thread")
    def test_stream_skips_empty_tokens(
        self,
        mock_thread_cls: MagicMock,
        mock_streamer_cls: MagicMock,
        model_manager: ModelManager,
    ) -> None:
        """generate_stream() should skip empty string tokens."""
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter(["Hello", "", "world"]))
        mock_streamer_cls.return_value = mock_streamer

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        tokens = list(
            generate_stream(
                model_manager=model_manager,
                prompt="Test",
                max_new_tokens=32,
            )
        )

        assert tokens == ["Hello", "world"]

    @patch("llm_serving.core.streaming.TextIteratorStreamer")
    @patch("llm_serving.core.streaming.threading.Thread")
    def test_stream_passes_seed_to_thread(
        self,
        mock_thread_cls: MagicMock,
        mock_streamer_cls: MagicMock,
        model_manager: ModelManager,
    ) -> None:
        """generate_stream() should pass seed to the background thread."""
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter([]))
        mock_streamer_cls.return_value = mock_streamer

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        list(
            generate_stream(
                model_manager=model_manager,
                prompt="Test",
                max_new_tokens=32,
                seed=42,
            )
        )

        # Check that Thread was created with seed=42 in args
        thread_call_kwargs = mock_thread_cls.call_args
        args = thread_call_kwargs.kwargs["args"]
        # args should be (model, generation_kwargs, errors, seed)
        assert args[3] == 42

    @patch("llm_serving.core.streaming.TextIteratorStreamer")
    @patch("llm_serving.core.streaming.threading.Thread")
    def test_stream_passes_none_seed_to_thread(
        self,
        mock_thread_cls: MagicMock,
        mock_streamer_cls: MagicMock,
        model_manager: ModelManager,
    ) -> None:
        """generate_stream() should pass None seed when not provided."""
        mock_streamer = MagicMock()
        mock_streamer.__iter__ = MagicMock(return_value=iter([]))
        mock_streamer_cls.return_value = mock_streamer

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        list(
            generate_stream(
                model_manager=model_manager,
                prompt="Test",
                max_new_tokens=32,
            )
        )

        thread_call_kwargs = mock_thread_cls.call_args
        args = thread_call_kwargs.kwargs["args"]
        assert args[3] is None


class TestGenerateInThread:
    """Tests for the _generate_in_thread() helper."""

    def test_calls_model_generate(self) -> None:
        """_generate_in_thread() should call model.generate() with provided kwargs."""
        mock_model = MagicMock()
        gen_kwargs = {"input_ids": torch.tensor([[1, 2, 3]]), "max_new_tokens": 10}
        errors: list[BaseException] = []

        _generate_in_thread(mock_model, gen_kwargs, errors)

        mock_model.generate.assert_called_once_with(**gen_kwargs)
        assert len(errors) == 0

    def test_captures_exception(self) -> None:
        """_generate_in_thread() should capture exceptions into the errors list."""
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("CUDA OOM")
        errors: list[BaseException] = []

        _generate_in_thread(mock_model, {}, errors)

        assert len(errors) == 1
        assert "CUDA OOM" in str(errors[0])

    def test_sets_seed_when_provided(self) -> None:
        """_generate_in_thread() should call torch.manual_seed() when seed is given."""
        mock_model = MagicMock()
        errors: list[BaseException] = []

        with patch("llm_serving.core.streaming.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            _generate_in_thread(mock_model, {}, errors, seed=42)

            mock_torch.manual_seed.assert_called_once_with(42)

    def test_skips_seed_when_none(self) -> None:
        """_generate_in_thread() should NOT call torch.manual_seed() when seed is None."""
        mock_model = MagicMock()
        errors: list[BaseException] = []

        with patch("llm_serving.core.streaming.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            _generate_in_thread(mock_model, {}, errors, seed=None)

            mock_torch.manual_seed.assert_not_called()
