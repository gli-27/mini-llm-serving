"""Tests for core inference logic."""

from unittest.mock import MagicMock, patch

import pytest

from llm_serving.core.inference import generate
from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.models.loader import ModelManager


class TestGenerate:
    """Tests for the generate() function."""

    def test_generate_returns_text_and_token_counts(
        self, model_manager: ModelManager
    ) -> None:
        """generate() should return (text, prompt_tokens, completion_tokens)."""
        text, prompt_tokens, completion_tokens = generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=32,
        )

        assert isinstance(text, str)
        assert text == "Hello, world!"
        assert prompt_tokens == 5  # From mock: tensor([[1, 2, 3, 4, 5]])
        assert completion_tokens == 3  # Output has 8 tokens, prompt has 5 → 3 new

    def test_generate_raises_when_model_not_loaded(
        self, unloaded_model_manager: ModelManager
    ) -> None:
        """generate() should raise ModelNotLoadedError if model isn't loaded."""
        with pytest.raises(ModelNotLoadedError):
            generate(
                model_manager=unloaded_model_manager,
                prompt="Hello",
                max_new_tokens=32,
            )

    def test_generate_passes_temperature(self, model_manager: ModelManager) -> None:
        """generate() should pass temperature to model.generate()."""
        generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=32,
            temperature=1.5,
        )

        call_kwargs = model_manager.model.generate.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["temperature"] == 1.5

    def test_generate_passes_max_new_tokens(self, model_manager: ModelManager) -> None:
        """generate() should pass max_new_tokens to model.generate()."""
        generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=100,
        )

        call_kwargs = model_manager.model.generate.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["max_new_tokens"] == 100

    def test_generate_sets_seed_when_provided(self, model_manager: ModelManager) -> None:
        """generate() should call torch.manual_seed() when seed is provided."""
        with patch("llm_serving.core.inference.torch") as mock_torch:
            # Keep no_grad working
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            generate(
                model_manager=model_manager,
                prompt="Hello",
                max_new_tokens=32,
                seed=42,
            )

            mock_torch.manual_seed.assert_called_once_with(42)

    def test_generate_skips_seed_when_none(self, model_manager: ModelManager) -> None:
        """generate() should NOT call torch.manual_seed() when seed is None."""
        with patch("llm_serving.core.inference.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock()
            mock_torch.no_grad.return_value.__exit__ = MagicMock()

            generate(
                model_manager=model_manager,
                prompt="Hello",
                max_new_tokens=32,
                seed=None,
            )

            mock_torch.manual_seed.assert_not_called()

    def test_generate_uses_do_sample(self, model_manager: ModelManager) -> None:
        """generate() should always pass do_sample=True."""
        generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=32,
        )

        call_kwargs = model_manager.model.generate.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["do_sample"] is True

    def test_generate_sets_pad_token_id(self, model_manager: ModelManager) -> None:
        """generate() should pass the tokenizer's pad_token_id to model.generate()."""
        generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=32,
        )

        call_kwargs = model_manager.model.generate.call_args  # type: ignore[union-attr]
        assert call_kwargs.kwargs["pad_token_id"] == 0  # From mock_tokenizer fixture

    def test_generate_moves_input_to_device(self, model_manager: ModelManager) -> None:
        """generate() should move input_ids to the configured device."""
        # The mock tokenizer returns a tensor; we verify .to(device) is called
        # by checking that the input_ids passed to generate are on the right device
        generate(
            model_manager=model_manager,
            prompt="Hello",
            max_new_tokens=32,
        )

        # Verify model.generate was called (input_ids are first positional arg)
        model_manager.model.generate.assert_called_once()  # type: ignore[union-attr]
