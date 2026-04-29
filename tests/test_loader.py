"""Tests for ModelManager lifecycle management."""

from unittest.mock import MagicMock, patch

import pytest

from llm_serving.config import Settings
from llm_serving.models.loader import ModelManager


class TestModelManager:
    """Tests for ModelManager initialization and state tracking."""

    def test_initial_state_not_loaded(self, settings: Settings) -> None:
        """A fresh ModelManager should report is_loaded=False."""
        manager = ModelManager(settings)
        assert manager.is_loaded is False
        assert manager.model is None
        assert manager.tokenizer is None

    def test_is_loaded_with_both(self, model_manager: ModelManager) -> None:
        """is_loaded should be True when both model and tokenizer are set."""
        assert model_manager.is_loaded is True

    def test_is_loaded_with_only_model(self, settings: Settings, mock_model: MagicMock) -> None:
        """is_loaded should be False if only model is set (no tokenizer)."""
        manager = ModelManager(settings)
        manager.model = mock_model
        assert manager.is_loaded is False

    def test_is_loaded_with_only_tokenizer(
        self, settings: Settings, mock_tokenizer: MagicMock
    ) -> None:
        """is_loaded should be False if only tokenizer is set (no model)."""
        manager = ModelManager(settings)
        manager.tokenizer = mock_tokenizer
        assert manager.is_loaded is False

    def test_settings_stored(self, settings: Settings) -> None:
        """ModelManager should store the provided settings."""
        manager = ModelManager(settings)
        assert manager.settings is settings
        assert manager.settings.model_name == "test-model"


class TestModelLoading:
    """Tests for the load_model() method."""

    @patch("llm_serving.models.loader.AutoModelForCausalLM")
    @patch("llm_serving.models.loader.AutoTokenizer")
    def test_load_model_success(
        self, mock_auto_tokenizer: MagicMock, mock_auto_model: MagicMock, settings: Settings
    ) -> None:
        """load_model() should load tokenizer and model from HuggingFace."""
        # Setup mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Setup mock model
        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        mock_model.parameters.return_value = [param]
        mock_auto_model.from_pretrained.return_value = mock_model

        manager = ModelManager(settings)
        manager.load_model()

        assert manager.is_loaded is True
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model")
        mock_auto_model.from_pretrained.assert_called_once_with("test-model")
        mock_model.to.assert_called_once_with("cpu")
        mock_model.eval.assert_called_once()

    @patch("llm_serving.models.loader.AutoModelForCausalLM")
    @patch("llm_serving.models.loader.AutoTokenizer")
    def test_load_model_sets_pad_token_when_missing(
        self, mock_auto_tokenizer: MagicMock, mock_auto_model: MagicMock, settings: Settings
    ) -> None:
        """load_model() should set pad_token to eos_token when pad_token is None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        param = MagicMock()
        param.numel.return_value = 1000
        mock_model.parameters.return_value = [param]
        mock_auto_model.from_pretrained.return_value = mock_model

        manager = ModelManager(settings)
        manager.load_model()

        assert mock_tokenizer.pad_token == "</s>"

    @patch("llm_serving.models.loader.AutoModelForCausalLM")
    @patch("llm_serving.models.loader.AutoTokenizer")
    def test_load_model_failure_propagates(
        self, mock_auto_tokenizer: MagicMock, mock_auto_model: MagicMock, settings: Settings
    ) -> None:
        """load_model() should propagate exceptions from HuggingFace."""
        mock_auto_tokenizer.from_pretrained.side_effect = OSError("Model not found")

        manager = ModelManager(settings)
        with pytest.raises(OSError, match="Model not found"):
            manager.load_model()

        assert manager.is_loaded is False
