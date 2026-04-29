"""Shared test fixtures for the LLM serving test suite."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from llm_serving.config import Settings
from llm_serving.models.loader import ModelManager


@pytest.fixture
def settings() -> Settings:
    """Create test settings with fast defaults."""
    return Settings(
        app_env="testing",
        model_name="test-model",
        device="cpu",
        max_new_tokens=32,
        generation_timeout_s=10.0,
        max_concurrent_requests=1,
        log_level="debug",
    )


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Create a mock tokenizer that behaves like a HuggingFace tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token_id = 0

    # tokenizer(prompt, return_tensors="pt") returns {"input_ids": tensor}
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    tokenizer.return_value = {"input_ids": input_ids}

    # tokenizer.decode() returns generated text
    tokenizer.decode.return_value = "Hello, world!"

    return tokenizer


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model that behaves like a HuggingFace causal LM."""
    model = MagicMock()

    # model.generate() returns output_ids tensor (prompt + generated)
    output_ids = torch.tensor([[1, 2, 3, 4, 5, 10, 11, 12]])
    model.generate.return_value = output_ids

    # model.parameters() for param count
    param = MagicMock()
    param.numel.return_value = 1000
    model.parameters.return_value = [param]

    return model


@pytest.fixture
def model_manager(
    settings: Settings, mock_tokenizer: MagicMock, mock_model: MagicMock
) -> ModelManager:
    """Create a ModelManager with mocked model and tokenizer."""
    manager = ModelManager(settings)
    manager.tokenizer = mock_tokenizer
    manager.model = mock_model
    return manager


@pytest.fixture
def unloaded_model_manager(settings: Settings) -> ModelManager:
    """Create a ModelManager that has NOT loaded a model."""
    return ModelManager(settings)


@pytest.fixture
def app(model_manager: ModelManager):
    """Create a FastAPI test app with mocked model manager and executor."""
    from concurrent.futures import ThreadPoolExecutor

    from llm_serving.main import app as fastapi_app

    fastapi_app.state.model_manager = model_manager
    fastapi_app.state.inference_executor = ThreadPoolExecutor(max_workers=1)

    return fastapi_app
