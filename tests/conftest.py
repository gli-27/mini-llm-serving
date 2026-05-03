"""Shared test fixtures for the LLM serving test suite."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from llm_serving.config import Settings
from llm_serving.models.loader import ModelManager
from llm_serving.queue.redis_client import RedisClient


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
def mock_redis_client() -> MagicMock:
    """Create a mock RedisClient with async health_check returning True."""
    client = MagicMock(spec=RedisClient)
    client.health_check = AsyncMock(return_value=True)
    client.connect = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_rate_limiter() -> MagicMock:
    """Create a mock TokenBucketRateLimiter that allows all requests."""
    from llm_serving.queue.rate_limiter import TokenBucketRateLimiter

    limiter = MagicMock(spec=TokenBucketRateLimiter)
    limiter.try_consume = AsyncMock(return_value=(True, {"remaining": 9.0, "limit": 10.0, "retry_after": 0.0}))
    return limiter


@pytest.fixture
def mock_priority_queue() -> MagicMock:
    """Create a mock PriorityQueue with zero depth."""
    from llm_serving.queue.priority_queue import PriorityQueue

    queue = MagicMock(spec=PriorityQueue)
    queue.queue_depth = AsyncMock(return_value=0)
    queue.enqueue = AsyncMock(return_value=1)
    queue.dequeue = AsyncMock(return_value=None)
    return queue


@pytest.fixture
def mock_worker_pool() -> MagicMock:
    """Create a mock InferenceWorkerPool that resolves futures immediately."""
    import asyncio
    from llm_serving.core.worker import InferenceWorkerPool

    pool = MagicMock(spec=InferenceWorkerPool)

    def _register_request(request_id: str) -> asyncio.Future:
        """Return a Future that will be resolved by the test's generate mock."""
        loop = asyncio.get_event_loop()
        future: asyncio.Future = loop.create_future()
        # Store it so we can resolve it from the generate mock
        pool._test_futures[request_id] = future
        return future

    pool._test_futures = {}
    pool.register_request = MagicMock(side_effect=_register_request)
    pool.cancel_request = MagicMock()
    pool.start = AsyncMock()
    pool.stop = AsyncMock()
    return pool


@pytest.fixture
def app(
    model_manager: ModelManager,
    mock_redis_client: MagicMock,
    mock_rate_limiter: MagicMock,
    mock_priority_queue: MagicMock,
    mock_worker_pool: MagicMock,
    settings: Settings,
):
    """Create a FastAPI test app with mocked dependencies."""
    from concurrent.futures import ThreadPoolExecutor

    from llm_serving.main import app as fastapi_app

    fastapi_app.state.model_manager = model_manager
    fastapi_app.state.inference_executor = ThreadPoolExecutor(max_workers=1)
    fastapi_app.state.redis_client = mock_redis_client
    fastapi_app.state.rate_limiter = mock_rate_limiter
    fastapi_app.state.priority_queue = mock_priority_queue
    fastapi_app.state.worker_pool = mock_worker_pool
    fastapi_app.state.settings = settings

    return fastapi_app
