"""Tests for API endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving.models.loader import ModelManager


class TestHealthEndpoint:
    """Tests for GET /health."""

    async def test_health_returns_200_when_loaded(self, app, model_manager: ModelManager) -> None:
        """GET /health should return 200 with model name and redis=true when all healthy."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model"] == "test-model"
        assert data["redis"] is True

    async def test_health_returns_503_when_not_loaded(self, app) -> None:
        """GET /health should return 503 when model is not loaded."""
        app.state.model_manager = ModelManager(
            app.state.model_manager.settings
        )  # Unloaded manager

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 503

    async def test_health_returns_503_when_redis_down(self, app) -> None:
        """GET /health should return 503 when Redis is unreachable."""
        app.state.redis_client.health_check = AsyncMock(return_value=False)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 503
        data = response.json()
        assert data["detail"]["error"]["message"] == "Redis is unreachable"


class TestModelsEndpoint:
    """Tests for GET /v1/models."""

    async def test_list_models_returns_loaded_model(
        self, app, model_manager: ModelManager
    ) -> None:
        """GET /v1/models should list the loaded model."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert len(data["models"]) == 1
        assert data["models"][0]["id"] == "test-model"
        assert data["models"][0]["ready"] is True

    async def test_list_models_shows_not_ready_when_unloaded(self, app) -> None:
        """GET /v1/models should show ready=false when model isn't loaded."""
        app.state.model_manager = ModelManager(app.state.model_manager.settings)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["models"][0]["ready"] is False


class TestCompletionsEndpoint:
    """Tests for POST /v1/completions."""

    async def test_completions_returns_503_when_not_loaded(self, app) -> None:
        """POST /v1/completions should return 503 when model isn't loaded."""
        app.state.model_manager = ModelManager(app.state.model_manager.settings)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )

        assert response.status_code == 503

    @patch("llm_serving.api.router.generate")
    async def test_completions_sync_success(
        self, mock_generate: MagicMock, app
    ) -> None:
        """POST /v1/completions should return generated text for sync requests."""
        mock_generate.return_value = ("Generated text", 5, 10)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 32},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Generated text"
        assert data["object"] == "text_completion"
        assert data["model"] == "test-model"
        assert data["usage"]["prompt_tokens"] == 5
        assert data["usage"]["completion_tokens"] == 10
        assert data["usage"]["total_tokens"] == 15
        assert data["id"].startswith("cmpl-")

    async def test_completions_validates_prompt_required(self, app) -> None:
        """POST /v1/completions should return 422 when prompt is missing."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"max_tokens": 10},
            )

        assert response.status_code == 422

    async def test_completions_validates_prompt_not_empty(self, app) -> None:
        """POST /v1/completions should return 422 for empty prompt."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "", "max_tokens": 10},
            )

        assert response.status_code == 422

    async def test_completions_validates_max_tokens_range(self, app) -> None:
        """POST /v1/completions should reject max_tokens outside 1-2048."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 0},
            )
            assert response.status_code == 422

            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 9999},
            )
            assert response.status_code == 422

    async def test_completions_validates_temperature_range(self, app) -> None:
        """POST /v1/completions should reject temperature outside 0.0-2.0."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "temperature": -0.1},
            )
            assert response.status_code == 422

            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "temperature": 2.1},
            )
            assert response.status_code == 422

    @patch("llm_serving.api.router.generate")
    async def test_completions_timeout_returns_504(
        self, mock_generate: MagicMock, app
    ) -> None:
        """POST /v1/completions should return 504 on generation timeout."""
        import asyncio

        mock_generate.side_effect = asyncio.TimeoutError()

        # Set a very short timeout for testing
        app.state.model_manager.settings.generation_timeout_s = 0.01

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )

        assert response.status_code == 504

    @patch("llm_serving.api.router.generate_stream")
    async def test_completions_streaming_returns_sse(
        self, mock_stream: MagicMock, app
    ) -> None:
        """POST /v1/completions with stream=true should return SSE response."""
        mock_stream.return_value = iter(["Hello", " world"])

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hi", "max_tokens": 10, "stream": True},
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Parse SSE events
        lines = response.text.strip().split("\n\n")
        assert len(lines) >= 2  # At least token events + [DONE]

        # Last event should be [DONE]
        assert lines[-1].strip() == "data: [DONE]"

        # First event should contain "Hello"
        first_data = lines[0].replace("data: ", "")
        chunk = json.loads(first_data)
        assert chunk["content"] == "Hello"
        assert chunk["object"] == "text_completion.chunk"


class TestSchemaValidation:
    """Tests for request/response schema validation."""

    async def test_default_values_applied(self, app) -> None:
        """Defaults should be applied when optional fields are omitted."""
        with patch("llm_serving.api.router.generate") as mock_gen:
            mock_gen.return_value = ("text", 3, 5)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    json={"prompt": "Hello"},
                )

            assert response.status_code == 200
            # Verify defaults were passed
            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs["max_new_tokens"] == 256  # default
            assert call_kwargs.kwargs["temperature"] == 0.7  # default

    async def test_seed_field_accepted(self, app) -> None:
        """The seed field should be accepted and passed through."""
        with patch("llm_serving.api.router.generate") as mock_gen:
            mock_gen.return_value = ("text", 3, 5)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    json={"prompt": "Hello", "seed": 42},
                )

            assert response.status_code == 200
            call_kwargs = mock_gen.call_args
            assert call_kwargs.kwargs["seed"] == 42
