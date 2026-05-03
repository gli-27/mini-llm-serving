"""Tests for the global exception handler."""

from fastapi import Request

from llm_serving.exceptions import (
    CircuitOpenError,
    GenerationError,
    ModelNotLoadedError,
    QueueFullError,
    RateLimitExceededError,
)
from llm_serving.middleware.error_handler import (
    _EXCEPTION_MAP,
    global_exception_handler,
)


class TestExceptionMap:
    """Tests for the exception → (status_code, error_type) mapping."""

    def test_circuit_open_maps_to_503(self) -> None:
        """CircuitOpenError should map to 503 circuit_open."""
        status_code, error_type = _EXCEPTION_MAP[CircuitOpenError]
        assert status_code == 503
        assert error_type == "circuit_open"

    def test_model_not_loaded_maps_to_503(self) -> None:
        """ModelNotLoadedError should map to 503 model_unavailable."""
        status_code, error_type = _EXCEPTION_MAP[ModelNotLoadedError]
        assert status_code == 503
        assert error_type == "model_unavailable"

    def test_queue_full_maps_to_503(self) -> None:
        """QueueFullError should map to 503 queue_full."""
        status_code, error_type = _EXCEPTION_MAP[QueueFullError]
        assert status_code == 503
        assert error_type == "queue_full"

    def test_rate_limit_exceeded_maps_to_429(self) -> None:
        """RateLimitExceededError should map to 429 rate_limited."""
        status_code, error_type = _EXCEPTION_MAP[RateLimitExceededError]
        assert status_code == 429
        assert error_type == "rate_limited"

    def test_generation_error_maps_to_500(self) -> None:
        """GenerationError should map to 500 generation_error."""
        status_code, error_type = _EXCEPTION_MAP[GenerationError]
        assert status_code == 500
        assert error_type == "generation_error"

    def test_unknown_exception_defaults_to_500(self) -> None:
        """Unknown exceptions should default to (500, 'internal_error')."""
        status_code, error_type = _EXCEPTION_MAP.get(
            ValueError, (500, "internal_error")
        )
        assert status_code == 500
        assert error_type == "internal_error"


class TestGlobalExceptionHandler:
    """Tests for the global_exception_handler function directly."""

    async def test_known_exception_returns_correct_status(self) -> None:
        """Known exceptions should return their mapped status code."""

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        response = await global_exception_handler(
            request, CircuitOpenError("Circuit is open")
        )

        assert response.status_code == 503
        assert response.body is not None

    async def test_rate_limit_has_retry_after(self) -> None:
        """429 responses should include Retry-After header."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        response = await global_exception_handler(
            request, RateLimitExceededError("Too many requests")
        )

        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "1"

    async def test_non_429_no_retry_after(self) -> None:
        """Non-429 responses should not have Retry-After header."""
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        response = await global_exception_handler(
            request, GenerationError("fail")
        )

        assert response.status_code == 500
        # Retry-After should not be in response headers
        assert response.headers.get("Retry-After") is None

    async def test_response_body_openai_format(self) -> None:
        """Response body should match OpenAI error format."""
        import json

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/completions",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        response = await global_exception_handler(
            request, ModelNotLoadedError("Model not loaded")
        )

        body = json.loads(response.body)
        assert "error" in body
        assert body["error"]["message"] == "Model not loaded"
        assert body["error"]["type"] == "model_unavailable"
        assert body["error"]["code"] == 503

    async def test_unknown_exception_returns_internal_error(self) -> None:
        """Unknown exceptions should return 500 internal_error."""
        import json

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "query_string": b"",
            "headers": [],
        }
        request = Request(scope)

        response = await global_exception_handler(
            request, ValueError("unexpected")
        )

        assert response.status_code == 500
        body = json.loads(response.body)
        assert body["error"]["type"] == "internal_error"
        assert body["error"]["message"] == "unexpected"
