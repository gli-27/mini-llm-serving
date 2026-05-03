"""Tests for rate limiting and load shedding middleware."""

from unittest.mock import AsyncMock

from httpx import ASGITransport, AsyncClient


class TestRateLimitMiddleware:
    """Tests for the rate limiting middleware."""

    async def test_allowed_request_passes_through(self, app) -> None:
        """Requests within rate limit should pass through with headers."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Limit" in response.headers
        assert response.headers["X-RateLimit-Remaining"] == "9"
        assert response.headers["X-RateLimit-Limit"] == "10"

    async def test_rate_limited_returns_429(self, app) -> None:
        """Requests exceeding rate limit should get 429 response."""
        # Override rate limiter to deny
        app.state.rate_limiter.try_consume = AsyncMock(
            return_value=(False, {"remaining": 0.0, "limit": 10.0, "retry_after": 2.5})
        )

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 429
        data = response.json()
        assert data["error"]["message"] == "Rate limit exceeded"
        assert data["error"]["type"] == "rate_limit_error"
        assert data["error"]["code"] == 429
        assert response.headers["Retry-After"] == "3"  # int(2.5) + 1
        assert response.headers["X-RateLimit-Remaining"] == "0"
        assert response.headers["X-RateLimit-Limit"] == "10"

    async def test_extracts_bearer_api_key(self, app) -> None:
        """Middleware should extract API key from Authorization header."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/health",
                headers={"Authorization": "Bearer my-api-key-123"},
            )

        assert response.status_code == 200
        # Verify rate limiter was called with the correct key
        app.state.rate_limiter.try_consume.assert_awaited_with("my-api-key-123")

    async def test_missing_auth_uses_default_key(self, app) -> None:
        """Requests without Authorization header should use 'default' key."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        app.state.rate_limiter.try_consume.assert_awaited_with("default")

    async def test_malformed_auth_uses_default_key(self, app) -> None:
        """Non-Bearer auth headers should fall back to 'default' key."""
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get(
                "/health",
                headers={"Authorization": "Basic dXNlcjpwYXNz"},
            )

        assert response.status_code == 200
        app.state.rate_limiter.try_consume.assert_awaited_with("default")


class TestLoadShedderMiddleware:
    """Tests for the load shedding middleware."""

    async def test_normal_load_passes_through(self, app) -> None:
        """Requests under queue depth limit should pass through."""
        app.state.priority_queue.queue_depth = AsyncMock(return_value=5)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 200

    async def test_overloaded_returns_503(self, app) -> None:
        """Requests when queue depth >= max should get 503."""
        # Set queue depth to exceed max (default=100 in test settings)
        app.state.priority_queue.queue_depth = AsyncMock(return_value=100)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 503
        data = response.json()
        assert data["error"]["type"] == "overloaded_error"
        assert data["error"]["code"] == 503
        assert "overloaded" in data["error"]["message"].lower()
        assert response.headers["Retry-After"] == "5"

    async def test_health_bypasses_load_shedding(self, app) -> None:
        """GET /health should bypass load shedding even when overloaded."""
        app.state.priority_queue.queue_depth = AsyncMock(return_value=999)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        # Should get 200 (healthy) not 503 (overloaded)
        assert response.status_code == 200

    async def test_docs_bypasses_load_shedding(self, app) -> None:
        """GET /docs should bypass load shedding."""
        app.state.priority_queue.queue_depth = AsyncMock(return_value=999)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/docs")

        # /docs returns 200 (FastAPI auto-generates it)
        assert response.status_code == 200

    async def test_completions_subject_to_load_shedding(self, app) -> None:
        """POST /v1/completions should be subject to load shedding."""
        app.state.priority_queue.queue_depth = AsyncMock(return_value=200)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/v1/completions",
                json={"prompt": "Hello", "max_tokens": 10},
            )

        assert response.status_code == 503

    async def test_at_threshold_sheds_load(self, app) -> None:
        """When queue depth equals max_queue_depth, load shedding triggers."""
        # Default max_queue_depth is 100
        app.state.settings.max_queue_depth = 50
        app.state.priority_queue.queue_depth = AsyncMock(return_value=50)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 503

    async def test_just_under_threshold_allows(self, app) -> None:
        """When queue depth is just below max, requests pass through."""
        app.state.settings.max_queue_depth = 50
        app.state.priority_queue.queue_depth = AsyncMock(return_value=49)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/v1/models")

        assert response.status_code == 200
