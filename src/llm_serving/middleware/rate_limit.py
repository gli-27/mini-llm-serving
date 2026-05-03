"""Rate limiting middleware using token bucket algorithm.

Extracts the API key from the Authorization header (Bearer token),
calls the TokenBucketRateLimiter, and returns 429 with Retry-After
if the request is denied. Sets rate limit response headers on all
responses.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that enforces per-API-key rate limiting.

    Retrieves the ``TokenBucketRateLimiter`` from ``request.app.state``
    at request time (initialized during lifespan startup).

    Extracts the API key from ``Authorization: Bearer <key>`` header.
    If no header is present, defaults to "default" (shared bucket for
    unauthenticated requests).

    On rate limit exceeded, returns 429 with:
    - ``Retry-After`` header (seconds until a token is available)
    - JSON error body in OpenAI-compatible format

    On all responses, sets:
    - ``X-RateLimit-Remaining``: Tokens remaining in the bucket
    - ``X-RateLimit-Limit``: Maximum bucket size
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check rate limit before passing the request through.

        Args:
            request: The incoming HTTP request.
            call_next: Callable to invoke the next middleware/route handler.

        Returns:
            The response, either 429 (rate limited) or the normal response
            with rate limit headers attached.
        """
        rate_limiter = request.app.state.rate_limiter
        api_key = self._extract_api_key(request)
        allowed, info = await rate_limiter.try_consume(api_key)

        if not allowed:
            retry_after = info["retry_after"]
            logger.warning(
                "Rate limited request",
                api_key=api_key,
                retry_after=retry_after,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "type": "rate_limit_error",
                        "code": 429,
                    }
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Limit": str(int(info["limit"])),
                },
            )

        # Request is allowed — forward to the next handler
        response = await call_next(request)

        # Attach rate limit headers to the response
        response.headers["X-RateLimit-Remaining"] = str(int(info["remaining"]))
        response.headers["X-RateLimit-Limit"] = str(int(info["limit"]))

        return response

    @staticmethod
    def _extract_api_key(request: Request) -> str:
        """Extract the API key from the Authorization header.

        Expects ``Authorization: Bearer <api_key>``. If the header is
        missing or malformed, returns "default" as the fallback key
        (all unauthenticated requests share a single bucket).

        Args:
            request: The incoming HTTP request.

        Returns:
            The extracted API key string, or "default".
        """
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:].strip()
        return "default"
