"""Global exception handler for consistent error responses.

Maps all custom and unhandled exceptions to OpenAI-compatible JSON
error responses with appropriate HTTP status codes. Ensures clients
always receive a structured error body, even for unexpected failures.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from llm_serving.exceptions import (
    CircuitOpenError,
    GenerationError,
    ModelNotLoadedError,
    QueueFullError,
    RateLimitExceededError,
)
from llm_serving.logging import get_logger

logger = get_logger(__name__)

# Map exception types to (status_code, error_type).
# Unknown exceptions fall through to the default (500, "internal_error").
_EXCEPTION_MAP: dict[type[Exception], tuple[int, str]] = {
    CircuitOpenError: (503, "circuit_open"),
    ModelNotLoadedError: (503, "model_unavailable"),
    QueueFullError: (503, "queue_full"),
    RateLimitExceededError: (429, "rate_limited"),
    GenerationError: (500, "generation_error"),
}


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Unified error response format matching the OpenAI error schema.

    Maps known exception types to specific HTTP status codes and error
    types. Unknown exceptions default to 500 Internal Server Error.

    Logs 5xx errors at ERROR level and 4xx errors at WARNING level.

    Args:
        request: The incoming HTTP request that triggered the exception.
        exc: The unhandled exception.

    Returns:
        A JSONResponse with OpenAI-compatible error body:
        ``{"error": {"message": "...", "type": "...", "code": N}}``
    """
    status_code, error_type = _EXCEPTION_MAP.get(
        type(exc), (500, "internal_error")
    )

    # Log severity based on status code
    if status_code >= 500:
        logger.error(
            "Unhandled exception",
            status_code=status_code,
            error_type=error_type,
            path=request.url.path,
            method=request.method,
            exc_info=exc,
        )
    else:
        logger.warning(
            "Client error",
            status_code=status_code,
            error_type=error_type,
            path=request.url.path,
            method=request.method,
            message=str(exc),
        )

    headers: dict[str, str] = {}
    if status_code == 429:
        headers["Retry-After"] = "1"

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": str(exc),
                "type": error_type,
                "code": status_code,
            }
        },
        headers=headers or None,
    )
