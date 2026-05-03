"""Load shedding middleware based on priority queue depth.

Monitors the inference priority queue depth and rejects requests with
503 Service Unavailable when the queue exceeds a configurable maximum.
This prevents unbounded memory growth and maintains responsiveness
under heavy load.

Bypass paths (health checks, docs) are not subject to load shedding.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from llm_serving.logging import get_logger

logger = get_logger(__name__)

# Paths that bypass load shedding — these must always be reachable
# even when the system is overloaded (for health checks and docs).
_BYPASS_PATHS = frozenset({"/health", "/docs", "/openapi.json", "/redoc"})


class LoadShedderMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that rejects requests when the queue is too deep.

    Retrieves the ``PriorityQueue`` and ``max_queue_depth`` from
    ``request.app.state`` at request time (initialized during lifespan).

    Checks the priority queue depth before each request. If the depth
    exceeds ``max_queue_depth``, returns 503 with a ``Retry-After``
    header to signal the client to back off.

    Health and documentation endpoints are bypassed to ensure
    observability remains available during overload.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Check queue depth and shed load if overloaded.

        Args:
            request: The incoming HTTP request.
            call_next: Callable to invoke the next middleware/route handler.

        Returns:
            The response, either 503 (overloaded) or the normal response.
        """
        # Bypass load shedding for health/docs endpoints
        if request.url.path in _BYPASS_PATHS:
            return await call_next(request)

        priority_queue = request.app.state.priority_queue
        max_queue_depth = request.app.state.settings.max_queue_depth
        depth = await priority_queue.queue_depth()

        if depth >= max_queue_depth:
            logger.warning(
                "Load shedding: queue depth exceeded",
                queue_depth=depth,
                max_queue_depth=max_queue_depth,
                path=request.url.path,
            )
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "message": "Server overloaded — too many pending requests",
                        "type": "overloaded_error",
                        "code": 503,
                    }
                },
                headers={
                    "Retry-After": "5",
                },
            )

        return await call_next(request)
