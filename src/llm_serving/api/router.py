"""API routes for the LLM serving platform."""

import asyncio
import functools
import json
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from llm_serving.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    StreamChunk,
    UsageInfo,
)
from llm_serving.core.circuit_breaker import CircuitBreaker, CircuitState
from llm_serving.core.inference import generate
from llm_serving.core.streaming import generate_stream
from llm_serving.core.worker import InferenceWorkerPool
from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager
from llm_serving.queue.priority_queue import PriorityQueue
from llm_serving.queue.redis_client import RedisClient

logger = get_logger(__name__)

router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    """Dependency that retrieves the ModelManager from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared ModelManager instance.
    """
    return request.app.state.model_manager  # type: ignore[no-any-return]


def get_inference_executor(request: Request) -> ThreadPoolExecutor:
    """Dependency that retrieves the inference thread pool executor.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared ThreadPoolExecutor for inference concurrency control.
    """
    return request.app.state.inference_executor  # type: ignore[no-any-return]


def get_redis_client(request: Request) -> RedisClient:
    """Dependency that retrieves the RedisClient from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared RedisClient instance.
    """
    return request.app.state.redis_client  # type: ignore[no-any-return]


def get_priority_queue(request: Request) -> PriorityQueue:
    """Dependency that retrieves the PriorityQueue from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared PriorityQueue instance.
    """
    return request.app.state.priority_queue  # type: ignore[no-any-return]


def get_worker_pool(request: Request) -> InferenceWorkerPool:
    """Dependency that retrieves the InferenceWorkerPool from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared InferenceWorkerPool instance.
    """
    return request.app.state.worker_pool  # type: ignore[no-any-return]


def get_circuit_breaker(request: Request) -> CircuitBreaker:
    """Dependency that retrieves the CircuitBreaker from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared CircuitBreaker instance.
    """
    return request.app.state.circuit_breaker  # type: ignore[no-any-return]


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
    redis_client: RedisClient = Depends(get_redis_client),
    priority_queue: PriorityQueue = Depends(get_priority_queue),
    circuit_breaker: CircuitBreaker = Depends(get_circuit_breaker),
) -> HealthResponse:
    """Health check endpoint with 3-state health reporting.

    Returns:
        - "healthy": model loaded + Redis up + circuit closed
        - "degraded": model loaded but Redis down OR circuit half-open
        - "unhealthy": model not loaded OR circuit open (returns 503)
    """
    model_loaded = model_manager.is_loaded
    redis_healthy = await redis_client.health_check()
    cb_state = circuit_breaker.state
    queue_depth = await priority_queue.queue_depth()

    # Determine health status
    if not model_loaded or cb_state == CircuitState.OPEN:
        health_status: str = "unhealthy"
    elif not redis_healthy or cb_state == CircuitState.HALF_OPEN:
        health_status = "degraded"
    else:
        health_status = "healthy"

    response = HealthResponse(
        status=health_status,  # type: ignore[arg-type]
        model=model_manager.settings.model_name,
        redis=redis_healthy,
        circuit_breaker=cb_state.value,
        queue_depth=queue_depth,
    )

    # Return 503 for unhealthy
    if health_status == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=response.model_dump(),
        )

    return response


@router.post("/v1/completions", response_model=None, status_code=status.HTTP_200_OK)
async def create_completion(
    body: CompletionRequest,
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
    executor: ThreadPoolExecutor = Depends(get_inference_executor),
    priority_queue: PriorityQueue = Depends(get_priority_queue),
    worker_pool: InferenceWorkerPool = Depends(get_worker_pool),
) -> CompletionResponse | StreamingResponse:
    """Generate a text completion from a prompt.

    If ``stream=False`` (default), the request is enqueued into the
    priority queue and processed by the background worker pool. The
    handler awaits a Future that the worker resolves after inference.
    CRITICAL requests are dequeued before STANDARD/BATCH.

    If ``stream=True``, returns a StreamingResponse with SSE-formatted
    token chunks (streaming bypasses the queue for now — the worker
    pool handles non-streaming; streaming uses direct executor dispatch).

    Args:
        body: The completion request with prompt and generation parameters.
        request: The raw FastAPI request (for disconnect detection).
        model_manager: The injected ModelManager dependency.
        executor: The inference thread pool executor.
        priority_queue: The priority queue for request scheduling.
        worker_pool: The background inference worker pool.

    Returns:
        CompletionResponse for sync, StreamingResponse for streaming.

    Raises:
        HTTPException: 503 if model not loaded, 504 on timeout, 500 on failure.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Model not loaded", "type": "server_error", "code": 503}},
        )

    # Streaming bypasses the worker pool — uses direct executor dispatch
    # (streaming requires a token-by-token generator, not a Future-based result)
    if body.stream:
        return _create_streaming_response(body, request, model_manager, executor)

    # Non-streaming: enqueue into priority queue → worker dequeues → resolves Future
    request_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    # Register the Future BEFORE enqueue so the worker finds it after ZPOPMIN
    result_future = worker_pool.register_request(request_id)

    queue_depth = await priority_queue.enqueue(
        request_id=request_id,
        priority=body.priority,
        payload={
            "prompt": body.prompt,
            "max_tokens": body.max_tokens,
            "temperature": body.temperature,
            "seed": body.seed,
        },
    )
    logger.info(
        "Request enqueued",
        request_id=request_id,
        priority=body.priority,
        queue_depth=queue_depth,
    )

    # Await the result from the background worker (priority-aware scheduling)
    timeout_s = model_manager.settings.generation_timeout_s
    try:
        generated_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
            result_future,
            timeout=timeout_s,
        )
    except asyncio.CancelledError:
        worker_pool.cancel_request(request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Request cancelled", "type": "server_error", "code": 503}},
        )
    except TimeoutError:
        worker_pool.cancel_request(request_id)
        logger.error("Generation timed out after %.1fs", timeout_s)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": {
                    "message": f"Generation timed out after {timeout_s}s",
                    "type": "timeout_error",
                    "code": 504,
                }
            },
        )
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model not loaded",
                    "type": "server_error",
                    "code": 503,
                }
            },
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "Inference failed",
                    "type": "server_error",
                    "code": 500,
                }
            },
        )

    return CompletionResponse(
        id=request_id,
        model=model_manager.settings.model_name,
        content=generated_text,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


async def _create_sync_response(
    body: CompletionRequest,
    model_manager: ModelManager,
    executor: ThreadPoolExecutor,
) -> CompletionResponse:
    """Handle a synchronous (non-streaming) completion request.

    Runs generate() in the thread pool executor (non-blocking) with
    asyncio.wait_for for timeout control.

    Args:
        body: The validated completion request.
        model_manager: The loaded ModelManager.
        executor: Thread pool for running sync inference.

    Returns:
        A CompletionResponse with generated text and usage stats.
    """
    loop = asyncio.get_event_loop()
    timeout_s = model_manager.settings.generation_timeout_s

    try:
        generated_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                functools.partial(
                    generate,
                    model_manager=model_manager,
                    prompt=body.prompt,
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    seed=body.seed,
                ),
            ),
            timeout=timeout_s,
        )
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Model not loaded",
                    "type": "server_error",
                    "code": 503,
                }
            },
        )
    except TimeoutError:
        logger.error("Generation timed out after %.1fs", timeout_s)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail={
                "error": {
                    "message": f"Generation timed out after {timeout_s}s",
                    "type": "timeout_error",
                    "code": 504,
                }
            },
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "message": "Inference failed",
                    "type": "server_error",
                    "code": 500,
                }
            },
        )

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    return CompletionResponse(
        id=completion_id,
        model=model_manager.settings.model_name,
        content=generated_text,
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _create_streaming_response(
    body: CompletionRequest,
    request: Request,
    model_manager: ModelManager,
    executor: ThreadPoolExecutor,
) -> StreamingResponse:
    """Handle a streaming SSE completion request.

    Token generation runs in the executor thread pool. Tokens are passed
    to the async event_generator via an asyncio.Queue, so the event loop
    is never blocked.

    Args:
        body: The validated completion request.
        request: The raw FastAPI request for disconnect detection.
        model_manager: The loaded ModelManager.
        executor: Thread pool for running sync inference.

    Returns:
        A StreamingResponse with SSE-formatted token chunks.
    """
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    model_name = model_manager.settings.model_name

    async def event_generator() -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events: one per token, then [DONE].

        Uses an asyncio.Queue as a bridge between the sync token generator
        (running in the executor) and the async SSE response.
        """
        token_queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _produce_tokens() -> None:
            """Run in executor thread: generate tokens and enqueue them."""
            try:
                for token_text in generate_stream(
                    model_manager=model_manager,
                    prompt=body.prompt,
                    max_new_tokens=body.max_tokens,
                    temperature=body.temperature,
                    seed=body.seed,
                ):
                    if token_text:
                        loop.call_soon_threadsafe(token_queue.put_nowait, token_text)
            except Exception as exc:
                logger.exception("Streaming generation failed: %s", exc)
                # Send error as a special token
                error_data = json.dumps(
                    {"error": {"message": str(exc), "type": "server_error", "code": 500}}
                )
                loop.call_soon_threadsafe(token_queue.put_nowait, f"__ERROR__:{error_data}")
            finally:
                # None sentinel signals end of stream
                loop.call_soon_threadsafe(token_queue.put_nowait, None)

        # Submit token production to the executor
        executor.submit(_produce_tokens)

        # Consume tokens from the async queue (non-blocking to event loop)
        timeout_s = model_manager.settings.generation_timeout_s
        try:
            while True:
                try:
                    token = await asyncio.wait_for(token_queue.get(), timeout=timeout_s)
                except TimeoutError:
                    logger.error(
                        "Streaming timed out after %.1fs for %s",
                        timeout_s,
                        completion_id,
                    )
                    error_data = json.dumps(
                        {
                            "error": {
                                "message": "Generation timed out",
                                "type": "timeout_error",
                                "code": 504,
                            }
                        }
                    )
                    yield f"data: {error_data}\n\n"
                    break

                if token is None:
                    break

                # Check for error sentinel
                if token.startswith("__ERROR__:"):
                    yield f"data: {token[len('__ERROR__:') :]}\n\n"
                    break

                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected during streaming for %s", completion_id)
                    return

                chunk = StreamChunk(
                    id=completion_id,
                    model=model_name,
                    content=token,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception as exc:
            logger.exception("SSE event generator failed: %s", exc)
            error_data = json.dumps(
                {"error": {"message": "Streaming failed", "type": "server_error", "code": 500}}
            )
            yield f"data: {error_data}\n\n"

        # Sentinel to signal end of stream
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/v1/models", response_model=ModelsResponse)
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager),
) -> ModelsResponse:
    """List available models and their readiness status."""
    return ModelsResponse(
        models=[
            ModelInfo(
                id=model_manager.settings.model_name,
                ready=model_manager.is_loaded,
            )
        ]
    )
