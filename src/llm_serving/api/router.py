"""API routes for the LLM serving platform."""

import json
import logging
import uuid
from collections.abc import AsyncGenerator

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
from llm_serving.core.inference import generate
from llm_serving.core.streaming import generate_stream
from llm_serving.models.loader import ModelManager, ModelNotLoadedError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_model_manager(request: Request) -> ModelManager:
    """Dependency that retrieves the ModelManager from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The shared ModelManager instance.
    """
    return request.app.state.model_manager  # type: ignore[no-any-return]


@router.get("/health", response_model=HealthResponse)
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager),
) -> HealthResponse:
    """Health check endpoint.

    Returns 200 with model name if healthy, 503 if model is not loaded.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Model not loaded", "type": "server_error", "code": 503}},
        )
    return HealthResponse(
        status="healthy",
        model=model_manager.settings.model_name,
    )


@router.post("/v1/completions", response_model=None, status_code=status.HTTP_200_OK)
async def create_completion(
    body: CompletionRequest,
    request: Request,
    model_manager: ModelManager = Depends(get_model_manager),
) -> CompletionResponse | StreamingResponse:
    """Generate a text completion from a prompt.

    If ``stream=False`` (default), returns a full CompletionResponse.
    If ``stream=True``, returns a StreamingResponse with SSE-formatted
    token chunks followed by a ``data: [DONE]`` sentinel.

    Args:
        body: The completion request with prompt and generation parameters.
        request: The raw FastAPI request (for disconnect detection).
        model_manager: The injected ModelManager dependency.

    Returns:
        CompletionResponse for sync, StreamingResponse for streaming.

    Raises:
        HTTPException: 503 if model not loaded, 500 on inference failure.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Model not loaded", "type": "server_error", "code": 503}},
        )

    if body.stream:
        return _create_streaming_response(body, request, model_manager)

    return _create_sync_response(body, model_manager)


def _create_sync_response(
    body: CompletionRequest,
    model_manager: ModelManager,
) -> CompletionResponse:
    """Handle a synchronous (non-streaming) completion request.

    Args:
        body: The validated completion request.
        model_manager: The loaded ModelManager.

    Returns:
        A CompletionResponse with generated text and usage stats.
    """
    try:
        generated_text, prompt_tokens, completion_tokens = generate(
            model_manager=model_manager,
            prompt=body.prompt,
            max_new_tokens=body.max_tokens,
        )
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": {"message": "Model not loaded", "type": "server_error", "code": 503}},
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": {"message": "Inference failed", "type": "server_error", "code": 500}},
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
) -> StreamingResponse:
    """Handle a streaming SSE completion request.

    Args:
        body: The validated completion request.
        request: The raw FastAPI request for disconnect detection.
        model_manager: The loaded ModelManager.

    Returns:
        A StreamingResponse with SSE-formatted token chunks.
    """
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    model_name = model_manager.settings.model_name

    async def event_generator() -> AsyncGenerator[str, None]:
        """Yield SSE-formatted events: one per token, then [DONE]."""
        try:
            token_stream = generate_stream(
                model_manager=model_manager,
                prompt=body.prompt,
                max_new_tokens=body.max_tokens,
            )

            for token_text in token_stream:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.info("Client disconnected during streaming for %s", completion_id)
                    return

                chunk = StreamChunk(
                    id=completion_id,
                    model=model_name,
                    content=token_text,
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        except ModelNotLoadedError:
            error_data = json.dumps(
                {"error": {"message": "Model not loaded", "type": "server_error", "code": 503}}
            )
            yield f"data: {error_data}\n\n"
        except Exception as exc:
            logger.exception("Streaming inference failed: %s", exc)
            error_data = json.dumps(
                {"error": {"message": "Inference failed", "type": "server_error", "code": 500}}
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
