"""API routes for the LLM serving platform."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status

from llm_serving.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    UsageInfo,
)
from llm_serving.core.inference import generate
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
            detail="Model not loaded",
        )
    return HealthResponse(
        status="healthy",
        model=model_manager.settings.model_name,
    )


@router.post(
    "/v1/completions",
    response_model=CompletionResponse,
    status_code=status.HTTP_200_OK,
)
async def create_completion(
    request: CompletionRequest,
    model_manager: ModelManager = Depends(get_model_manager),
) -> CompletionResponse:
    """Generate a text completion from a prompt.

    Args:
        request: The completion request with prompt and generation parameters.
        model_manager: The injected ModelManager dependency.

    Returns:
        A CompletionResponse with the generated text and usage statistics.

    Raises:
        HTTPException: 503 if model not loaded, 500 on inference failure.
    """
    try:
        generated_text, prompt_tokens, completion_tokens = generate(
            model_manager=model_manager,
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
        )
    except ModelNotLoadedError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
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
