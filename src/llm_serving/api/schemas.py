"""Pydantic request/response models for the LLM serving API."""

from typing import Literal

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Request body for the /v1/completions endpoint.

    Attributes:
        prompt: The input text prompt for generation. Must be 1–4096 characters.
        max_tokens: Maximum number of tokens to generate. Must be 1–2048.
        temperature: Sampling temperature. Must be 0.0–2.0.
        stream: Whether to stream the response token-by-token via SSE.
    """

    prompt: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Input text prompt for generation.",
    )
    max_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    stream: bool = Field(
        default=False,
        description="Stream the response token-by-token via SSE.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible generation. "
        "If None, results are non-deterministic.",
    )
    priority: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Request priority: 1=critical, 2=standard, 3=batch.",
    )


class UsageInfo(BaseModel):
    """Token usage statistics for a completion response.

    Attributes:
        prompt_tokens: Number of tokens in the input prompt.
        completion_tokens: Number of tokens in the generated completion.
        total_tokens: Total tokens (prompt + completion).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Response body for the /v1/completions endpoint.

    Attributes:
        id: Unique completion identifier (cmpl-<8hex>).
        object: Object type, always "text_completion".
        model: Name of the model used for generation.
        content: The generated text content.
        usage: Token usage statistics.
    """

    id: str
    object: str = "text_completion"
    model: str
    content: str
    usage: UsageInfo


class HealthResponse(BaseModel):
    """Response body for the /health endpoint.

    Attributes:
        status: Health status: "healthy", "degraded", or "unhealthy".
        model: Name of the loaded model.
        redis: Whether the Redis connection is healthy.
        circuit_breaker: Current circuit breaker state.
        queue_depth: Number of pending requests in the inference queue.
    """

    status: Literal["healthy", "degraded", "unhealthy"]
    model: str
    redis: bool = False
    circuit_breaker: str = "closed"
    queue_depth: int = 0


class ModelInfo(BaseModel):
    """Information about a loaded model.

    Attributes:
        id: Model identifier (HuggingFace model name).
        ready: Whether the model is loaded and ready for inference.
    """

    id: str
    ready: bool


class StreamChunk(BaseModel):
    """A single SSE chunk in a streaming completion response.

    Attributes:
        id: Unique completion identifier (shared across all chunks).
        object: Object type, always "text_completion.chunk".
        model: Name of the model used for generation.
        content: The token text for this chunk.
    """

    id: str
    object: str = "text_completion.chunk"
    model: str
    content: str


class ModelsResponse(BaseModel):
    """Response body for the /v1/models endpoint.

    Attributes:
        models: List of available models with their status.
    """

    models: list[ModelInfo]
