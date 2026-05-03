"""Centralized exception classes for the LLM serving platform.

All custom exceptions live here so they can be imported from a single
location across the codebase. The global error handler maps these to
appropriate HTTP status codes and error types.
"""


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is loaded."""


class GenerationTimeoutError(TimeoutError):
    """Raised when model.generate() exceeds the configured timeout."""


class GenerationError(RuntimeError):
    """Raised when model inference fails for any reason."""


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is OPEN and requests are rejected."""


class QueueFullError(RuntimeError):
    """Raised when the inference queue is full (load shedding)."""


class RateLimitExceededError(RuntimeError):
    """Raised when a client exceeds their rate limit."""
