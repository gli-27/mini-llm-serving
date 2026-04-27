"""Centralized exception classes for the LLM serving platform.

All custom exceptions live here so they can be imported from a single
location across the codebase.
"""


class ModelNotLoadedError(RuntimeError):
    """Raised when inference is attempted before the model is loaded."""


class GenerationTimeoutError(TimeoutError):
    """Raised when model.generate() exceeds the configured timeout."""
