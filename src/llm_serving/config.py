"""Application configuration via environment variables.

Uses pydantic-settings with LLM_ prefix for 12-factor config.
All settings can be overridden via env vars or a .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """LLM serving platform configuration.

    All fields are read from environment variables with the ``LLM_`` prefix.
    For example, ``LLM_MODEL_NAME`` maps to ``model_name``.

    Attributes:
        model_name: HuggingFace model identifier to load.
        device: PyTorch device for inference (``cpu`` or ``cuda``).
        max_new_tokens: Maximum number of tokens to generate per request.
        host: Host address to bind the server to.
        port: Port number to bind the server to.
        log_level: Logging level (debug, info, warning, error, critical).
    """

    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cpu"
    max_new_tokens: int = 256
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    model_config = {
        "env_prefix": "LLM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings.

    Returns:
        Settings: The application configuration singleton.
    """
    return Settings()
