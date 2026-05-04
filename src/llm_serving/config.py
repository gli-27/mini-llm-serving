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
        app_env: Application environment (development, staging, production).
        model_name: HuggingFace model identifier to load.
        device: PyTorch device for inference (``cpu`` or ``cuda``).
        max_new_tokens: Maximum number of tokens to generate per request.
        generation_timeout_s: Max seconds for a single generation before timeout.
        max_concurrent_requests: Max concurrent inference requests (prevents GPU OOM).
        host: Host address to bind the server to.
        port: Port number to bind the server to.
        log_level: Logging level (debug, info, warning, error, critical).
        redis_url: Redis connection URL for queues and rate limiting.
        rate_limit_bucket_size: Max tokens in the rate limit bucket per API key.
        rate_limit_refill_rate: Tokens added per second to the rate limit bucket.
        max_queue_depth: Max queue depth before load shedding triggers (503).
        circuit_breaker_failure_threshold: Consecutive failures before circuit trips.
        circuit_breaker_recovery_timeout_s: Seconds before OPEN → HALF_OPEN probe.
        max_batch_size: Max requests per inference batch.
        max_batch_wait_ms: Max milliseconds to wait before flushing a partial batch.
        batching_enabled: Whether to enable dynamic batching.
        kv_cache_max_memory_mb: Max memory (MB) for KV prefix cache.
        kv_cache_max_entries: Max number of cached prefix entries.
        kv_cache_enabled: Whether to enable prefix KV caching.
        kv_cache_prefix_tokens: Tokens to cache as prefix (0=auto-detect).
    """

    app_env: str = "development"
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device: str = "cpu"
    max_new_tokens: int = 256
    generation_timeout_s: float = 60.0
    max_concurrent_requests: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_bucket_size: int = 10
    rate_limit_refill_rate: float = 2.0
    max_queue_depth: int = 100
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_s: float = 30.0
    max_batch_size: int = 8
    max_batch_wait_ms: int = 50
    batching_enabled: bool = True
    kv_cache_max_memory_mb: int = 512
    kv_cache_max_entries: int = 100
    kv_cache_enabled: bool = True
    kv_cache_prefix_tokens: int = 0

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
