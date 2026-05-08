"""Configuration for memory profiling and budget management."""

from pydantic_settings import BaseSettings


class MemoryProfilerConfig(BaseSettings):
    """Memory profiler configuration.

    Attributes:
        enabled: Whether memory profiling is active.
        high_watermark_ratio: Utilization ratio triggering KV cache eviction.
        critical_watermark_ratio: Utilization ratio triggering request rejection.
        total_memory_bytes: Total GPU memory budget (0=auto-detect from CUDA).
        enable_cuda: Whether to use CUDA memory APIs (False=CPU fallback).
    """

    enabled: bool = True
    high_watermark_ratio: float = 0.85
    critical_watermark_ratio: float = 0.95
    total_memory_bytes: int = 0  # 0 = auto-detect
    enable_cuda: bool = False  # Default False for CPU-only environments

    model_config = {
        "env_prefix": "LLM_MEM_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }
