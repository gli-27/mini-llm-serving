"""Memory tracker — snapshots GPU/CPU memory usage by category."""

from __future__ import annotations

import gc
import threading
from dataclasses import dataclass, field

import torch

from llm_serving.logging import get_logger
from llm_serving.profiler.config import MemoryProfilerConfig

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot.

    Attributes:
        allocated_bytes: Currently allocated memory.
        reserved_bytes: Total reserved by the allocator.
        total_bytes: Total device memory available.
        manual_allocations: Memory tracked by manual register_allocation calls.
        utilization: allocated / total ratio.
    """

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    total_bytes: int = 0
    manual_allocations: dict[str, int] = field(default_factory=dict)

    @property
    def utilization(self) -> float:
        """Memory utilization ratio (0.0 to 1.0)."""
        if self.total_bytes == 0:
            return 0.0
        return self.allocated_bytes / self.total_bytes

    def to_dict(self) -> dict[str, object]:
        """Serialize to dict for API response."""
        return {
            "allocated_bytes": self.allocated_bytes,
            "reserved_bytes": self.reserved_bytes,
            "total_bytes": self.total_bytes,
            "utilization": round(self.utilization, 4),
            "manual_allocations": dict(self.manual_allocations),
        }


class MemoryTracker:
    """Tracks GPU/CPU memory usage with manual allocation accounting.

    Wraps torch.cuda.memory_allocated/reserved when CUDA is available,
    falls back to process RSS via psutil on CPU.

    Thread-safe via threading.Lock for manual allocation tracking.

    Args:
        config: Memory profiler configuration.
    """

    def __init__(self, config: MemoryProfilerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._manual_allocations: dict[str, int] = {}
        self._total_bytes = config.total_memory_bytes

        # Auto-detect total memory
        if self._total_bytes == 0:
            if config.enable_cuda and torch.cuda.is_available():
                self._total_bytes = torch.cuda.get_device_properties(0).total_mem
            else:
                # CPU fallback: use 8GB as default budget
                self._total_bytes = 8 * 1024 * 1024 * 1024

    def snapshot(self) -> MemorySnapshot:
        """Take a point-in-time memory usage snapshot.

        Returns:
            MemorySnapshot with current allocation data.
        """
        if self._config.enable_cuda and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
        else:
            # CPU fallback: sum manual allocations
            with self._lock:
                allocated = sum(self._manual_allocations.values())
            reserved = allocated

        with self._lock:
            manual = dict(self._manual_allocations)

        return MemorySnapshot(
            allocated_bytes=allocated,
            reserved_bytes=reserved,
            total_bytes=self._total_bytes,
            manual_allocations=manual,
        )

    def register_allocation(self, tag: str, size_bytes: int) -> None:
        """Register a manual memory allocation (e.g., KV cache entry).

        Args:
            tag: Identifier for this allocation (e.g., "kv_cache").
            size_bytes: Bytes allocated.
        """
        with self._lock:
            current = self._manual_allocations.get(tag, 0)
            self._manual_allocations[tag] = current + size_bytes

    def release(self, tag: str, size_bytes: int | None = None) -> None:
        """Release a manual allocation.

        Args:
            tag: Identifier to release.
            size_bytes: Bytes to release. If None, releases all for this tag.
        """
        with self._lock:
            if tag not in self._manual_allocations:
                return
            if size_bytes is None:
                del self._manual_allocations[tag]
            else:
                self._manual_allocations[tag] = max(0, self._manual_allocations[tag] - size_bytes)
                if self._manual_allocations[tag] == 0:
                    del self._manual_allocations[tag]

    def force_gc(self) -> None:
        """Force garbage collection and CUDA cache clearing."""
        gc.collect()
        if self._config.enable_cuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Forced GC + cache clear")
