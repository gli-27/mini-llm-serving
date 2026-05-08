"""Memory pressure handler — eviction logic and request rejection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_serving.logging import get_logger
from llm_serving.profiler.watermark import WatermarkLevel, WatermarkMonitor

if TYPE_CHECKING:
    from llm_serving.core.kv_cache import KVCacheManager

logger = get_logger(__name__)


class MemoryPressureHandler:
    """Handles memory pressure by evicting KV cache and rejecting requests.

    On HIGH: evict LRU KV cache entries with exponential escalation
    (10% → 25% → 50% of cache).
    On CRITICAL: set rejecting=True, emergency evict all.
    On NORMAL: reset escalation and rejecting flag.

    Args:
        monitor: The WatermarkMonitor instance.
        kv_cache: Optional KVCacheManager for eviction.
    """

    def __init__(
        self,
        monitor: WatermarkMonitor,
        kv_cache: KVCacheManager | None = None,
    ) -> None:
        self._monitor = monitor
        self._kv_cache = kv_cache
        self._rejecting = False
        self._escalation_step = 0
        self._escalation_ratios = [0.10, 0.25, 0.50]

        # Register callbacks
        monitor.register_callback(WatermarkLevel.HIGH, self._on_high)
        monitor.register_callback(WatermarkLevel.CRITICAL, self._on_critical)
        monitor.register_callback(WatermarkLevel.NORMAL, self._on_normal)

    @property
    def is_rejecting(self) -> bool:
        """Whether the handler is rejecting new requests."""
        return self._rejecting

    def should_admit(self) -> bool:
        """Check if a new request should be admitted.

        Returns:
            True if the request can proceed, False if memory pressure
            requires rejection.
        """
        # Re-check watermark level
        self._monitor.check()
        return not self._rejecting

    def _on_high(self) -> None:
        """Handle HIGH watermark: evict KV cache entries."""
        if self._kv_cache is None:
            return

        ratio = self._escalation_ratios[
            min(self._escalation_step, len(self._escalation_ratios) - 1)
        ]
        stats = self._kv_cache.stats()
        current_memory = stats["current_memory_bytes"]
        evict_bytes = int(current_memory * ratio)

        if evict_bytes > 0:
            self._kv_cache.evict_lru(evict_bytes)
            logger.warning(
                "HIGH pressure: evicted %d bytes (%.0f%% of cache)",
                evict_bytes,
                ratio * 100,
            )

        self._escalation_step += 1

    def _on_critical(self) -> None:
        """Handle CRITICAL watermark: reject requests + emergency evict."""
        self._rejecting = True
        logger.error("CRITICAL pressure: rejecting new requests")

        if self._kv_cache is not None:
            self._kv_cache.clear()
            logger.error("CRITICAL: emergency cleared entire KV cache")

    def _on_normal(self) -> None:
        """Handle return to NORMAL: reset all pressure state."""
        self._rejecting = False
        self._escalation_step = 0
        logger.info("Memory pressure returned to NORMAL")
