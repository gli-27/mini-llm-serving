"""Watermark-based memory level monitoring."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

from llm_serving.logging import get_logger
from llm_serving.profiler.tracker import MemoryTracker

logger = get_logger(__name__)


class WatermarkLevel(StrEnum):
    """Memory pressure levels.

    NORMAL: Below high watermark — accept all requests.
    HIGH: Between high and critical — trigger KV cache eviction.
    CRITICAL: Above critical — reject new requests.
    """

    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class WatermarkMonitor:
    """Monitors memory utilization against watermark thresholds.

    Compares current utilization from MemoryTracker against configured
    high/critical ratios and fires callbacks on level transitions.

    Args:
        tracker: The MemoryTracker instance.
        high_watermark_ratio: Utilization triggering HIGH level.
        critical_watermark_ratio: Utilization triggering CRITICAL level.
    """

    def __init__(
        self,
        tracker: MemoryTracker,
        high_watermark_ratio: float = 0.85,
        critical_watermark_ratio: float = 0.95,
    ) -> None:
        self._tracker = tracker
        self._high_ratio = high_watermark_ratio
        self._critical_ratio = critical_watermark_ratio
        self._current_level = WatermarkLevel.NORMAL
        self._callbacks: dict[WatermarkLevel, list[Callable[[], None]]] = {
            WatermarkLevel.NORMAL: [],
            WatermarkLevel.HIGH: [],
            WatermarkLevel.CRITICAL: [],
        }

    @property
    def current_level(self) -> WatermarkLevel:
        """Current watermark level."""
        return self._current_level

    def register_callback(self, level: WatermarkLevel, fn: Callable[[], None]) -> None:
        """Register a callback to fire when a level is entered.

        Args:
            level: The watermark level to trigger on.
            fn: Callable to invoke on transition to this level.
        """
        self._callbacks[level].append(fn)

    def check(self) -> WatermarkLevel:
        """Check current memory utilization and update level.

        Fires callbacks if the level transitions.

        Returns:
            The current WatermarkLevel.
        """
        snapshot = self._tracker.snapshot()
        utilization = snapshot.utilization

        if utilization >= self._critical_ratio:
            new_level = WatermarkLevel.CRITICAL
        elif utilization >= self._high_ratio:
            new_level = WatermarkLevel.HIGH
        else:
            new_level = WatermarkLevel.NORMAL

        if new_level != self._current_level:
            old_level = self._current_level
            self._current_level = new_level
            logger.info(
                "Watermark level transition: %s → %s (utilization=%.2f)",
                old_level.value,
                new_level.value,
                utilization,
            )
            # Fire callbacks for the new level
            for cb in self._callbacks[new_level]:
                cb()

        return self._current_level

    def get_status(self) -> dict[str, object]:
        """Return current watermark status for API response."""
        snapshot = self._tracker.snapshot()
        return {
            "level": self._current_level.value,
            "utilization": round(snapshot.utilization, 4),
            "high_watermark_ratio": self._high_ratio,
            "critical_watermark_ratio": self._critical_ratio,
            "allocated_bytes": snapshot.allocated_bytes,
            "total_bytes": snapshot.total_bytes,
        }
