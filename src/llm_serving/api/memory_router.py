"""Memory profiling API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from llm_serving.profiler.tracker import MemoryTracker
from llm_serving.profiler.watermark import WatermarkMonitor

memory_router = APIRouter(prefix="/v1/memory", tags=["memory"])


@memory_router.get("/stats")
async def memory_stats(request: Request) -> dict:
    """Return current memory utilization breakdown.

    Returns allocated/reserved/total bytes, utilization ratio,
    and manual allocation breakdown by tag.
    """
    tracker: MemoryTracker | None = getattr(request.app.state, "memory_tracker", None)
    if tracker is None:
        return {"status": "disabled"}
    snapshot = tracker.snapshot()
    return snapshot.to_dict()


@memory_router.get("/watermark")
async def watermark_status(request: Request) -> dict:
    """Return current watermark level and thresholds."""
    monitor: WatermarkMonitor | None = getattr(request.app.state, "watermark_monitor", None)
    if monitor is None:
        return {"status": "disabled"}
    return monitor.get_status()


@memory_router.post("/gc")
async def trigger_gc(request: Request) -> dict:
    """Trigger garbage collection and CUDA cache clearing."""
    tracker: MemoryTracker | None = getattr(request.app.state, "memory_tracker", None)
    if tracker is None:
        return {"status": "disabled"}
    tracker.force_gc()
    snapshot = tracker.snapshot()
    return {
        "status": "gc_complete",
        "allocated_bytes_after": snapshot.allocated_bytes,
        "utilization_after": round(snapshot.utilization, 4),
    }
