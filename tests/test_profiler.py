"""Tests for GPU memory profiling and budget management."""

from llm_serving.profiler.config import MemoryProfilerConfig
from llm_serving.profiler.memory_estimator import MemoryEstimator, ModelMemoryConfig
from llm_serving.profiler.pressure import MemoryPressureHandler
from llm_serving.profiler.tracker import MemorySnapshot, MemoryTracker
from llm_serving.profiler.watermark import WatermarkLevel, WatermarkMonitor


class TestMemorySnapshot:
    """Tests for MemorySnapshot."""

    def test_utilization_calculation(self) -> None:
        """Utilization should be allocated/total."""
        snap = MemorySnapshot(allocated_bytes=500, total_bytes=1000)
        assert snap.utilization == 0.5

    def test_utilization_zero_total(self) -> None:
        """Utilization should be 0 when total is 0."""
        snap = MemorySnapshot(allocated_bytes=0, total_bytes=0)
        assert snap.utilization == 0.0

    def test_to_dict(self) -> None:
        """to_dict should contain all fields."""
        snap = MemorySnapshot(
            allocated_bytes=100,
            reserved_bytes=200,
            total_bytes=1000,
            manual_allocations={"kv_cache": 50},
        )
        d = snap.to_dict()
        assert d["allocated_bytes"] == 100
        assert d["utilization"] == 0.1
        assert d["manual_allocations"] == {"kv_cache": 50}


class TestMemoryTracker:
    """Tests for MemoryTracker (CPU fallback mode)."""

    def test_snapshot_initial_zero(self) -> None:
        """Initial snapshot should have 0 allocated."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        snap = tracker.snapshot()
        assert snap.allocated_bytes == 0
        assert snap.total_bytes == 1000

    def test_register_and_snapshot(self) -> None:
        """Registered allocations should appear in snapshot."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)

        tracker.register_allocation("kv_cache", 200)
        tracker.register_allocation("activations", 100)

        snap = tracker.snapshot()
        assert snap.allocated_bytes == 300
        assert snap.manual_allocations == {"kv_cache": 200, "activations": 100}

    def test_release_partial(self) -> None:
        """Partial release should decrement allocation."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)

        tracker.register_allocation("kv_cache", 500)
        tracker.release("kv_cache", 200)

        snap = tracker.snapshot()
        assert snap.allocated_bytes == 300

    def test_release_all(self) -> None:
        """Release with None should remove entire allocation."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)

        tracker.register_allocation("kv_cache", 500)
        tracker.release("kv_cache")

        snap = tracker.snapshot()
        assert snap.allocated_bytes == 0
        assert "kv_cache" not in snap.manual_allocations

    def test_force_gc_no_error(self) -> None:
        """force_gc should not raise on CPU."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        tracker.force_gc()  # Should not crash


class TestWatermarkMonitor:
    """Tests for WatermarkMonitor transitions."""

    def test_normal_level_below_threshold(self) -> None:
        """Below high watermark should be NORMAL."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )

        level = monitor.check()
        assert level == WatermarkLevel.NORMAL

    def test_high_level_transition(self) -> None:
        """Between high and critical should be HIGH."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        tracker.register_allocation("test", 900)  # 90% utilization

        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )
        level = monitor.check()
        assert level == WatermarkLevel.HIGH

    def test_critical_level_transition(self) -> None:
        """Above critical should be CRITICAL."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        tracker.register_allocation("test", 960)  # 96% utilization

        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )
        level = monitor.check()
        assert level == WatermarkLevel.CRITICAL

    def test_callback_fires_on_transition(self) -> None:
        """Callbacks should fire when level changes."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )

        fired: list[str] = []
        monitor.register_callback(WatermarkLevel.HIGH, lambda: fired.append("high"))

        tracker.register_allocation("test", 900)
        monitor.check()

        assert "high" in fired


class TestMemoryPressureHandler:
    """Tests for MemoryPressureHandler admission control."""

    def test_admits_when_normal(self) -> None:
        """Should admit requests when at NORMAL level."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )
        handler = MemoryPressureHandler(monitor)

        assert handler.should_admit() is True

    def test_rejects_when_critical(self) -> None:
        """Should reject requests when at CRITICAL level."""
        config = MemoryProfilerConfig(enable_cuda=False, total_memory_bytes=1000)
        tracker = MemoryTracker(config)
        tracker.register_allocation("test", 960)

        monitor = WatermarkMonitor(
            tracker, high_watermark_ratio=0.85, critical_watermark_ratio=0.95
        )
        handler = MemoryPressureHandler(monitor)

        # First check triggers transition → sets rejecting
        assert handler.should_admit() is False


class TestMemoryEstimator:
    """Tests for memory estimation formulas."""

    def test_kv_cache_formula(self) -> None:
        """KV cache: 2 * layers * hidden * seq_len * batch * dtype."""
        config = ModelMemoryConfig(num_layers=22, hidden_dim=2048, dtype_bytes=2)
        estimator = MemoryEstimator(config)

        # seq_len=100, batch=1
        expected = 2 * 22 * 2048 * 100 * 1 * 2
        assert estimator.estimate_kv_cache_bytes(100, 1) == expected

    def test_activation_formula(self) -> None:
        """Activation: 4 * batch * seq_len * hidden * dtype."""
        config = ModelMemoryConfig(hidden_dim=2048, dtype_bytes=2)
        estimator = MemoryEstimator(config)

        expected = 4 * 1 * 100 * 2048 * 2
        assert estimator.estimate_activation_bytes(100, 1) == expected

    def test_can_fit_true(self) -> None:
        """Should return True when request fits."""
        estimator = MemoryEstimator(ModelMemoryConfig(num_layers=2, hidden_dim=64, dtype_bytes=2))
        # Small model — should fit in 1MB
        assert estimator.can_fit(seq_len=10, available_bytes=1_000_000)

    def test_can_fit_false(self) -> None:
        """Should return False when request doesn't fit."""
        estimator = MemoryEstimator(
            ModelMemoryConfig(num_layers=22, hidden_dim=2048, dtype_bytes=2)
        )
        # TinyLlama at seq_len=1000 needs ~180MB — won't fit in 1KB
        assert estimator.can_fit(seq_len=1000, available_bytes=1000) is False
