"""Tests for the KV-Cache manager."""

import threading

import numpy as np

from llm_serving.core.kv_cache import (
    KVCacheManager,
    _compute_prefix_hash,
    _estimate_memory_bytes,
)


def _make_fake_kv(num_layers: int = 2, seq_len: int = 10, hidden: int = 64) -> tuple:
    """Create fake past_key_values for testing.

    Returns a tuple of (num_layers) tuples, each containing
    (key_array, value_array) with known .nbytes.
    """
    layers = []
    for _ in range(num_layers):
        # Use numpy arrays (they have .nbytes attribute like torch tensors)
        key = np.zeros((1, 8, seq_len, hidden), dtype=np.float32)
        value = np.zeros((1, 8, seq_len, hidden), dtype=np.float32)
        layers.append((key, value))
    return tuple(layers)


class TestHelpers:
    """Tests for helper functions."""

    def test_compute_prefix_hash_deterministic(self) -> None:
        """Same token_ids should produce same hash."""
        ids = [1, 2, 3, 4, 5]
        h1 = _compute_prefix_hash(ids)
        h2 = _compute_prefix_hash(ids)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex = 64 chars

    def test_compute_prefix_hash_different_for_different_inputs(self) -> None:
        """Different token_ids should produce different hashes."""
        h1 = _compute_prefix_hash([1, 2, 3])
        h2 = _compute_prefix_hash([1, 2, 4])
        assert h1 != h2

    def test_estimate_memory_bytes(self) -> None:
        """Memory estimation should sum all tensor nbytes."""
        kv = _make_fake_kv(num_layers=2, seq_len=10, hidden=64)
        mem = _estimate_memory_bytes(kv)
        # 2 layers × 2 tensors × (1 × 8 × 10 × 64) × 4 bytes = 81920
        expected = 2 * 2 * (1 * 8 * 10 * 64) * 4
        assert mem == expected

    def test_estimate_memory_empty(self) -> None:
        """Empty past_key_values should be 0 bytes."""
        assert _estimate_memory_bytes(()) == 0


class TestKVCacheManager:
    """Tests for KVCacheManager."""

    def test_get_miss_returns_none(self) -> None:
        """Cache miss should return None."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=10)
        result = cache.get([1, 2, 3])
        assert result is None

    def test_put_and_get_hit(self) -> None:
        """Put then get should return the cached entry."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=10)
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)

        cache.put([1, 2, 3], kv)
        entry = cache.get([1, 2, 3])

        assert entry is not None
        assert entry.prefix_length == 3
        assert entry.past_key_values is kv

    def test_stats_tracks_hits_and_misses(self) -> None:
        """Stats should track hit and miss counts."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=10)
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)

        cache.get([1, 2, 3])  # miss
        cache.put([1, 2, 3], kv)
        cache.get([1, 2, 3])  # hit
        cache.get([4, 5, 6])  # miss

        stats = cache.stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 2
        assert stats["num_entries"] == 1

    def test_lru_eviction_on_max_entries(self) -> None:
        """Should evict LRU when max_entries is reached."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=3)
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)

        cache.put([1], kv)
        cache.put([2], kv)
        cache.put([3], kv)

        # Adding 4th should evict [1] (oldest/LRU)
        cache.put([4], kv)

        assert cache.get([1]) is None  # evicted
        assert cache.get([2]) is not None
        assert cache.get([3]) is not None
        assert cache.get([4]) is not None

        stats = cache.stats()
        assert stats["eviction_count"] == 1

    def test_lru_eviction_on_memory_budget(self) -> None:
        """Should evict LRU when memory budget is exceeded."""
        kv_small = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)
        entry_size = _estimate_memory_bytes(kv_small)

        # Budget for exactly 2 entries
        cache = KVCacheManager(max_memory_bytes=entry_size * 2, max_entries=100)

        cache.put([1], kv_small)
        cache.put([2], kv_small)

        # Third entry should evict first (LRU)
        cache.put([3], kv_small)

        assert cache.get([1]) is None  # evicted
        assert cache.get([2]) is not None
        assert cache.get([3]) is not None

    def test_lru_order_updated_on_access(self) -> None:
        """Accessing an entry should move it to most-recently-used."""
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)
        entry_size = _estimate_memory_bytes(kv)

        cache = KVCacheManager(max_memory_bytes=entry_size * 3, max_entries=3)

        cache.put([1], kv)
        cache.put([2], kv)
        cache.put([3], kv)

        # Access [1] — moves it to most-recently-used
        cache.get([1])

        # Adding [4] should evict [2] (now the LRU), not [1]
        cache.put([4], kv)

        assert cache.get([1]) is not None  # accessed recently, kept
        assert cache.get([2]) is None  # evicted (was LRU)
        assert cache.get([3]) is not None
        assert cache.get([4]) is not None

    def test_put_oversized_entry_skipped(self) -> None:
        """Entry larger than entire budget should be silently skipped."""
        cache = KVCacheManager(max_memory_bytes=100, max_entries=10)
        kv_large = _make_fake_kv(num_layers=4, seq_len=100, hidden=128)

        # This entry is way larger than 100 bytes
        cache.put([1, 2, 3], kv_large)

        assert cache.get([1, 2, 3]) is None
        assert cache.stats()["num_entries"] == 0

    def test_put_duplicate_key_updates(self) -> None:
        """Putting same key twice should update the entry."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=10)
        kv1 = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)
        kv2 = _make_fake_kv(num_layers=1, seq_len=10, hidden=16)

        cache.put([1, 2], kv1)
        cache.put([1, 2], kv2)

        entry = cache.get([1, 2])
        assert entry is not None
        assert entry.past_key_values is kv2  # Updated
        assert cache.stats()["num_entries"] == 1  # Still just 1 entry

    def test_clear_removes_all_entries(self) -> None:
        """clear() should remove all entries and reset memory."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=10)
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)

        cache.put([1], kv)
        cache.put([2], kv)
        cache.put([3], kv)

        cache.clear()

        assert cache.get([1]) is None
        stats = cache.stats()
        assert stats["num_entries"] == 0
        assert stats["current_memory_bytes"] == 0

    def test_evict_lru_public_method(self) -> None:
        """evict_lru() should free at least the requested bytes."""
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)
        entry_size = _estimate_memory_bytes(kv)

        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=100)
        cache.put([1], kv)
        cache.put([2], kv)
        cache.put([3], kv)

        # Evict enough for 2 entries
        cache.evict_lru(entry_size * 2)

        stats = cache.stats()
        assert stats["num_entries"] == 1  # Only 1 left
        assert stats["eviction_count"] == 2

    def test_thread_safety(self) -> None:
        """Concurrent puts and gets should not crash."""
        cache = KVCacheManager(max_memory_bytes=10_000_000, max_entries=50)
        kv = _make_fake_kv(num_layers=1, seq_len=5, hidden=16)

        errors: list[Exception] = []

        def _writer(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    cache.put([i], kv)
            except Exception as e:
                errors.append(e)

        def _reader(start: int) -> None:
            try:
                for i in range(start, start + 20):
                    cache.get([i])
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=_writer, args=(0,)),
            threading.Thread(target=_writer, args=(20,)),
            threading.Thread(target=_reader, args=(0,)),
            threading.Thread(target=_reader, args=(10,)),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"


class TestGenerateWithCache:
    """Tests for generate_with_cache() integration."""

    def test_cache_disabled_falls_back_to_generate(self) -> None:
        """When kv_cache_manager=None, should delegate to standard generate()."""
        from unittest.mock import MagicMock, patch

        from llm_serving.core.inference import generate_with_cache

        with patch("llm_serving.core.inference.generate") as mock_gen:
            mock_gen.return_value = ("result", 5, 10)

            result = generate_with_cache(
                model_manager=MagicMock(is_loaded=True),
                prompt="Hello",
                max_new_tokens=10,
                kv_cache_manager=None,
                cache_prefix_tokens=5,
            )

            assert result == ("result", 5, 10)
            mock_gen.assert_called_once()

    def test_no_prefix_tokens_falls_back(self) -> None:
        """When cache_prefix_tokens=None, should delegate to standard generate()."""
        from unittest.mock import MagicMock, patch

        from llm_serving.core.inference import generate_with_cache

        cache = KVCacheManager(max_memory_bytes=10_000_000)

        with patch("llm_serving.core.inference.generate") as mock_gen:
            mock_gen.return_value = ("result", 5, 10)

            result = generate_with_cache(
                model_manager=MagicMock(is_loaded=True),
                prompt="Hello",
                max_new_tokens=10,
                kv_cache_manager=cache,
                cache_prefix_tokens=None,
            )

            assert result == ("result", 5, 10)
            mock_gen.assert_called_once()
