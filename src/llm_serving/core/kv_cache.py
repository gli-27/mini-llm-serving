"""Prefix KV-Cache manager for prompt caching.

Stores past_key_values from previous model forward passes so that
repeated prefixes (system prompts, conversation history) can skip
recomputation. Uses an ``OrderedDict``-based LRU eviction with a
memory budget.

Design decisions:
    - Hash: SHA-256 of ``str(token_ids)`` for deterministic prefix matching.
    - Memory: sum of ``.nbytes`` of all tensors in past_key_values.
    - Thread safety: ``threading.Lock`` (accessed from executor threads).
    - LRU: ``OrderedDict.move_to_end()`` on access, ``popitem(last=False)``
      for eviction (stdlib, no dependency, O(1) operations).

Usage::

    cache = KVCacheManager(max_memory_bytes=1_000_000_000, max_entries=100)

    # On inference:
    entry = cache.get(prefix_token_ids)
    if entry:
        output = model.generate(input_ids, past_key_values=entry.past_key_values)
    else:
        output = model.generate(input_ids, use_cache=True)
        cache.put(prefix_token_ids, output.past_key_values)
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from llm_serving.logging import get_logger

logger = get_logger(__name__)


def _compute_prefix_hash(token_ids: list[int]) -> str:
    """Compute SHA-256 hash of a token ID sequence.

    Args:
        token_ids: List of integer token IDs representing the prefix.

    Returns:
        Hex digest string of the SHA-256 hash.
    """
    return hashlib.sha256(str(token_ids).encode()).hexdigest()


def _estimate_memory_bytes(past_key_values: tuple) -> int:
    """Estimate total memory usage of past_key_values tensors.

    past_key_values is typically a tuple of (num_layers) tuples,
    each containing (key_tensor, value_tensor).

    Args:
        past_key_values: The cached key-value tensors from model output.

    Returns:
        Estimated memory in bytes.
    """
    total = 0
    for layer_kv in past_key_values:
        if isinstance(layer_kv, tuple):
            for tensor in layer_kv:
                if hasattr(tensor, "nbytes"):
                    total += tensor.nbytes
                elif hasattr(tensor, "element_size") and hasattr(tensor, "nelement"):
                    total += tensor.element_size() * tensor.nelement()
    return total


@dataclass
class KVCacheEntry:
    """Stores past_key_values for a specific prefix hash.

    Attributes:
        prefix_hash: SHA-256 of the tokenized prefix IDs.
        past_key_values: Cached key-value tensors (num_layers x (key, value)).
        prefix_length: Number of tokens in the cached prefix.
        created_at: Monotonic timestamp when entry was created.
        memory_bytes: Estimated memory usage of this entry.
    """

    prefix_hash: str
    past_key_values: tuple
    prefix_length: int
    created_at: float = field(default_factory=time.monotonic)
    memory_bytes: int = 0


class KVCacheManager:
    """Thread-safe LRU cache for prefix KV states.

    Uses ``OrderedDict`` for O(1) LRU operations:
    - ``move_to_end(key)`` on cache hit (mark as recently used)
    - ``popitem(last=False)`` for eviction (remove least recently used)

    Memory-budget aware: evicts LRU entries until the total cached
    memory is within ``max_memory_bytes``.

    Args:
        max_memory_bytes: Maximum total memory budget for cached KV states.
        max_entries: Maximum number of cache entries.
    """

    def __init__(self, max_memory_bytes: int, max_entries: int = 100) -> None:
        self._max_memory_bytes = max_memory_bytes
        self._max_entries = max_entries
        self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._lock = threading.Lock()

        # Stats
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        self._current_memory_bytes = 0

    def get(self, prefix_token_ids: list[int]) -> KVCacheEntry | None:
        """Look up cached KV state for a token prefix.

        On hit, moves the entry to the end of the OrderedDict (most
        recently used). This is O(1).

        Args:
            prefix_token_ids: The tokenized prefix to look up.

        Returns:
            The cached KVCacheEntry if found, None on miss.
        """
        prefix_hash = _compute_prefix_hash(prefix_token_ids)

        with self._lock:
            entry = self._cache.get(prefix_hash)
            if entry is not None:
                # Move to end = most recently used
                self._cache.move_to_end(prefix_hash)
                self._hit_count += 1
                logger.debug(
                    "KV cache hit",
                    prefix_hash=prefix_hash[:12],
                    prefix_length=entry.prefix_length,
                )
                return entry

            self._miss_count += 1
            return None

    def put(self, prefix_token_ids: list[int], past_key_values: tuple) -> None:
        """Store a KV cache entry for a prefix.

        If the cache is over budget (memory or entries), evicts LRU
        entries (front of OrderedDict) until the new entry fits.

        Args:
            prefix_token_ids: The tokenized prefix.
            past_key_values: The model's cached key-value tensors.
        """
        prefix_hash = _compute_prefix_hash(prefix_token_ids)
        memory_bytes = _estimate_memory_bytes(past_key_values)

        # Don't cache if single entry exceeds entire budget
        if memory_bytes > self._max_memory_bytes:
            logger.warning(
                "KV entry too large to cache",
                prefix_hash=prefix_hash[:12],
                memory_bytes=memory_bytes,
                max_memory_bytes=self._max_memory_bytes,
            )
            return

        entry = KVCacheEntry(
            prefix_hash=prefix_hash,
            past_key_values=past_key_values,
            prefix_length=len(prefix_token_ids),
            created_at=time.monotonic(),
            memory_bytes=memory_bytes,
        )

        with self._lock:
            # If key already exists, remove old entry's memory first
            if prefix_hash in self._cache:
                old_entry = self._cache.pop(prefix_hash)
                self._current_memory_bytes -= old_entry.memory_bytes

            # Evict LRU entries until memory fits
            while (
                self._cache and self._current_memory_bytes + memory_bytes > self._max_memory_bytes
            ):
                self._evict_one_locked()

            # Evict if entry count exceeded
            while len(self._cache) >= self._max_entries:
                self._evict_one_locked()

            # Store at end of OrderedDict (most recently used)
            self._cache[prefix_hash] = entry
            self._current_memory_bytes += memory_bytes

            logger.debug(
                "KV cache put",
                prefix_hash=prefix_hash[:12],
                prefix_length=len(prefix_token_ids),
                memory_bytes=memory_bytes,
                total_memory=self._current_memory_bytes,
                num_entries=len(self._cache),
            )

    def evict_lru(self, needed_bytes: int) -> None:
        """Evict least-recently-used entries until ``needed_bytes`` are freed.

        Args:
            needed_bytes: Minimum bytes to free.
        """
        with self._lock:
            freed = 0
            while self._cache and freed < needed_bytes:
                freed += self._evict_one_locked()

    def _evict_one_locked(self) -> int:
        """Evict the single least-recently-used entry. Lock must be held.

        Uses ``popitem(last=False)`` which removes from the front of the
        OrderedDict (oldest/least recently used). This is O(1).

        Returns:
            Memory bytes freed by the eviction.
        """
        if not self._cache:
            return 0

        prefix_hash, entry = self._cache.popitem(last=False)
        self._current_memory_bytes -= entry.memory_bytes
        self._eviction_count += 1

        logger.debug(
            "KV cache evicted",
            prefix_hash=prefix_hash[:12],
            memory_bytes=entry.memory_bytes,
        )
        return entry.memory_bytes

    def stats(self) -> dict[str, int]:
        """Return cache statistics.

        Returns:
            Dict with hit_count, miss_count, eviction_count,
            current_memory_bytes, and num_entries.
        """
        with self._lock:
            return {
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "eviction_count": self._eviction_count,
                "current_memory_bytes": self._current_memory_bytes,
                "num_entries": len(self._cache),
            }

    def clear(self) -> None:
        """Clear all cache entries. Used on model reload.

        Resets memory tracking but preserves hit/miss/eviction counters.
        """
        with self._lock:
            self._cache.clear()
            self._current_memory_bytes = 0
            logger.info("KV cache cleared")
