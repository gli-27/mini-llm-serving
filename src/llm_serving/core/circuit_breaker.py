"""Circuit breaker for model inference protection.

Implements the Circuit Breaker pattern with three states:
    - CLOSED: Normal operation — requests pass through.
    - OPEN: Tripped after consecutive failures — fail fast, don't hit model.
    - HALF_OPEN: Probing after recovery timeout — allow limited requests to test.

State transitions:
    CLOSED → OPEN: When consecutive failure count reaches ``failure_threshold``.
    OPEN → HALF_OPEN: Automatically after ``recovery_timeout_s`` elapses.
    HALF_OPEN → CLOSED: On a successful probe request.
    HALF_OPEN → OPEN: On a failed probe request.

Thread-safe via asyncio.Lock. No background timer — the ``state`` property
auto-checks the recovery timeout on each access.
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class CircuitState(StrEnum):
    """Circuit breaker states.

    Attributes:
        CLOSED: Normal — requests pass through to inference.
        OPEN: Tripped — fail fast without hitting the model.
        HALF_OPEN: Probing — allow limited requests to test recovery.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for model inference protection.

    Tracks consecutive inference failures and trips to OPEN state when
    the threshold is reached. After a recovery timeout, transitions to
    HALF_OPEN to allow a probe request. If the probe succeeds, resets
    to CLOSED. If it fails, returns to OPEN.

    Args:
        failure_threshold: Consecutive failures before tripping to OPEN.
        recovery_timeout_s: Seconds to wait in OPEN before probing (HALF_OPEN).
        half_open_max_calls: Max concurrent calls allowed in HALF_OPEN state.

    Example::

        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout_s=30.0)

        if not await breaker.allow_request():
            return 503  # Circuit is open, fail fast

        try:
            result = await inference()
            await breaker.record_success()
        except Exception:
            await breaker.record_failure()
            raise
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_s: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout_s = recovery_timeout_s
        self._half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, with auto-transition from OPEN → HALF_OPEN.

        When in OPEN state, checks if the recovery timeout has elapsed.
        If so, returns HALF_OPEN (without acquiring the lock — read-only
        check for the common fast path).

        Returns:
            The current CircuitState.
        """
        if (
            self._state == CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self._recovery_timeout_s
        ):
            return CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count.

        Returns:
            Number of consecutive failures since last success or reset.
        """
        return self._failure_count

    async def allow_request(self) -> bool:
        """Check if a request should be allowed through the circuit breaker.

        - CLOSED: Always allows.
        - OPEN: Always denies (fail fast).
        - HALF_OPEN: Allows up to ``half_open_max_calls`` concurrent probes.

        Returns:
            True if the request is allowed, False if it should be rejected.
        """
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True

        if current_state == CircuitState.OPEN:
            return False

        # HALF_OPEN: allow limited probe requests
        async with self._lock:
            if self._state == CircuitState.OPEN:
                # Transition to HALF_OPEN under lock
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                logger.info(
                    "Circuit breaker: OPEN → HALF_OPEN (recovery timeout elapsed)",
                    recovery_timeout_s=self._recovery_timeout_s,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self._half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False

        return False  # pragma: no cover

    async def record_success(self) -> None:
        """Record a successful inference call.

        Resets the failure count. If in HALF_OPEN state, transitions
        back to CLOSED (the probe succeeded — system has recovered).
        """
        async with self._lock:
            previous_state = self._state
            self._failure_count = 0
            self._half_open_calls = 0

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                logger.info(
                    "Circuit breaker: HALF_OPEN → CLOSED (probe succeeded)",
                )
            elif previous_state != CircuitState.CLOSED:
                self._state = CircuitState.CLOSED
                logger.info(
                    "Circuit breaker: %s → CLOSED (success recorded)",
                    previous_state.value,
                )

    async def record_failure(self) -> None:
        """Record a failed inference call.

        Increments the consecutive failure count. If the threshold is
        reached, transitions to OPEN. If already in HALF_OPEN (probe
        failed), transitions back to OPEN immediately.
        """
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — go back to OPEN
                self._state = CircuitState.OPEN
                self._half_open_calls = 0
                logger.warning(
                    "Circuit breaker: HALF_OPEN → OPEN (probe failed)",
                    failure_count=self._failure_count,
                )
            elif self._failure_count >= self._failure_threshold:
                # Threshold reached — trip the circuit
                previous_state = self._state
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit breaker: %s → OPEN (failure threshold reached)",
                    previous_state.value,
                    failure_count=self._failure_count,
                    failure_threshold=self._failure_threshold,
                )

    def reset(self) -> None:
        """Manually reset the circuit breaker to CLOSED state.

        Useful for admin/testing purposes. Resets all counters and state
        without requiring a lock (synchronous reset).
        """
        previous_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        logger.info(
            "Circuit breaker: %s → CLOSED (manual reset)",
            previous_state.value,
        )
