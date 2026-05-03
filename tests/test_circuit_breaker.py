"""Tests for the circuit breaker."""

import asyncio

from llm_serving.core.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    def test_starts_closed(self) -> None:
        """Circuit breaker should start in CLOSED state."""
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED

    def test_starts_with_zero_failures(self) -> None:
        """Failure count should start at 0."""
        breaker = CircuitBreaker()
        assert breaker.failure_count == 0

    def test_custom_parameters(self) -> None:
        """Custom parameters should be stored correctly."""
        breaker = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout_s=60.0,
            half_open_max_calls=3,
        )
        assert breaker._failure_threshold == 10
        assert breaker._recovery_timeout_s == 60.0
        assert breaker._half_open_max_calls == 3


class TestClosedState:
    """Tests for CLOSED state behavior."""

    async def test_allows_requests_when_closed(self) -> None:
        """CLOSED state should always allow requests."""
        breaker = CircuitBreaker(failure_threshold=5)
        assert await breaker.allow_request() is True

    async def test_stays_closed_under_threshold(self) -> None:
        """Should stay CLOSED if failures are below threshold."""
        breaker = CircuitBreaker(failure_threshold=5)

        for _ in range(4):
            await breaker.record_failure()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 4
        assert await breaker.allow_request() is True

    async def test_success_resets_failure_count(self) -> None:
        """A success should reset the consecutive failure count."""
        breaker = CircuitBreaker(failure_threshold=5)

        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.failure_count == 3

        await breaker.record_success()
        assert breaker.failure_count == 0


class TestOpenState:
    """Tests for OPEN state behavior."""

    async def test_trips_at_threshold(self) -> None:
        """Should trip to OPEN when failure count reaches threshold."""
        breaker = CircuitBreaker(failure_threshold=3)

        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3

    async def test_denies_requests_when_open(self) -> None:
        """OPEN state should deny all requests (fail fast)."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=60.0)

        await breaker.record_failure()
        await breaker.record_failure()

        assert breaker.state == CircuitState.OPEN
        assert await breaker.allow_request() is False

    async def test_extra_failures_stay_open(self) -> None:
        """Additional failures while OPEN should keep it OPEN."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=60.0)

        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_failure()  # Extra failure while open

        assert breaker.state == CircuitState.OPEN
        assert breaker.failure_count == 3


class TestHalfOpenState:
    """Tests for HALF_OPEN state behavior."""

    async def test_transitions_to_half_open_after_timeout(self) -> None:
        """Should transition OPEN → HALF_OPEN after recovery timeout."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)

        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN

    async def test_allows_one_probe_in_half_open(self) -> None:
        """HALF_OPEN should allow exactly half_open_max_calls probes."""
        breaker = CircuitBreaker(
            failure_threshold=2, recovery_timeout_s=0.01, half_open_max_calls=1
        )

        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.02)

        # First probe should be allowed
        assert await breaker.allow_request() is True
        # Second probe should be denied
        assert await breaker.allow_request() is False

    async def test_success_in_half_open_closes_circuit(self) -> None:
        """A successful probe in HALF_OPEN should transition to CLOSED."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)

        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.02)

        # Allow probe
        assert await breaker.allow_request() is True

        # Record success → should close
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    async def test_failure_in_half_open_reopens_circuit(self) -> None:
        """A failed probe in HALF_OPEN should transition back to OPEN."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)

        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.02)

        # Allow probe
        assert await breaker.allow_request() is True

        # Probe fails → back to OPEN
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

    async def test_multiple_probes_allowed(self) -> None:
        """half_open_max_calls > 1 should allow multiple probes."""
        breaker = CircuitBreaker(
            failure_threshold=2, recovery_timeout_s=0.01, half_open_max_calls=3
        )

        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.02)

        # Should allow 3 probes
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True
        assert await breaker.allow_request() is True
        # 4th should be denied
        assert await breaker.allow_request() is False


class TestReset:
    """Tests for manual reset."""

    async def test_reset_from_open(self) -> None:
        """reset() should return to CLOSED from any state."""
        breaker = CircuitBreaker(failure_threshold=2)

        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert await breaker.allow_request() is True

    async def test_reset_clears_half_open_calls(self) -> None:
        """reset() should clear the half-open call counter."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)

        await breaker.record_failure()
        await breaker.record_failure()
        await asyncio.sleep(0.02)
        await breaker.allow_request()  # Use the one probe slot

        breaker.reset()
        assert breaker._half_open_calls == 0


class TestFullCycle:
    """Integration tests for the full circuit breaker lifecycle."""

    async def test_full_lifecycle(self) -> None:
        """Test the complete CLOSED → OPEN → HALF_OPEN → CLOSED cycle."""
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout_s=0.01)

        # 1. Start CLOSED — requests pass through
        assert breaker.state == CircuitState.CLOSED
        assert await breaker.allow_request() is True

        # 2. Accumulate failures → trip to OPEN
        await breaker.record_failure()
        await breaker.record_failure()
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN
        assert await breaker.allow_request() is False

        # 3. Wait for recovery → HALF_OPEN
        await asyncio.sleep(0.02)
        assert breaker.state == CircuitState.HALF_OPEN
        assert await breaker.allow_request() is True  # Probe allowed

        # 4. Probe succeeds → back to CLOSED
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0
        assert await breaker.allow_request() is True

    async def test_flapping_protection(self) -> None:
        """If probes keep failing, circuit stays OPEN (no rapid flapping)."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout_s=0.01)

        # Trip to OPEN
        await breaker.record_failure()
        await breaker.record_failure()

        # First recovery attempt — probe fails
        await asyncio.sleep(0.02)
        assert await breaker.allow_request() is True
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Second recovery attempt — probe fails again
        await asyncio.sleep(0.02)
        assert await breaker.allow_request() is True
        await breaker.record_failure()
        assert breaker.state == CircuitState.OPEN

        # Third recovery attempt — probe succeeds!
        await asyncio.sleep(0.02)
        assert await breaker.allow_request() is True
        await breaker.record_success()
        assert breaker.state == CircuitState.CLOSED
