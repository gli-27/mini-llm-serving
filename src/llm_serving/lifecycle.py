"""Graceful shutdown lifecycle manager — drain in-flight requests before stopping.

Interview: "The key insight is layered graceful shutdown: ALB deregistration
handles routing, the lifecycle manager drains in-flight requests, uvicorn's
shutdown triggers lifespan cleanup, and SIGTERM timeout is the hard backstop."
"""

import asyncio
import signal

import structlog

logger = structlog.get_logger(__name__)


class LifecycleManager:
    """Manages graceful shutdown — drain in-flight requests before stopping.

    On SIGTERM/SIGINT:
    1. Sets is_shutting_down = True → /ready returns 503 → ALB stops routing
    2. Waits for in-flight requests to complete (up to drain_timeout)
    3. Returns control to lifespan for cleanup (stop workers, close Redis)
    """

    def __init__(self, drain_timeout: float = 30.0) -> None:
        self._shutting_down = False
        self._in_flight = 0
        self._drain_timeout = drain_timeout
        self._drained = asyncio.Event()
        self._drained.set()  # Starts drained (0 in-flight)

    @property
    def is_shutting_down(self) -> bool:
        """Whether a shutdown signal has been received."""
        return self._shutting_down

    @property
    def in_flight(self) -> int:
        """Number of currently in-flight requests."""
        return self._in_flight

    def track_request(self) -> None:
        """Increment in-flight counter when a request starts."""
        self._in_flight += 1
        self._drained.clear()

    def complete_request(self) -> None:
        """Decrement in-flight counter when a request completes."""
        self._in_flight = max(0, self._in_flight - 1)
        if self._in_flight == 0:
            self._drained.set()

    def register_signals(self) -> None:
        """Register SIGTERM/SIGINT handlers on the running event loop."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_signal, sig)
        logger.info("lifecycle.signals_registered")

    def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal — mark as shutting down."""
        self._shutting_down = True
        logger.info(
            "lifecycle.shutdown_signal",
            signal=sig.name,
            in_flight=self._in_flight,
        )

    async def wait_for_drain(self) -> None:
        """Wait for in-flight requests to complete, up to drain_timeout."""
        if self._in_flight == 0:
            return
        logger.info(
            "lifecycle.draining",
            in_flight=self._in_flight,
            timeout=self._drain_timeout,
        )
        try:
            await asyncio.wait_for(
                self._drained.wait(), timeout=self._drain_timeout
            )
            logger.info("lifecycle.drained")
        except TimeoutError:
            logger.warning(
                "lifecycle.drain_timeout",
                remaining=self._in_flight,
            )
