"""Metrics tracking for speculative decoding."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpeculativeMetrics:
    """Tracks speculative decoding performance metrics.

    Attributes:
        total_accepted: Total tokens accepted from draft.
        total_drafted: Total tokens drafted (K per step).
        total_steps: Number of speculative decoding steps.
        total_generated: Total tokens in final output.
    """

    total_accepted: int = 0
    total_drafted: int = 0
    total_steps: int = 0
    total_generated: int = 0

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens accepted by target model."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def avg_tokens_per_step(self) -> float:
        """Average tokens produced per speculative step."""
        if self.total_steps == 0:
            return 0.0
        return self.total_generated / self.total_steps

    @property
    def speedup_estimate(self) -> float:
        """Estimated speedup vs. standard autoregressive decoding.

        Approximation: tokens_per_step / 1 (standard = 1 token per step).
        Real speedup depends on draft/target model size ratio.
        """
        return self.avg_tokens_per_step

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to a serializable dict."""
        return {
            "acceptance_rate": round(self.acceptance_rate, 4),
            "avg_tokens_per_step": round(self.avg_tokens_per_step, 2),
            "total_steps": self.total_steps,
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "total_drafted": self.total_drafted,
            "speedup_estimate": round(self.speedup_estimate, 2),
        }
