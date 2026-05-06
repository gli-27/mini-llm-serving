"""Speculative decoding orchestrator.

Coordinates the draft model, target model verification, and rejection
sampling into a complete generation loop.
"""

from __future__ import annotations

import torch

from llm_serving.logging import get_logger
from llm_serving.speculative.config import SpeculativeConfig
from llm_serving.speculative.draft_runner import DraftModelRunner
from llm_serving.speculative.metrics import SpeculativeMetrics
from llm_serving.speculative.sampler import RejectionSampler
from llm_serving.speculative.verifier import TokenVerifier

logger = get_logger(__name__)


class SpeculativeOrchestrator:
    """Orchestrates the speculative decoding loop.

    Coordinates: draft → verify → sample → collect accepted tokens → repeat.

    Args:
        draft_runner: The draft model runner.
        verifier: The target model verifier.
        config: Speculative decoding configuration.
    """

    def __init__(
        self,
        draft_runner: DraftModelRunner,
        verifier: TokenVerifier,
        config: SpeculativeConfig,
    ) -> None:
        self._draft = draft_runner
        self._verifier = verifier
        self._config = config
        self._sampler = RejectionSampler()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> tuple[list[int], SpeculativeMetrics]:
        """Generate tokens using speculative decoding.

        Loop:
        1. Draft model generates K candidate tokens.
        2. Target model verifies all K in one forward pass.
        3. Rejection sampling accepts/rejects each token.
        4. Collect accepted + bonus/resampled token.
        5. Update input_ids and repeat.

        Args:
            input_ids: Starting token IDs, shape [1, seq_len].
            max_new_tokens: Maximum tokens to generate.
            eos_token_id: Stop generation on this token ID.

        Returns:
            Tuple of (generated_token_ids, metrics).
        """
        metrics = SpeculativeMetrics()
        generated: list[int] = []
        current_ids = input_ids
        k = self._config.num_draft_tokens
        temperature = self._config.temperature

        while len(generated) < max_new_tokens:
            metrics.total_steps += 1

            # Step 1: Draft model generates K candidates
            draft_tokens, draft_probs = self._draft.speculate(
                current_ids, k=k, temperature=temperature
            )
            metrics.total_drafted += k

            # Step 2: Target model verifies in single forward pass
            target_probs = self._verifier.verify(current_ids, draft_tokens, temperature=temperature)

            # Step 3: Rejection sampling
            result = self._sampler(draft_tokens, draft_probs, target_probs)
            metrics.total_accepted += result.num_accepted

            # Step 4: Collect tokens
            new_tokens: list[int] = []

            # Add accepted tokens
            for token_id in result.accepted_tokens.tolist():
                new_tokens.append(token_id)
                if eos_token_id is not None and token_id == eos_token_id:
                    generated.extend(new_tokens)
                    metrics.total_generated = len(generated)
                    return generated, metrics

            # Add bonus or resampled token
            if result.bonus_token is not None:
                new_tokens.append(result.bonus_token)
                if eos_token_id is not None and result.bonus_token == eos_token_id:
                    generated.extend(new_tokens)
                    metrics.total_generated = len(generated)
                    return generated, metrics
            elif result.resampled_token is not None:
                new_tokens.append(result.resampled_token)
                if eos_token_id is not None and result.resampled_token == eos_token_id:
                    generated.extend(new_tokens)
                    metrics.total_generated = len(generated)
                    return generated, metrics

            generated.extend(new_tokens)

            # Truncate if over max
            if len(generated) >= max_new_tokens:
                generated = generated[:max_new_tokens]
                break

            # Step 5: Update input_ids
            new_token_tensor = torch.tensor(
                [new_tokens], dtype=torch.long, device=current_ids.device
            )
            current_ids = torch.cat([current_ids, new_token_tensor], dim=1)

            # Check acceptance rate for fallback
            if (
                self._config.fallback_on_low_acceptance
                and metrics.total_steps >= 3
                and metrics.acceptance_rate < self._config.min_acceptance_rate
            ):
                logger.warning(
                    "Speculative decoding acceptance rate too low, "
                    "falling back to standard decoding",
                    acceptance_rate=metrics.acceptance_rate,
                )
                break

        metrics.total_generated = len(generated)
        return generated, metrics
