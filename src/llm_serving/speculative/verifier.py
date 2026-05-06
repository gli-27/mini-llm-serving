"""Token verifier using the target model for speculative decoding.

Performs a single forward pass with the target model on the full
sequence (input + draft tokens) to get probability distributions
for verification.
"""

from __future__ import annotations

import torch

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class TokenVerifier:
    """Verifies draft tokens using the target model.

    Takes the existing target model (from ModelManager) and runs a single
    forward pass to get probabilities for all draft token positions + one
    bonus position.

    Args:
        target_model: The loaded target model (from ModelManager).
    """

    def __init__(self, target_model: torch.nn.Module) -> None:
        self._model = target_model

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Verify draft tokens with a single target model forward pass.

        Concatenates input_ids with draft_tokens, runs a single forward
        pass, and extracts logits for the K draft positions + 1 bonus.

        Args:
            input_ids: Original input token IDs, shape [1, seq_len].
            draft_tokens: Draft token IDs to verify, shape [K].
            temperature: Sampling temperature for softmax.

        Returns:
            Target probabilities, shape [K+1, vocab_size].
            Position i has probs for verifying draft token i.
            Position K has probs for the bonus token.
        """
        k = draft_tokens.shape[0]
        prefix_len = input_ids.shape[1]

        # Concatenate input + draft tokens for single forward pass
        full_ids = torch.cat([input_ids, draft_tokens.unsqueeze(0)], dim=1)  # [1, seq_len + K]

        # Single forward pass through target model
        outputs = self._model(full_ids, use_cache=False)
        logits = outputs.logits[0]  # [seq_len + K, vocab_size]

        # Extract logits for positions [prefix_len-1 : prefix_len-1+K+1]
        # Position prefix_len-1 gives probs for what should be at position prefix_len
        # (i.e., verifying draft_tokens[0])
        start = prefix_len - 1
        end = start + k + 1
        relevant_logits = logits[start:end]  # [K+1, vocab_size]

        # Apply temperature and softmax
        if temperature != 1.0:
            relevant_logits = relevant_logits / temperature

        target_probs = torch.softmax(relevant_logits, dim=-1)  # [K+1, vocab_size]

        return target_probs
