"""Rejection sampler for speculative decoding.

Implements the token-level acceptance/rejection criterion from
"Fast Inference from Transformers via Speculative Decoding" (Leviathan et al. 2023).

Accept criterion: random() < min(1, p_target(x) / p_draft(x))
On rejection: resample from normalize(max(0, p_target - p_draft))
All K accepted: bonus token sampled from target_probs[K]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AcceptResult:
    """Result of the rejection sampling step.

    Attributes:
        num_accepted: Number of draft tokens accepted (0 to K).
        accepted_tokens: Tensor of accepted token IDs.
        bonus_token: If all K accepted, bonus token from target. Else None.
        resampled_token: If rejected, token resampled from residual. Else None.
    """

    num_accepted: int
    accepted_tokens: torch.Tensor
    bonus_token: int | None
    resampled_token: int | None


class RejectionSampler:
    """Rejection sampler for speculative decoding verification.

    Given K draft tokens with their probabilities and the target model's
    probabilities, determines which tokens to accept using the standard
    speculative decoding acceptance criterion.

    The math guarantees that the output token distribution is identical
    to sampling directly from the target model.
    """

    def __call__(
        self,
        draft_tokens: torch.Tensor,
        draft_probs: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> AcceptResult:
        """Run rejection sampling on draft tokens.

        Args:
            draft_tokens: Token IDs from draft model, shape [K].
            draft_probs: Draft probabilities, shape [K, vocab_size].
            target_probs: Target probabilities, shape [K+1, vocab_size].
                Position i has target probs for the i-th draft token.
                Position K has target probs for the bonus token (if all accept).

        Returns:
            AcceptResult with acceptance details.
        """
        k = draft_tokens.shape[0]
        device = draft_tokens.device

        accepted_tokens_list: list[int] = []

        for i in range(k):
            token_id = draft_tokens[i].item()

            # Get p_target and p_draft for this specific token
            p_target = target_probs[i, token_id].item()
            p_draft = draft_probs[i, token_id].item()

            # Avoid division by zero: if draft gives 0 prob but target > 0,
            # acceptance ratio is infinity → always accept
            if p_draft == 0:
                if p_target > 0:
                    # Draft assigned 0 prob but target wants it → accept
                    accepted_tokens_list.append(token_id)
                    continue
                else:
                    # Both 0 → accept (token is impossible for both)
                    accepted_tokens_list.append(token_id)
                    continue

            # Accept criterion: random() < min(1, p_target / p_draft)
            acceptance_ratio = min(1.0, p_target / p_draft)
            r = torch.rand(1, device=device).item()

            if r < acceptance_ratio:
                # ACCEPT this token
                accepted_tokens_list.append(token_id)
            else:
                # REJECT — resample from residual distribution
                # residual = normalize(max(0, p_target - p_draft))
                residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
                residual_sum = residual.sum()

                if residual_sum > 0:
                    residual = residual / residual_sum
                    resampled = torch.multinomial(residual, num_samples=1)
                    resampled_token = resampled.item()
                else:
                    # Fallback: sample from target distribution directly
                    resampled = torch.multinomial(target_probs[i], num_samples=1)
                    resampled_token = resampled.item()

                return AcceptResult(
                    num_accepted=i,
                    accepted_tokens=torch.tensor(
                        accepted_tokens_list, dtype=torch.long, device=device
                    ),
                    bonus_token=None,
                    resampled_token=resampled_token,
                )

        # ALL K tokens accepted — sample bonus token from target_probs[K]
        bonus = torch.multinomial(target_probs[k], num_samples=1)

        return AcceptResult(
            num_accepted=k,
            accepted_tokens=torch.tensor(accepted_tokens_list, dtype=torch.long, device=device),
            bonus_token=bonus.item(),
            resampled_token=None,
        )
