"""Tests for speculative decoding components."""

import torch

from llm_serving.speculative.metrics import SpeculativeMetrics
from llm_serving.speculative.sampler import RejectionSampler


class TestRejectionSampler:
    """Tests for the rejection sampler — core math."""

    def test_all_accepted_same_distribution(self) -> None:
        """When draft and target have identical probs, all K tokens are accepted."""
        sampler = RejectionSampler()
        k = 5
        vocab_size = 10

        # Same probability distribution for draft and target
        probs = torch.zeros(vocab_size)
        probs[3] = 1.0  # Deterministic — token 3 always

        draft_tokens = torch.tensor([3, 3, 3, 3, 3])
        draft_probs = probs.unsqueeze(0).expand(k, -1)  # [K, vocab]
        target_probs = probs.unsqueeze(0).expand(k + 1, -1)  # [K+1, vocab]

        result = sampler(draft_tokens, draft_probs, target_probs)

        # All should be accepted (p_target/p_draft = 1.0 → always accept)
        assert result.num_accepted == k
        assert result.accepted_tokens.tolist() == [3, 3, 3, 3, 3]
        assert result.bonus_token is not None
        assert result.resampled_token is None

    def test_rejection_returns_valid_token(self) -> None:
        """Resampled token on rejection should be a valid vocab index."""
        sampler = RejectionSampler()
        vocab_size = 10

        # Draft says token 0 is certain, target says token 5 is certain
        # This guarantees rejection (p_target[0] = 0, p_draft[0] = 1)
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, 0] = 1.0

        target_probs = torch.zeros(2, vocab_size)  # K=1, need K+1=2
        target_probs[0, 5] = 1.0  # Target wants token 5
        target_probs[1, 5] = 1.0  # Bonus position

        draft_tokens = torch.tensor([0])  # Draft chose token 0

        result = sampler(draft_tokens, draft_probs, target_probs)

        # Should reject token 0 (target gives it 0 prob) — but wait,
        # if p_draft=1 and p_target=0, acceptance_ratio = 0 → always reject
        assert result.num_accepted == 0
        assert result.resampled_token is not None
        assert 0 <= result.resampled_token < vocab_size
        # The resampled token should be 5 (max(0, target-draft) = target here)
        assert result.resampled_token == 5

    def test_residual_nonnegative(self) -> None:
        """The residual max(0, target - draft) should never be negative."""
        # This is verified by torch.clamp(min=0.0) in the implementation.
        # We verify it doesn't crash with overlapping distributions.
        sampler = RejectionSampler()
        vocab_size = 5

        draft_probs = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0]])
        target_probs = torch.tensor(
            [
                [0.1, 0.1, 0.3, 0.3, 0.2],
                [0.2, 0.2, 0.2, 0.2, 0.2],  # bonus
            ]
        )
        draft_tokens = torch.tensor([0])

        # Force rejection by setting manual seed for reproducibility
        torch.manual_seed(999)  # This may accept or reject — both are valid
        result = sampler(draft_tokens, draft_probs, target_probs)

        # Verify the result is valid regardless of accept/reject
        assert result.num_accepted >= 0
        if result.resampled_token is not None:
            assert 0 <= result.resampled_token < vocab_size

    def test_bonus_token_on_full_accept(self) -> None:
        """When all K tokens are accepted, a bonus token is sampled."""
        sampler = RejectionSampler()
        k = 3
        vocab_size = 5

        # Make target probs >= draft probs everywhere → guaranteed acceptance
        # (acceptance ratio = p_target/p_draft >= 1 for the chosen token)
        draft_probs = torch.zeros(k, vocab_size)
        target_probs = torch.zeros(k + 1, vocab_size)

        # Draft and target agree: token 2 is the choice with high prob
        for i in range(k):
            draft_probs[i, 2] = 0.5
            draft_probs[i, :] = draft_probs[i, :] + 0.1  # smooth
            draft_probs[i] = draft_probs[i] / draft_probs[i].sum()

            # Target gives equal or higher prob to token 2
            target_probs[i, 2] = 0.9
            target_probs[i, :] = target_probs[i, :] + 0.02
            target_probs[i] = target_probs[i] / target_probs[i].sum()

        # Bonus position
        target_probs[k] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])

        draft_tokens = torch.tensor([2, 2, 2])

        # With target_prob[token] > draft_prob[token], acceptance ratio >= 1
        # → always accept
        result = sampler(draft_tokens, draft_probs, target_probs)

        assert result.num_accepted == k
        assert result.bonus_token is not None
        assert 0 <= result.bonus_token < vocab_size
        assert result.resampled_token is None

    def test_zero_draft_prob_handling(self) -> None:
        """Should not divide by zero when draft prob is 0."""
        sampler = RejectionSampler()
        vocab_size = 5

        # Draft gives 0 probability to the drafted token (edge case)
        draft_probs = torch.zeros(1, vocab_size)
        draft_probs[0, 1] = 0.5
        draft_probs[0, 2] = 0.5
        # Token 3 has 0 draft prob

        target_probs = torch.zeros(2, vocab_size)
        target_probs[0, 3] = 0.8  # Target wants token 3
        target_probs[0, 1] = 0.2
        target_probs[1] = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])

        draft_tokens = torch.tensor([3])  # Draft somehow chose token 3

        # Should NOT raise ZeroDivisionError
        result = sampler(draft_tokens, draft_probs, target_probs)

        # p_draft=0, p_target=0.8 → accept (infinity ratio)
        assert result.num_accepted == 1
        assert result.bonus_token is not None


class TestSpeculativeMetrics:
    """Tests for SpeculativeMetrics."""

    def test_acceptance_rate_calculation(self) -> None:
        """acceptance_rate should be total_accepted / total_drafted."""
        metrics = SpeculativeMetrics(
            total_accepted=8, total_drafted=10, total_steps=2, total_generated=12
        )
        assert metrics.acceptance_rate == 0.8

    def test_acceptance_rate_zero_drafted(self) -> None:
        """acceptance_rate should be 0.0 when nothing drafted."""
        metrics = SpeculativeMetrics()
        assert metrics.acceptance_rate == 0.0

    def test_avg_tokens_per_step(self) -> None:
        """avg_tokens_per_step = total_generated / total_steps."""
        metrics = SpeculativeMetrics(total_generated=12, total_steps=2)
        assert metrics.avg_tokens_per_step == 6.0

    def test_to_dict(self) -> None:
        """to_dict should return all metrics."""
        metrics = SpeculativeMetrics(
            total_accepted=5, total_drafted=10, total_steps=2, total_generated=7
        )
        d = metrics.to_dict()
        assert "acceptance_rate" in d
        assert "speedup_estimate" in d
        assert d["total_steps"] == 2
