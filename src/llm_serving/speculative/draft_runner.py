"""Draft model runner for speculative decoding.

Loads a small draft model and performs K autoregressive forward passes
to generate candidate tokens with their probability distributions.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class DraftModelRunner:
    """Runs the draft model to generate candidate tokens.

    Performs K sequential forward passes to produce draft tokens
    and their full vocabulary probability distributions.

    Args:
        model_name: HuggingFace model identifier for the draft model.
        device: Device to load the model on (cpu/cuda).
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    def load(self) -> None:
        """Load the draft model and tokenizer."""
        logger.info("Loading draft model: %s", self._model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
        self._model.to(self._device)
        self._model.eval()

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info("Draft model loaded on %s", self._device)

    @property
    def is_loaded(self) -> bool:
        """Check if draft model is loaded."""
        return self._model is not None

    @torch.no_grad()
    def speculate(
        self,
        input_ids: torch.Tensor,
        k: int,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate K draft tokens autoregressively.

        Performs K forward passes, collecting the sampled token and the
        full vocabulary probability distribution at each step.

        Args:
            input_ids: Input token IDs, shape [1, seq_len].
            k: Number of draft tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Tuple of:
                - draft_tokens: shape [K] — sampled token IDs
                - draft_probs: shape [K, vocab_size] — probability distributions
        """
        assert self._model is not None

        current_ids = input_ids
        draft_tokens: list[int] = []
        draft_probs_list: list[torch.Tensor] = []

        for _ in range(k):
            outputs = self._model(current_ids, use_cache=False)
            # Get logits for the last position
            logits = outputs.logits[0, -1, :]  # [vocab_size]

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)  # [vocab_size]
            draft_probs_list.append(probs)

            # Sample a token
            token = torch.multinomial(probs, num_samples=1)  # [1]
            draft_tokens.append(token.item())

            # Append to input for next step
            current_ids = torch.cat([current_ids, token.unsqueeze(0)], dim=1)

        return (
            torch.tensor(draft_tokens, dtype=torch.long, device=input_ids.device),
            torch.stack(draft_probs_list),  # [K, vocab_size]
        )
