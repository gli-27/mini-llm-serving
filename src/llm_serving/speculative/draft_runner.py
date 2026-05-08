"""Draft model runner for speculative decoding.

Loads a small draft model and performs K autoregressive forward passes
to generate candidate tokens with their probability distributions.
Uses KV cache for efficient sequential generation (only passes the
last token after the first step).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_serving.logging import get_logger

logger = get_logger(__name__)


class DraftModelRunner:
    """Runs the draft model to generate candidate tokens.

    Performs K sequential forward passes with KV cache to produce draft
    tokens and their full vocabulary probability distributions. After the
    first step, only the last token is passed (KV cache stores the rest).

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
        """Generate K draft tokens autoregressively with KV cache.

        First step: forward full input_ids, cache KV states.
        Steps 2..K: forward only the last token, reuse cached KV.

        This is 2-3x faster than recomputing the full sequence each step.

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

        draft_tokens: list[int] = []
        draft_probs_list: list[torch.Tensor] = []
        past_kv = None
        next_input = input_ids  # First step: full input

        for _step in range(k):
            outputs = self._model(next_input, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values

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

            # Next step: only pass the new token (KV cache has the rest)
            next_input = token.unsqueeze(0)  # [1, 1]

        return (
            torch.tensor(draft_tokens, dtype=torch.long, device=input_ids.device),
            torch.stack(draft_probs_list),  # [K, vocab_size]
        )
