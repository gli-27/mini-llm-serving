"""Pre-compute KV cache and activation memory from model config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelMemoryConfig:
    """Model architecture parameters for memory estimation.

    Defaults are for TinyLlama-1.1B.

    Attributes:
        num_layers: Number of transformer layers.
        hidden_dim: Hidden dimension size.
        num_heads: Number of attention heads.
        dtype_bytes: Bytes per element (2=fp16, 4=fp32).
    """

    num_layers: int = 22
    hidden_dim: int = 2048
    num_heads: int = 32
    dtype_bytes: int = 2  # fp16


class MemoryEstimator:
    """Estimates memory requirements for KV cache and activations.

    Uses model architecture parameters to pre-compute memory needs
    before allocation, enabling admission control.

    Args:
        config: Model memory configuration.
    """

    def __init__(self, config: ModelMemoryConfig | None = None) -> None:
        self._config = config or ModelMemoryConfig()

    def estimate_kv_cache_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """Estimate KV cache memory for a given sequence length.

        Formula: 2 * num_layers * hidden_dim * seq_len * batch_size * dtype_bytes
        (factor 2 for key + value tensors)

        Args:
            seq_len: Sequence length (number of tokens cached).
            batch_size: Number of sequences in the batch.

        Returns:
            Estimated bytes required.
        """
        return (
            2
            * self._config.num_layers
            * self._config.hidden_dim
            * seq_len
            * batch_size
            * self._config.dtype_bytes
        )

    def estimate_activation_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """Estimate activation memory for a forward pass.

        Formula: 4 * batch_size * seq_len * hidden_dim * dtype_bytes
        (approximate — includes intermediate activations)

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.

        Returns:
            Estimated bytes required.
        """
        return 4 * batch_size * seq_len * self._config.hidden_dim * self._config.dtype_bytes

    def estimate_total_inference_bytes(self, seq_len: int, batch_size: int = 1) -> int:
        """Estimate total memory for inference (KV + activations).

        Args:
            seq_len: Sequence length.
            batch_size: Batch size.

        Returns:
            Total estimated bytes.
        """
        return self.estimate_kv_cache_bytes(seq_len, batch_size) + self.estimate_activation_bytes(
            seq_len, batch_size
        )

    def can_fit(self, seq_len: int, available_bytes: int, batch_size: int = 1) -> bool:
        """Check if a request can fit in available memory.

        Args:
            seq_len: Sequence length of the request.
            available_bytes: Available memory budget.
            batch_size: Batch size.

        Returns:
            True if the request fits in available memory.
        """
        needed = self.estimate_total_inference_bytes(seq_len, batch_size)
        return needed <= available_bytes
