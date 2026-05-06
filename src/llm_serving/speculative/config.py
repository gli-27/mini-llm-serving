"""Configuration for speculative decoding."""

from pydantic import BaseModel


class SpeculativeConfig(BaseModel):
    """Configuration for the speculative decoding pipeline.

    Attributes:
        enabled: Whether speculative decoding is active.
        draft_model_name: HuggingFace model ID for the draft model.
        num_draft_tokens: Number of tokens the draft model generates per step.
        temperature: Sampling temperature for both draft and target.
        min_acceptance_rate: Below this rate, fallback to standard decoding.
        fallback_on_low_acceptance: If True, disable speculation when rate is low.
    """

    enabled: bool = False
    draft_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    num_draft_tokens: int = 5
    temperature: float = 1.0
    min_acceptance_rate: float = 0.3
    fallback_on_low_acceptance: bool = True
