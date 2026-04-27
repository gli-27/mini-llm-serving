"""Model loading and lifecycle management.

Provides a singleton ModelManager that handles loading models from HuggingFace,
managing their state, and tracking readiness.
"""

import logging

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from llm_serving.config import Settings
from llm_serving.exceptions import ModelNotLoadedError  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages the lifecycle of an LLM: loading, state tracking, and access.

    Responsible only for model/tokenizer loading and readiness — inference
    logic lives in ``core/inference.py``.

    Args:
        settings: Application settings containing model name, device, etc.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings: Settings = settings
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None

    @property
    def is_loaded(self) -> bool:
        """Check whether the model and tokenizer are loaded.

        Returns:
            True if both model and tokenizer are loaded and ready.
        """
        return self.model is not None and self.tokenizer is not None

    def load_model(self) -> None:
        """Load the model and tokenizer from HuggingFace.

        Loads the tokenizer and model specified in settings, moves the model
        to the configured device, and sets it to eval mode. Handles TinyLlama's
        missing pad_token by falling back to eos_token.

        Raises:
            Exception: If model loading fails for any reason.
        """
        model_name = self.settings.model_name
        device = self.settings.device

        logger.info("Loading tokenizer for %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # TinyLlama doesn't have a pad_token — fall back to eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token to eos_token: %s", self.tokenizer.eos_token)

        logger.info("Loading model %s", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(
            "Model loaded: name=%s, device=%s, parameters=%s",
            model_name,
            device,
            f"{param_count:,}",
        )
