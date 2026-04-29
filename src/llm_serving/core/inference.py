"""Inference logic for text generation.

Handles tokenization, model forward pass, and decoding — separated from
model lifecycle management in ``models/loader.py``.

Note: This module contains synchronous (blocking) functions. They are
designed to be called via ``run_in_executor()`` from the async router
so they don't block the event loop.
"""

import torch

from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager

logger = get_logger(__name__)


def generate(
    model_manager: ModelManager,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    seed: int | None = None,
) -> tuple[str, int, int]:
    """Generate text from a prompt using the loaded model.

    This is a synchronous blocking function. The router calls it via
    ``loop.run_in_executor()`` to avoid blocking the async event loop,
    wrapped with ``asyncio.wait_for()`` for timeout control.

    Args:
        model_manager: The ModelManager holding the loaded model and tokenizer.
        prompt: The input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. Higher = more random, lower = more deterministic.
        seed: Optional random seed for reproducible generation.

    Returns:
        A tuple of (generated_text, prompt_token_count, completion_token_count).

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
    """
    if not model_manager.is_loaded:
        raise ModelNotLoadedError(
            "Model is not loaded. Call load_model() before generate()."
        )

    # Safe to assert after is_loaded check — satisfies the type checker
    assert model_manager.tokenizer is not None
    assert model_manager.model is not None

    tokenizer = model_manager.tokenizer
    model = model_manager.model
    device = model_manager.settings.device

    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    generated_ids = output_ids[0, prompt_tokens:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    logger.info(
        "Generated %d tokens from %d prompt tokens",
        completion_tokens,
        prompt_tokens,
    )

    return generated_text, prompt_tokens, completion_tokens
