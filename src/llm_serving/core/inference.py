"""Inference logic for text generation.

Handles tokenization, model forward pass, and decoding — separated from
model lifecycle management in ``models/loader.py``.

Note: This module contains synchronous (blocking) functions. They are
designed to be called via ``run_in_executor()`` from the async router
so they don't block the event loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager

if TYPE_CHECKING:
    from llm_serving.core.kv_cache import KVCacheManager

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
        raise ModelNotLoadedError("Model is not loaded. Call load_model() before generate().")

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


def generate_batch(
    model_manager: ModelManager,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_lengths: list[int],
    max_new_tokens: int,
    temperature: float = 0.7,
    per_request_max_tokens: list[int] | None = None,
) -> list[tuple[str, int, int]]:
    """Generate text for a batch of padded inputs.

    This is the batched counterpart to :func:`generate`. It runs a single
    ``model.generate()`` call for multiple requests simultaneously,
    amortizing the fixed overhead of the forward pass.

    The caller (BatchScheduler) is responsible for tokenization and
    left-padding. This function handles only the model forward pass
    and output splitting/decoding.

    Args:
        model_manager: The ModelManager holding the loaded model and tokenizer.
        input_ids: Left-padded input tensor of shape [batch_size, max_seq_len].
        attention_mask: Mask tensor (1=real, 0=padding) same shape as input_ids.
        prompt_lengths: Original (unpadded) token count for each item.
        max_new_tokens: Maximum new tokens to generate (max across batch).
        temperature: Sampling temperature (max across batch).
        per_request_max_tokens: Per-request max_tokens for truncation. If None,
            all items use max_new_tokens.

    Returns:
        List of (generated_text, prompt_tokens, completion_tokens) tuples,
        one per batch item.

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
    """
    if not model_manager.is_loaded:
        raise ModelNotLoadedError("Model is not loaded. Call load_model() before generate_batch().")

    assert model_manager.tokenizer is not None
    assert model_manager.model is not None

    tokenizer = model_manager.tokenizer
    model = model_manager.model
    pad_token_id = tokenizer.pad_token_id or 0
    max_prompt_len = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=pad_token_id,
        )

    # Split outputs and decode each individually
    results: list[tuple[str, int, int]] = []
    for i, prompt_len in enumerate(prompt_lengths):
        # Generated tokens start after the full padded prompt
        generated_start = max_prompt_len
        generated_ids = output_ids[i, generated_start:]

        # Truncate to per-request max_tokens if specified
        if per_request_max_tokens and len(generated_ids) > per_request_max_tokens[i]:
            generated_ids = generated_ids[: per_request_max_tokens[i]]

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion_tokens = len(generated_ids)

        results.append((generated_text, prompt_len, completion_tokens))

    logger.info(
        "Batch generated: %d items, %d max new tokens",
        len(results),
        max_new_tokens,
    )

    return results


def generate_with_cache(
    model_manager: ModelManager,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    seed: int | None = None,
    kv_cache_manager: KVCacheManager | None = None,
    cache_prefix_tokens: int | None = None,
) -> tuple[str, int, int]:
    """Generate text with optional KV cache reuse for prefix.

    Flow:
    1. Tokenize full prompt.
    2. If ``cache_prefix_tokens`` is set, split into prefix + suffix.
    3. Cache hit → use past_key_values, only forward suffix tokens.
    4. Cache miss → forward full prompt with ``use_cache=True``, store prefix KV.
    5. Generate remaining tokens from the combined state.

    Falls back to standard ``generate()`` if cache is disabled or
    ``cache_prefix_tokens`` is not specified.

    Args:
        model_manager: The ModelManager holding the loaded model and tokenizer.
        prompt: The input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        seed: Optional random seed for reproducible generation.
        kv_cache_manager: Optional KVCacheManager for prefix caching.
        cache_prefix_tokens: Number of tokens to treat as cacheable prefix.
            If None, caching is skipped (full generate).

    Returns:
        A tuple of (generated_text, prompt_token_count, completion_token_count).

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
    """
    # Fall back to standard generate if no cache or no prefix specified
    if kv_cache_manager is None or cache_prefix_tokens is None:
        return generate(
            model_manager=model_manager,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )

    if not model_manager.is_loaded:
        raise ModelNotLoadedError(
            "Model is not loaded. Call load_model() before generate_with_cache()."
        )

    assert model_manager.tokenizer is not None
    assert model_manager.model is not None

    tokenizer = model_manager.tokenizer
    model = model_manager.model
    device = model_manager.settings.device

    # Tokenize full prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    prompt_tokens = input_ids.shape[1]

    if seed is not None:
        torch.manual_seed(seed)

    # Split into prefix and suffix
    prefix_len = min(cache_prefix_tokens, prompt_tokens)
    prefix_ids = input_ids[0, :prefix_len].tolist()

    # Try cache lookup
    cached_entry = kv_cache_manager.get(prefix_ids)

    with torch.no_grad():
        if cached_entry is not None:
            # Cache HIT: forward only the suffix tokens with cached KV
            suffix_ids = input_ids[:, prefix_len:]
            logger.info(
                "KV cache hit: forwarding %d suffix tokens (skipped %d prefix)",
                suffix_ids.shape[1],
                prefix_len,
            )

            output_ids = model.generate(
                suffix_ids,
                past_key_values=cached_entry.past_key_values,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

            # Output includes suffix + generated (not prefix)
            generated_ids = output_ids[0, suffix_ids.shape[1] :]
        else:
            # Cache MISS: compute prefix KV, cache it, then generate
            # using the cached KV + suffix (avoids redundant recomputation)
            logger.info(
                "KV cache miss: computing prefix KV for %d tokens",
                prefix_len,
            )

            # Step 1: Forward prefix to get its KV state
            prefix_input = input_ids[:, :prefix_len]
            prefix_output = model(prefix_input, use_cache=True)
            prefix_kv = prefix_output.past_key_values

            # Step 2: Cache the prefix KV for future reuse
            kv_cache_manager.put(prefix_ids, prefix_kv)

            # Step 3: Generate using cached prefix KV + suffix tokens
            # This avoids recomputing the prefix — we already have its KV
            suffix_ids = input_ids[:, prefix_len:]
            output_ids = model.generate(
                suffix_ids,
                past_key_values=prefix_kv,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )

            # Output includes suffix + generated (not prefix)
            generated_ids = output_ids[0, suffix_ids.shape[1] :]

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    completion_tokens = len(generated_ids)

    logger.info(
        "Generated %d tokens from %d prompt tokens (cache=%s)",
        completion_tokens,
        prompt_tokens,
        "hit" if cached_entry else "miss",
    )

    return generated_text, prompt_tokens, completion_tokens
