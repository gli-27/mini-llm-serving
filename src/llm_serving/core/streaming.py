"""SSE streaming inference using TextIteratorStreamer.

Runs model.generate() in a background thread and yields tokens
one-by-one via a TextIteratorStreamer, enabling Server-Sent Events
streaming to the client.
"""

import threading
from collections.abc import Generator

import torch
from transformers import TextIteratorStreamer

from llm_serving.exceptions import ModelNotLoadedError
from llm_serving.logging import get_logger
from llm_serving.models.loader import ModelManager

logger = get_logger(__name__)


def generate_stream(
    model_manager: ModelManager,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.7,
    seed: int | None = None,
) -> Generator[str, None, None]:
    """Generate text token-by-token as a streaming generator.

    Uses HuggingFace's TextIteratorStreamer to yield tokens as they are
    generated. The model.generate() call runs in a background thread so
    the main thread can yield tokens to the SSE response as they arrive.

    If the background thread raises an exception (OOM, CUDA error, etc.),
    it is re-raised in the main thread after the streamer drains.

    Args:
        model_manager: The ModelManager holding the loaded model and tokenizer.
        prompt: The input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature. Higher = more random, lower = more deterministic.
        seed: Optional random seed for reproducible generation.

    Yields:
        Individual token strings as they are generated.

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
        RuntimeError: If the background generation thread fails.
    """
    if not model_manager.is_loaded:
        raise ModelNotLoadedError(
            "Model is not loaded. Call load_model() before generate_stream()."
        )

    assert model_manager.tokenizer is not None
    assert model_manager.model is not None

    tokenizer = model_manager.tokenizer
    model = model_manager.model
    device = model_manager.settings.device

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    # Create a streamer that yields tokens one-by-one.
    # timeout ensures the main thread doesn't hang forever if the
    # background thread dies — it will raise queue.Empty after timeout.
    timeout_s = model_manager.settings.generation_timeout_s
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=timeout_s,
    )

    # Generation kwargs for the background thread
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    # Mutable container to capture exceptions from the background thread.
    # A list works because threads share the same process memory — the
    # background thread appends to it, the main thread reads from it.
    errors: list[BaseException] = []

    # Run model.generate() in a background thread — it blocks until done,
    # but the streamer yields tokens to our main thread as they're produced.
    # NOTE: The seed is passed to _generate_in_thread and set INSIDE the
    # background thread because PyTorch's CPU RNG state is thread-local.
    # Setting torch.manual_seed() in the main thread would NOT affect the
    # RNG used by model.generate() in the background thread.
    thread = threading.Thread(
        target=_generate_in_thread,
        args=(model, generation_kwargs, errors, seed),
        daemon=True,
    )
    thread.start()

    # Yield tokens as they arrive from the streamer
    token_count = 0
    try:
        for token_text in streamer:
            if token_text:
                token_count += 1
                yield token_text
    except Exception:
        # TextIteratorStreamer raises queue.Empty on timeout —
        # check if the background thread died
        if errors:
            raise RuntimeError(
                f"Generation failed in background thread: {errors[0]}"
            ) from errors[0]
        raise

    thread.join(timeout=5.0)

    # Check if the background thread raised after streamer finished normally
    if errors:
        raise RuntimeError(
            f"Generation failed in background thread: {errors[0]}"
        ) from errors[0]

    logger.info("Streamed %d tokens", token_count)


def _generate_in_thread(
    model: torch.nn.Module,
    generation_kwargs: dict[str, object],
    errors: list[BaseException],
    seed: int | None = None,
) -> None:
    """Run model.generate() in a background thread.

    Sets the RNG seed inside this thread (not the caller) because
    PyTorch's CPU RNG state is thread-local. If an exception occurs
    (OOM, CUDA error, etc.), it is captured into the ``errors`` list
    so the main thread can detect and re-raise it.

    Args:
        model: The loaded language model.
        generation_kwargs: Keyword arguments for model.generate().
        errors: Mutable list to store any exception from this thread.
        seed: Optional random seed for reproducible generation.
    """
    try:
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            model.generate(**generation_kwargs)
    except BaseException as exc:
        logger.exception("Background generation thread failed: %s", exc)
        errors.append(exc)
