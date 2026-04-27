"""SSE streaming inference using TextIteratorStreamer.

Runs model.generate() in a background thread and yields tokens
one-by-one via a TextIteratorStreamer, enabling Server-Sent Events
streaming to the client.
"""

import logging
import threading
from collections.abc import Generator

import torch
from transformers import TextIteratorStreamer

from llm_serving.models.loader import ModelManager, ModelNotLoadedError

logger = logging.getLogger(__name__)


def generate_stream(
    model_manager: ModelManager,
    prompt: str,
    max_new_tokens: int,
) -> Generator[str, None, None]:
    """Generate text token-by-token as a streaming generator.

    Uses HuggingFace's TextIteratorStreamer to yield tokens as they are
    generated. The model.generate() call runs in a background thread so
    the main thread can yield tokens to the SSE response as they arrive.

    Args:
        model_manager: The ModelManager holding the loaded model and tokenizer.
        prompt: The input text prompt.
        max_new_tokens: Maximum number of new tokens to generate.

    Yields:
        Individual token strings as they are generated.

    Raises:
        ModelNotLoadedError: If the model has not been loaded yet.
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

    # Create a streamer that yields tokens one-by-one
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    # Generation kwargs for the background thread
    generation_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }

    # Run model.generate() in a background thread — it blocks until done,
    # but the streamer yields tokens to our main thread as they're produced
    thread = threading.Thread(
        target=_generate_in_thread,
        args=(model, generation_kwargs),
        daemon=True,
    )
    thread.start()

    # Yield tokens as they arrive from the streamer
    token_count = 0
    for token_text in streamer:
        if token_text:
            token_count += 1
            yield token_text

    thread.join()
    logger.info("Streamed %d tokens", token_count)


def _generate_in_thread(
    model: torch.nn.Module,
    generation_kwargs: dict[str, object],
) -> None:
    """Run model.generate() in a background thread.

    Args:
        model: The loaded language model.
        generation_kwargs: Keyword arguments for model.generate().
    """
    with torch.no_grad():
        model.generate(**generation_kwargs)
