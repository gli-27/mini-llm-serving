"""FastAPI application entrypoint with lifespan-managed model loading."""

import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_serving.api.router import router
from llm_serving.config import get_settings
from llm_serving.logging import get_logger, setup_logging
from llm_serving.models.loader import ModelManager

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup: configures structured logging, loads settings, creates
    ModelManager, loads the model, and creates a ThreadPoolExecutor for
    inference concurrency control.

    On shutdown: shuts down the executor and logs a message.
    """
    settings = get_settings()

    # Configure structured logging (JSON in production, console in dev)
    json_format = settings.app_env != "development"
    setup_logging(log_level=settings.log_level, json_format=json_format)

    logger.info("Starting LLM Serving Platform")
    logger.info("Settings loaded", settings=settings.model_dump())

    # Load model
    model_manager = ModelManager(settings)
    start = time.perf_counter()
    model_manager.load_model()
    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2fs", elapsed)

    # ThreadPoolExecutor for inference — solves three problems at once:
    # 1. Concurrency control (max_workers limits parallel inference)
    # 2. Non-blocking (sync generate() runs in thread, doesn't block event loop)
    # 3. Timeout support (via asyncio.wait_for wrapping run_in_executor)
    inference_executor = ThreadPoolExecutor(
        max_workers=settings.max_concurrent_requests,
        thread_name_prefix="inference",
    )
    logger.info("Inference thread pool: max_workers=%d", settings.max_concurrent_requests)

    app.state.model_manager = model_manager
    app.state.inference_executor = inference_executor

    yield

    inference_executor.shutdown(wait=False)
    logger.info("Shutting down LLM Serving Platform")


app = FastAPI(
    title="Mini LLM Serving Platform",
    description="Production-grade LLM inference with dynamic batching and rate limiting.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
