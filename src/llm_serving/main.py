"""FastAPI application entrypoint with lifespan-managed model loading."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_serving.api.router import router
from llm_serving.config import get_settings
from llm_serving.models.loader import ModelManager

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup: loads settings, creates ModelManager, loads the model,
    and stores the manager in app.state for dependency injection.

    On shutdown: logs a shutdown message.
    """
    settings = get_settings()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

    logger.info("Starting LLM Serving Platform")
    logger.info("Settings: %s", settings.model_dump_json())

    # Load model
    model_manager = ModelManager(settings)
    start = time.perf_counter()
    model_manager.load_model()
    elapsed = time.perf_counter() - start
    logger.info("Model loaded in %.2fs", elapsed)

    app.state.model_manager = model_manager

    yield

    logger.info("Shutting down LLM Serving Platform")


app = FastAPI(
    title="Mini LLM Serving Platform",
    description="Production-grade LLM inference with dynamic batching and rate limiting.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
