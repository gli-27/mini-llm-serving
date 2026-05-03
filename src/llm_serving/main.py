"""FastAPI application entrypoint with lifespan-managed model loading."""

import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_serving.api.router import router
from llm_serving.config import get_settings
from llm_serving.core.circuit_breaker import CircuitBreaker
from llm_serving.core.worker import InferenceWorkerPool
from llm_serving.logging import get_logger, setup_logging
from llm_serving.middleware.error_handler import global_exception_handler
from llm_serving.middleware.load_shedder import LoadShedderMiddleware
from llm_serving.middleware.rate_limit import RateLimitMiddleware
from llm_serving.models.loader import ModelManager
from llm_serving.queue.priority_queue import PriorityQueue
from llm_serving.queue.rate_limiter import TokenBucketRateLimiter
from llm_serving.queue.redis_client import RedisClient

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown.

    On startup: configures structured logging, loads settings, creates
    ModelManager, loads the model, creates a ThreadPoolExecutor for
    inference concurrency control, connects to Redis, and initializes
    the rate limiter and priority queue.

    On shutdown: closes Redis, shuts down the executor, and logs.
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

    # Connect to Redis
    redis_client = RedisClient(settings.redis_url)
    await redis_client.connect()

    # Create rate limiter and priority queue
    rate_limiter = TokenBucketRateLimiter(
        redis_client,
        bucket_size=settings.rate_limit_bucket_size,
        refill_rate=settings.rate_limit_refill_rate,
    )
    priority_queue = PriorityQueue(redis_client)

    # Create circuit breaker
    circuit_breaker = CircuitBreaker(
        failure_threshold=settings.circuit_breaker_failure_threshold,
        recovery_timeout_s=settings.circuit_breaker_recovery_timeout_s,
    )

    # Create and start background worker pool
    worker_pool = InferenceWorkerPool(
        priority_queue=priority_queue,
        model_manager=model_manager,
        executor=inference_executor,
        circuit_breaker=circuit_breaker,
        num_workers=settings.max_concurrent_requests,
    )
    await worker_pool.start()

    app.state.model_manager = model_manager
    app.state.inference_executor = inference_executor
    app.state.redis_client = redis_client
    app.state.rate_limiter = rate_limiter
    app.state.priority_queue = priority_queue
    app.state.worker_pool = worker_pool
    app.state.circuit_breaker = circuit_breaker
    app.state.settings = settings

    yield

    await worker_pool.stop()
    await redis_client.close()
    inference_executor.shutdown(wait=False)
    logger.info("Shutting down LLM Serving Platform")


app = FastAPI(
    title="Mini LLM Serving Platform",
    description="Production-grade LLM inference with dynamic batching and rate limiting.",
    version="0.1.0",
    lifespan=lifespan,
)

# Global exception handler — catches all unhandled exceptions
app.add_exception_handler(Exception, global_exception_handler)

# Middleware is added in reverse order — last added = first to execute.
# Order: LoadShedder → RateLimit → route handler
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoadShedderMiddleware)

app.include_router(router)
