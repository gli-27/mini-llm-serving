"""FastAPI application entrypoint with lifespan-managed model loading."""

import time
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from llm_serving.api.router import router
from llm_serving.config import get_settings
from llm_serving.core.batcher import BatchScheduler
from llm_serving.core.circuit_breaker import CircuitBreaker
from llm_serving.core.kv_cache import KVCacheManager
from llm_serving.core.worker import InferenceWorkerPool
from llm_serving.logging import get_logger, setup_logging
from llm_serving.middleware.error_handler import global_exception_handler
from llm_serving.middleware.load_shedder import LoadShedderMiddleware
from llm_serving.middleware.rate_limit import RateLimitMiddleware
from llm_serving.models.loader import ModelManager
from llm_serving.queue.priority_queue import PriorityQueue
from llm_serving.queue.rate_limiter import TokenBucketRateLimiter
from llm_serving.queue.redis_client import RedisClient
from llm_serving.speculative.config import SpeculativeConfig
from llm_serving.speculative.draft_runner import DraftModelRunner
from llm_serving.speculative.orchestrator import SpeculativeOrchestrator
from llm_serving.speculative.verifier import TokenVerifier

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

    # Create KV cache manager (if enabled)
    kv_cache_manager: KVCacheManager | None = None
    if settings.kv_cache_enabled:
        kv_cache_manager = KVCacheManager(
            max_memory_bytes=settings.kv_cache_max_memory_mb * 1024 * 1024,
            max_entries=settings.kv_cache_max_entries,
        )
        logger.info(
            "KV cache enabled",
            max_memory_mb=settings.kv_cache_max_memory_mb,
            max_entries=settings.kv_cache_max_entries,
        )

    # Create batch scheduler (if enabled)
    batch_scheduler: BatchScheduler | None = None
    if settings.batching_enabled:
        batch_scheduler = BatchScheduler(
            model_manager=model_manager,
            executor=inference_executor,
            max_batch_size=settings.max_batch_size,
            max_batch_wait_ms=settings.max_batch_wait_ms,
        )
        await batch_scheduler.start()
        logger.info(
            "Dynamic batching enabled",
            max_batch_size=settings.max_batch_size,
            max_batch_wait_ms=settings.max_batch_wait_ms,
        )

    # Create and start background worker pool
    worker_pool = InferenceWorkerPool(
        priority_queue=priority_queue,
        model_manager=model_manager,
        executor=inference_executor,
        circuit_breaker=circuit_breaker,
        batch_scheduler=batch_scheduler,
        num_workers=settings.max_concurrent_requests,
    )
    await worker_pool.start()

    app.state.model_manager = model_manager
    app.state.inference_executor = inference_executor
    app.state.redis_client = redis_client
    app.state.rate_limiter = rate_limiter
    app.state.priority_queue = priority_queue
    app.state.worker_pool = worker_pool
    # Create speculative decoding orchestrator (if enabled)
    spec_orchestrator: SpeculativeOrchestrator | None = None
    if settings.spec_enabled:
        spec_config = SpeculativeConfig(
            enabled=True,
            draft_model_name=settings.spec_draft_model_name,
            num_draft_tokens=settings.spec_num_draft_tokens,
            temperature=settings.spec_temperature,
        )
        draft_runner = DraftModelRunner(
            model_name=settings.spec_draft_model_name,
            device=settings.device,
        )
        draft_runner.load()
        verifier = TokenVerifier(target_model=model_manager.model)
        spec_orchestrator = SpeculativeOrchestrator(
            draft_runner=draft_runner,
            verifier=verifier,
            config=spec_config,
        )
        logger.info(
            "Speculative decoding enabled",
            draft_model=settings.spec_draft_model_name,
            num_draft_tokens=settings.spec_num_draft_tokens,
        )

    app.state.circuit_breaker = circuit_breaker
    app.state.kv_cache_manager = kv_cache_manager
    app.state.spec_orchestrator = spec_orchestrator
    app.state.settings = settings

    yield

    await worker_pool.stop()
    if batch_scheduler:
        await batch_scheduler.stop()
    if kv_cache_manager:
        kv_cache_manager.clear()
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
