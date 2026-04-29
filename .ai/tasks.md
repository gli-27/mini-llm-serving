# Implementation Tasks

## Phase 1: Core API (Milestone 1) — COMPLETE
- [x] `config.py` — Pydantic Settings with LLM_ prefix, env var loading
- [x] `models/loader.py` — Load TinyLlama model + tokenizer, singleton ModelManager
- [x] `api/schemas.py` — Request/response Pydantic models (CompletionRequest, CompletionResponse, Usage)
- [x] `api/router.py` — `/v1/completions` (POST), `/health` (GET), `/v1/models` (GET)
- [x] `main.py` — FastAPI app with lifespan, error handlers
- [x] `core/inference.py` — Single-request inference logic
- [x] `exceptions.py` — Centralized exception classes (ModelNotLoadedError, GenerationTimeoutError)

## Phase 2: SSE Streaming (Milestone 2) — COMPLETE
- [x] `core/streaming.py` — Token-by-token streaming via TextIteratorStreamer + background thread
- [x] `api/schemas.py` — Add `stream: bool` field to CompletionRequest, `StreamChunk` model
- [x] `api/router.py` — StreamingResponse path when `stream=true`, SSE format, client disconnect detection
- [x] Seed thread safety — `torch.manual_seed()` set inside background thread (PyTorch RNG is thread-local)

## Phase 2.5: Quality Gaps — COMPLETE
- [x] Structured logging — Replaced stdlib `logging` with `structlog` across all modules (`logging.py`)
- [x] `app_env` config field — Controls JSON (production) vs console (development) log rendering
- [x] Test suite — `conftest.py`, `test_config.py`, `test_loader.py`, `test_inference.py`, `test_streaming.py`, `test_api.py`, `test_logging.py`
- [x] Dockerfile — Multi-stage build, non-root user, healthcheck
- [x] docker-compose — API + Redis services

## Phase 3: Dynamic Batching + Priority Queues (Milestone 3)
- [ ] Redis config in Settings — `redis_url`, connection management
- [ ] Redis request queue — push incoming requests, pop in batches
- [ ] `core/batching.py` — Batch scheduler: collect up to MAX_BATCH_SIZE or wait MAX_WAIT_TIME_MS
- [ ] Priority-aware scheduling — Redis sorted set or dual lists (queue:high + queue:default)
- [ ] Batched forward pass — group requests by model, run single batched inference
- [ ] Response routing — map batch outputs back to individual request futures

## Phase 4: Rate Limiting + Priority Tiers (Milestone 4)
- [ ] Token bucket algorithm in Redis (Lua script for atomic decrement)
- [ ] Per-API-key rate limiting with configurable tiers (high/default)
- [ ] FastAPI dependency for rate limit enforcement
- [ ] 429 responses with `Retry-After` header and remaining quota info

## Phase 5: Multi-Model Routing (Milestone 5)
- [ ] Model registry — dict mapping model names to loaded model instances
- [ ] Request dispatch — route to correct model based on request's `model` field
- [ ] `models/loader.py` — Support loading multiple models at startup
- [ ] `/v1/models` endpoint — return all available models with status

## Phase 6: Load Shedding + Graceful Degradation (Milestone 6)
- [ ] `middleware/load_shedding.py` — Monitor queue depth, reject when overloaded
- [ ] Priority-aware rejection — shed low-priority requests first (503)
- [ ] Circuit breaker pattern — detect unhealthy model backends, fail fast
- [ ] Graceful shutdown — drain in-flight requests before stopping

## Phase 7: Observability (Milestone 7)
- [x] Structured logging with structlog (JSON format)
- [ ] Correlation ID propagation through all layers
- [ ] OpenTelemetry tracing instrumentation
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] CloudWatch metrics publishing (latency, throughput, queue depth, batch size)
- [ ] `middleware/tracing.py` — Request tracing middleware

## Phase 8: Infrastructure & Deployment (Milestone 8)
- [ ] CDK (Python) for IaC — ECS Fargate, ALB, ElastiCache, CloudWatch
- [ ] ECS task definition (Fargate)
- [ ] ALB + target group configuration
- [ ] ElastiCache Redis cluster
- [ ] CloudWatch alarms + autoscaling policies (latency, queue depth)
- [ ] S3 bucket for model artifacts (optional)
- [ ] ECR repository

## Phase 9: Testing & Benchmarks (Milestone 9)
- [x] Unit tests for config, schemas, loader, inference
- [x] Integration tests for API endpoints (sync + streaming)
- [x] Streaming tests (token generation, seed thread safety, error handling)
- [ ] Batching tests (concurrent requests → single batch)
- [ ] Rate limiting tests (token bucket, priority tiers)
- [ ] Load shedding tests (queue depth triggers)
- [ ] Multi-model routing tests
- [ ] Locust load test script
- [ ] Performance benchmark report (latency p50/p95/p99, throughput)
