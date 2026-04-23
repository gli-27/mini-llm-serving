# Implementation Tasks

## Phase 1: Core API (Milestone 1)
- [ ] `config.py` — Pydantic Settings with LLM_ prefix, env var loading
- [ ] `models/loader.py` — Load TinyLlama model + tokenizer, singleton ModelManager
- [ ] `api/schemas.py` — Request/response Pydantic models (CompletionRequest, CompletionResponse, Usage)
- [ ] `api/router.py` — `/v1/completions` (POST), `/health` (GET), `/v1/models` (GET)
- [ ] `main.py` — FastAPI app with lifespan, CORS, error handlers
- [ ] `core/inference.py` — Single-request inference logic

## Phase 2: SSE Streaming (Milestone 2)
- [ ] `core/streaming.py` — Token-by-token streaming via SSE (sse-starlette)
- [ ] `api/schemas.py` — Add `stream: bool` field to CompletionRequest
- [ ] `api/router.py` — StreamingResponse path when `stream=true`

## Phase 3: Dynamic Batching + Priority Queues (Milestone 3)
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
- [ ] Structured logging with structlog (JSON format, correlation IDs)
- [ ] OpenTelemetry tracing instrumentation
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] CloudWatch metrics publishing (latency, throughput, queue depth, batch size)
- [ ] Request ID propagation through all layers
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
- [ ] Unit tests for config, schemas, loader, inference
- [ ] Integration tests for API endpoints (sync + streaming)
- [ ] Batching tests (concurrent requests → single batch)
- [ ] Rate limiting tests (token bucket, priority tiers)
- [ ] Load shedding tests (queue depth triggers)
- [ ] Multi-model routing tests
- [ ] Locust load test script
- [ ] Performance benchmark report (latency p50/p95/p99, throughput)
