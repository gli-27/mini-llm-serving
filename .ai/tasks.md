# Implementation Tasks

## Phase 1: Core API
- [ ] `core/config.py` — Pydantic Settings with env var loading
- [ ] `models/loader.py` — Load TinyLlama model + tokenizer from HuggingFace
- [ ] `api/schemas.py` — Request/response Pydantic models (CompletionRequest, CompletionResponse, etc.)
- [ ] `api/router.py` — `/v1/completions` (POST), `/health` (GET) endpoints
- [ ] `main.py` — FastAPI app with lifespan, CORS, error handlers

## Phase 2: Dynamic Batching
- [ ] Redis request queue — push incoming requests, pop in batches
- [ ] Batch scheduler — collect up to MAX_BATCH_SIZE or wait MAX_WAIT_TIME_MS
- [ ] `core/inference.py` — Batched forward pass with HuggingFace model
- [ ] SSE streaming — Token-by-token streaming via SSE

## Phase 3: Rate Limiting
- [ ] Token bucket algorithm in Redis (Lua script for atomicity)
- [ ] FastAPI middleware/dependency for per-API-key rate limiting
- [ ] 429 responses with `Retry-After` header

## Phase 4: Observability
- [ ] Structured logging with structlog (JSON format)
- [ ] OpenTelemetry tracing instrumentation
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] CloudWatch metrics publishing (latency, throughput, queue depth)
- [ ] Request ID propagation through all layers

## Phase 5: Infrastructure & Deployment
- [ ] ECS task definition (Fargate)
- [ ] ALB + target group configuration
- [ ] ElastiCache Redis cluster
- [ ] CloudWatch alarms + autoscaling policies
- [ ] S3 bucket for model artifacts (optional)
- [ ] ECR repository

## Phase 6: Testing & Benchmarks
- [ ] Unit tests for config, schemas, loader
- [ ] Integration tests for API endpoints
- [ ] Batching tests (concurrent requests → single batch)
- [ ] Rate limiting tests
- [ ] Locust load test script
- [ ] Performance benchmark report
