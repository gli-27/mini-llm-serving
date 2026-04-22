# Mini LLM Serving — AI Context

## Project Overview
A production-grade LLM inference serving platform. Serves TinyLlama 1.1B via FastAPI with dynamic request batching, token bucket rate limiting, streaming responses (SSE), and full observability. Deployed on AWS ECS Fargate with CI/CD via GitHub Actions.

## Architecture

```
Client → ALB → FastAPI (API Gateway)
                   ↓
            Redis (Request Queue + Rate Limit State)
                   ↓
             Batch Scheduler (collects requests up to MAX_BATCH_SIZE or MAX_WAIT_TIME_MS)
                   ↓
            Inference Worker (HuggingFace Transformers, TinyLlama 1.1B)
                   ↓
           SSE Streaming Response → Client
                   ↓
           CloudWatch Metrics → ECS Autoscaling
```

## Key Design Decisions
1. **Dynamic Batching**: Requests are queued in Redis. A batch scheduler collects up to `MAX_BATCH_SIZE` requests or waits `MAX_WAIT_TIME_MS`, whichever comes first, then runs a single batched forward pass.
2. **Token Bucket Rate Limiting**: Per-API-key rate limiting implemented in Redis using a token bucket algorithm. Atomic operations via Lua scripts.
3. **Streaming**: SSE (Server-Sent Events) via `sse-starlette` for real-time token-by-token output.
4. **Structured Logging**: `structlog` with JSON output for machine-parseable logs.
5. **Distributed Tracing**: OpenTelemetry with OTLP exporter for end-to-end request tracing.
6. **Infrastructure**: ECS Fargate (serverless containers), ElastiCache Redis, ALB, CloudWatch.

## Tech Stack
- **Language**: Python 3.11+
- **Framework**: FastAPI + Uvicorn
- **Model**: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (HuggingFace Transformers)
- **Queue/Cache**: Redis 7
- **Infra**: AWS ECS Fargate, ALB, ElastiCache, CloudWatch, S3, ECR
- **CI/CD**: GitHub Actions
- **Observability**: structlog, OpenTelemetry, Prometheus client, CloudWatch

## Module Map
| Module | Responsibility |
|--------|---------------|
| `llm_serving/main.py` | FastAPI app, lifespan, middleware |
| `llm_serving/core/config.py` | Pydantic settings from env vars |
| `llm_serving/core/inference.py` | Batch scheduler, inference loop |
| `llm_serving/api/router.py` | HTTP endpoints (`/v1/completions`, `/health`) |
| `llm_serving/api/schemas.py` | Pydantic request/response models |
| `llm_serving/models/loader.py` | Model + tokenizer loading, caching |

## Conventions
- All source code under `src/llm_serving/`
- Tests under `tests/`
- Config via environment variables (Pydantic Settings)
- Type hints everywhere; `mypy --strict`
- Linting with `ruff`
- Async-first design (FastAPI async endpoints)

## Important Constraints
- No GPU assumed — CPU inference with TinyLlama for demo purposes
- Single-process inference worker (no multi-GPU)
- Redis is required for both batching queue and rate limiting
- Docker multi-stage build for small image size
