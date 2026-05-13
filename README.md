# ⚡ Mini LLM Serving Platform

**Production-grade LLM inference server with priority queues, dynamic batching, speculative decoding, GPU memory profiling, and SSE streaming.**

```
Client ──▶ ALB ──▶ [LoadShedder] ──▶ [RateLimiter] ──▶ Priority Queue
                                                              │
                                                    Worker Pool (N threads)
                                                              │
                                                    [CircuitBreaker] ──▶ Model
                                                              │
                                                     SSE Stream ──▶ Client
```

---

## Features

1. ⚡ **Model Loading & Management** — HuggingFace model auto-download, device selection (CPU/CUDA/MPS), tokenizer caching
2. 📡 **SSE Streaming** — Token-by-token Server-Sent Events via `asyncio.Queue` bridge (sync generator → async SSE)
3. 🔒 **Concurrency Control** — Thread pool executor with configurable worker count, `asyncio.wait_for` timeout
4. 📋 **Priority Queues** — Redis sorted set (ZADD/ZPOPMIN) with 3 priority levels: critical > standard > batch
5. 🚦 **Token Bucket Rate Limiting** — Redis Lua script for atomic token bucket (per-client, configurable refill rate)
6. 🛡️ **Load Shedding** — Middleware rejects requests when queue depth exceeds threshold (503 + Retry-After)
7. ⚡ **Circuit Breaker** — 3-state (CLOSED → OPEN → HALF_OPEN) with failure threshold, recovery timeout, half-open probe
8. 📦 **Dynamic Batching** — `BatchScheduler` groups requests by timeout OR size trigger, reduces per-request overhead
9. 💾 **KV-Cache** — `OrderedDict` LRU with configurable memory budget, prompt-hash keying, eviction on pressure
10. 🔮 **Speculative Decoding** — Draft model + target verifier with rejection sampling, acceptance rate metrics
11. 📊 **GPU Memory Profiling** — Real-time tracker, watermark monitor (warning/critical thresholds), pressure handler (admission control)
12. 🔄 **Graceful Shutdown** — FastAPI lifespan: drain in-flight requests, close Redis, shutdown executor
13. 📈 **Health Probes** — `/health` (3-state liveness: healthy/degraded/unhealthy) + `/ready` (ALB readiness: model loaded?)

---

## Quick Start

### Docker Compose

```bash
git clone https://github.com/gli-27/mini-llm-serving.git
cd mini-llm-serving

# Start API + Redis
docker compose up -d

# Verify
curl http://localhost:8000/health
curl http://localhost:8000/ready
```

### Local Development

```bash
pip install -e ".[dev]"

# Start Redis (required)
docker run -d -p 6379:6379 redis:7-alpine

# Run server
uvicorn llm_serving.main:app --reload
```

### Test a Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "The meaning of life is", "max_tokens": 50}'
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | 3-state health check (healthy/degraded/unhealthy) |
| `GET` | `/ready` | Readiness probe (model loaded?) |
| `POST` | `/v1/completions` | Generate completion (sync or SSE streaming) |
| `POST` | `/v1/completions/speculative` | Generate with speculative decoding |
| `GET` | `/v1/models` | List available models + readiness |
| `GET` | `/v1/cache/stats` | KV cache hit/miss/eviction stats |
| `GET` | `/v1/memory/stats` | GPU memory usage snapshot |
| `GET` | `/v1/memory/watermarks` | Memory watermark thresholds + violations |
| `GET` | `/v1/memory/pressure` | Current memory pressure level |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                            │
│                                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ LoadShedder  │──│ RateLimiter  │──│  API Router              │   │
│  │ (middleware)  │  │ (middleware)  │  │  /health /ready          │   │
│  └──────────────┘  └──────────────┘  │  /v1/completions          │   │
│                                       │  /v1/models               │   │
│                                       └────────────┬─────────────┘   │
│                                                     │                 │
│                                          ┌──────────▼──────────┐     │
│                                          │   Priority Queue     │     │
│                                          │   (Redis ZADD/      │     │
│                                          │    ZPOPMIN)          │     │
│                                          └──────────┬──────────┘     │
│                                                     │                 │
│  ┌──────────────────────────────────────────────────▼──────────────┐ │
│  │                   Worker Pool (N threads)                        │ │
│  │  ┌───────────┐  ┌───────────┐  ┌──────────────┐  ┌──────────┐ │ │
│  │  │ Circuit   │  │ Dynamic   │  │  KV-Cache    │  │ Memory   │ │ │
│  │  │ Breaker   │──│ Batcher   │──│  (LRU)       │──│ Profiler │ │ │
│  │  └───────────┘  └───────────┘  └──────────────┘  └──────────┘ │ │
│  └──────────────────────────────┬──────────────────────────────────┘ │
│                                  │                                    │
│                       ┌──────────▼──────────┐                        │
│                       │   Model Manager      │                        │
│                       │   (HuggingFace)      │                        │
│                       │                      │                        │
│                       │   ┌──────────────┐   │                        │
│                       │   │  Speculative  │   │                        │
│                       │   │  Decoding     │   │                        │
│                       │   │  (draft +     │   │                        │
│                       │   │   verifier)   │   │                        │
│                       │   └──────────────┘   │                        │
│                       └──────────────────────┘                        │
│                                  │                                    │
│                       ┌──────────▼──────────┐                        │
│                       │  SSE Streaming       │                        │
│                       │  (asyncio.Queue      │                        │
│                       │   bridge)            │                        │
│                       └──────────────────────┘                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Configuration

All settings via environment variables with `LLM_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | HuggingFace model name |
| `LLM_DEVICE` | `auto` | Device: `cpu`, `cuda`, `mps`, `auto` |
| `LLM_MAX_CONCURRENT` | `4` | Max concurrent inference threads |
| `LLM_GENERATION_TIMEOUT_S` | `30.0` | Per-request timeout (seconds) |
| `LLM_REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `LLM_QUEUE_MAX_SIZE` | `100` | Priority queue max depth |
| `LLM_RATE_LIMIT_RPM` | `60` | Requests per minute per client |
| `LLM_LOAD_SHED_THRESHOLD` | `50` | Queue depth to start shedding |
| `LLM_CB_FAILURE_THRESHOLD` | `5` | Circuit breaker open threshold |
| `LLM_CB_RECOVERY_TIMEOUT` | `30` | Seconds before half-open probe |
| `LLM_BATCH_MAX_SIZE` | `8` | Dynamic batch max size |
| `LLM_BATCH_TIMEOUT_MS` | `50` | Batch accumulation timeout (ms) |
| `LLM_KV_CACHE_BUDGET_MB` | `512` | KV-cache memory budget |
| `LLM_SPEC_ENABLED` | `false` | Enable speculative decoding |
| `LLM_MEMORY_WARNING_PCT` | `0.80` | Memory warning watermark |
| `LLM_MEMORY_CRITICAL_PCT` | `0.95` | Memory critical watermark |

---

## Performance

> **Placeholder** — to be filled with actual benchmark results.

| Metric | Value |
|--------|-------|
| p50 latency (TinyLlama, 50 tokens) | _TBD_ |
| p95 latency | _TBD_ |
| p99 latency | _TBD_ |
| Throughput (requests/sec) | _TBD_ |
| Speculative decoding acceptance rate | _TBD_ |
| KV-cache hit rate | _TBD_ |

### Load Testing

```bash
# Install locust
pip install locust

# Run load test
locust -f tests/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 60s --headless
```

---

## Project Structure

```
mini-llm-serving/
├── src/llm_serving/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── config.py               # pydantic-settings (LLM_ prefix)
│   ├── exceptions.py           # Custom exceptions
│   ├── logging.py              # structlog configuration
│   ├── api/
│   │   ├── router.py           # /health, /ready, /v1/completions, /v1/models
│   │   ├── schemas.py          # Request/response Pydantic models
│   │   └── memory_router.py    # /v1/memory/* endpoints
│   ├── core/
│   │   ├── inference.py        # Synchronous generate() function
│   │   ├── streaming.py        # Token-by-token generator
│   │   ├── worker.py           # Background worker pool (ZPOPMIN consumer)
│   │   ├── batcher.py          # Dynamic batch scheduler
│   │   ├── circuit_breaker.py  # 3-state circuit breaker
│   │   └── kv_cache.py         # LRU KV-cache with memory budget
│   ├── middleware/
│   │   ├── load_shedder.py     # Queue depth load shedding
│   │   ├── rate_limit.py       # Token bucket rate limiter
│   │   └── error_handler.py    # Global exception handler
│   ├── models/
│   │   └── loader.py           # HuggingFace model loading + management
│   ├── queue/
│   │   ├── redis_client.py     # Redis connection wrapper
│   │   ├── priority_queue.py   # Sorted set priority queue
│   │   └── rate_limiter.py     # Lua-based token bucket
│   ├── speculative/
│   │   ├── orchestrator.py     # Speculative decoding coordinator
│   │   ├── draft_runner.py     # Draft model inference
│   │   ├── verifier.py         # Target model verification
│   │   ├── sampler.py          # Rejection sampling
│   │   ├── metrics.py          # Acceptance rate tracking
│   │   └── config.py           # Speculative decoding config
│   └── profiler/
│       ├── tracker.py          # Real-time memory tracking
│       ├── watermark.py        # Warning/critical thresholds
│       ├── pressure.py         # Admission control handler
│       ├── memory_estimator.py # Per-request memory estimation
│       └── config.py           # Profiler configuration
├── tests/                      # 192 tests
├── Dockerfile                  # Multi-stage build
├── docker-compose.yml          # API + Redis
├── Makefile                    # Dev commands
└── pyproject.toml              # Dependencies + ruff + pytest config
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| Framework | FastAPI + uvicorn |
| ML Framework | PyTorch + HuggingFace Transformers |
| Queue/Cache | Redis (sorted sets, Lua scripts) |
| Validation | Pydantic v2 |
| Logging | structlog (JSON) |
| Testing | pytest + pytest-asyncio + fakeredis |
| Linting | ruff |
| Container | Docker (multi-stage) |

---

## Companion Project

| | mini-llm-serving | mini-agent-orchestrator |
|---|---|---|
| **Role** | Single-inference optimization | Multi-inference orchestration |
| **Analogy** | The engine | The conductor |
| **Focus** | GPU memory, KV-cache, speculative decoding, batching | DAG scheduling, parallel dispatch, data flow, persistence |
| **Repo** | [gli-27/mini-llm-serving](https://github.com/gli-27/mini-llm-serving) | [gli-27/mini-agent-orchestrator](https://github.com/gli-27/mini-agent-orchestrator) |

> **This project** optimizes how a single LLM inference request is served. **The companion** orchestrates multiple such requests across agents in a DAG.

---

## Testing

```bash
pytest tests/ -v                    # All 192 tests
pytest tests/ -v -k "streaming"     # Filter by keyword
pytest tests/ --cov=llm_serving     # With coverage
```

---

## License

MIT
