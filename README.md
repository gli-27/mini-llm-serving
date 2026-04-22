# Mini LLM Serving Platform

A production-grade LLM inference serving system built with FastAPI, featuring dynamic request batching, token bucket rate limiting, and deployment on AWS ECS Fargate.

## Features

- **Dynamic Request Batching** — Redis-backed batch scheduler for optimal GPU/CPU utilization
- **Token Bucket Rate Limiting** — Per-API-key rate limiting via Redis
- **Streaming Responses** — Server-Sent Events (SSE) for real-time token streaming
- **Dockerized Deployment** — Multi-stage Docker build, deployed on AWS ECS Fargate
- **Observability** — Structured logging (structlog), distributed tracing (OpenTelemetry), CloudWatch metrics
- **Autoscaling** — CloudWatch-driven ECS autoscaling based on request latency and queue depth
- **CI/CD** — GitHub Actions pipeline: lint → test → build → push to ECR → deploy to ECS

## Architecture

```
Client → ALB → API Gateway (FastAPI)
                    ↓
             Request Queue (Redis)
                    ↓
              Batch Scheduler
                    ↓
         Inference Workers (ECS tasks)
                    ↓
        Streaming Response (SSE) → Client
                    ↓
        CloudWatch Metrics → Autoscaling
```

## Tech Stack

| Layer        | Technology                                    |
|-------------|-----------------------------------------------|
| API         | Python 3.11+, FastAPI, Uvicorn                |
| Model       | TinyLlama 1.1B (HuggingFace Transformers)     |
| Queue/Cache | Redis                                         |
| Infra       | AWS ECS Fargate, ALB, ElastiCache, CloudWatch  |
| CI/CD       | GitHub Actions → ECR → ECS                    |
| Observability | structlog, OpenTelemetry, Prometheus, CloudWatch |

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Start Redis
docker compose up redis -d

# Run the API server
uvicorn llm_serving.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
# Build and start all services
docker compose up --build

# Or use Make
make docker-up
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

See [.env.example](.env.example) for all available configuration options.

## Development

```bash
# Install dev dependencies
make dev

# Run linter + type checker
make lint

# Run tests with coverage
make test

# Clean build artifacts
make clean
```

## Project Structure

```
src/llm_serving/
├── __init__.py
├── main.py            # FastAPI app entrypoint
├── core/
│   ├── __init__.py
│   ├── config.py      # Pydantic settings
│   └── inference.py   # Model inference + batch processing
├── api/
│   ├── __init__.py
│   ├── router.py      # API routes
│   └── schemas.py     # Request/response models
└── models/
    ├── __init__.py
    └── loader.py       # Model loading + caching
tests/
├── __init__.py
├── test_api.py
├── test_batching.py
└── test_rate_limit.py
```

## License

MIT
