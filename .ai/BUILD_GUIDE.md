# Current Build Guide — Milestone 1: Project Scaffold + Model Loading

Status: IN PROGRESS

## Objective
Build the foundation: FastAPI app skeleton, model loading, and a sync /v1/completions endpoint.

## Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Returns {"status": "healthy", "model": "<name>"} |
| POST | /v1/completions | Sync text generation |
| GET | /v1/models | List loaded models |

## POST /v1/completions Request
{"prompt": "string 1-4096 chars required", "max_tokens": "int 1-2048 default 256", "temperature": "float 0.0-2.0 default 0.7"}

## POST /v1/completions Response
{"id": "cmpl-<8hex>", "object": "text_completion", "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "content": "generated text", "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}}

## Config (env vars, LLM_ prefix via pydantic-settings)
- LLM_MODEL_NAME = TinyLlama/TinyLlama-1.1B-Chat-v1.0
- LLM_DEVICE = cpu
- LLM_MAX_NEW_TOKENS = 256
- LLM_HOST = 0.0.0.0
- LLM_PORT = 8000

## Key Design Decisions
1. Model loads at startup via FastAPI lifespan — no requests until ready
2. Singleton ModelManager — load once, share across requests
3. pad_token fallback to eos_token (TinyLlama needs this)
4. All config from env vars, never hardcoded

## Definition of Done
- GET /health returns 200 + model name
- POST /v1/completions returns generated text + usage stats
- GET /v1/models returns model list
- Model loads once at startup (verify in logs)
- Request validation works (empty prompt -> 422)
- All code has type hints
