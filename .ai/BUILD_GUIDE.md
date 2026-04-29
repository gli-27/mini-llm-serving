# Current Build Guide — Milestone 2: SSE Streaming

Status: COMPLETE

## Objective
Add token-by-token streaming via Server-Sent Events (SSE) to the /v1/completions endpoint.

## What Changed from Milestone 1
- Added `stream: bool = False` to CompletionRequest
- New `StreamChunk` schema for SSE events
- New `core/streaming.py` — TextIteratorStreamer + background thread
- Router branches: `stream=false` → sync response, `stream=true` → SSE StreamingResponse
- Client disconnect detection via `request.is_disconnected()`

## SSE Format
Each token is sent as:
```
data: {"id":"cmpl-<8hex>","object":"text_completion.chunk","model":"<name>","content":"<token>"}\n\n
```

End-of-stream sentinel:
```
data: [DONE]\n\n
```

## Streaming Headers
- Content-Type: text/event-stream
- Cache-Control: no-cache
- Connection: keep-alive
- X-Accel-Buffering: no (for nginx/ALB proxy)

## How It Works
1. Request arrives with `stream: true`
2. Router creates a `StreamingResponse` with an async generator
3. `generate_stream()` tokenizes the prompt, creates a `TextIteratorStreamer`
4. `model.generate()` runs in a background thread (daemon)
5. Main thread yields tokens from the streamer as SSE events
6. On client disconnect, generator exits early
7. `data: [DONE]` sentinel sent at the end

## Testing
```bash
# Streaming with curl
curl -N -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50, "stream": true}'

# Non-streaming (still works)
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_tokens": 50}'
```

## Error Format (OpenAI-compatible)
```json
{"error": {"message": "...", "type": "server_error", "code": 503}}
```

## Definition of Done
- `stream: false` still returns full CompletionResponse (backward compatible)
- `stream: true` streams tokens one-by-one via SSE
- SSE format: `data: {json}\n\n` per token, `data: [DONE]\n\n` sentinel
- Client disconnect detected and logged
- Errors sent as SSE events in OpenAI error format
- StreamingResponse headers set correctly
- All code has type hints and docstrings

## Previous Milestones
### Milestone 1: Project Scaffold + Model Loading — COMPLETE
- FastAPI app skeleton, model loading, sync /v1/completions endpoint
- GET /health, POST /v1/completions, GET /v1/models
- Pydantic settings with LLM_ prefix
- Singleton ModelManager loaded at startup via lifespan
