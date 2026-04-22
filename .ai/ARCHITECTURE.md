# Architecture Decision Records

## ADR-001: FastAPI over Flask/Django
Decision: Use FastAPI
Rationale: Native async, Pydantic validation, StreamingResponse for SSE. Industry standard for ML serving (vLLM, TGI).

## ADR-002: TinyLlama as base model
Decision: TinyLlama-1.1B-Chat-v1.0
Rationale: Small enough for CPU dev (~2.2GB). Model-agnostic infra — swap by changing one env var.

## ADR-003: Singleton ModelManager
Decision: Global ModelManager, loaded at startup
Rationale: Model loading is expensive. Load once in lifespan, share across handlers.

## ADR-004: Environment-driven config
Decision: pydantic-settings with LLM_ prefix
Rationale: 12-factor app. Same code local/Docker/ECS, only env vars change.

## ADR-005: Layered architecture
Decision: Separate api/, models/, core/
Rationale: api/ = HTTP, models/ = lifecycle, core/ = logic. Unit test core/ without HTTP. Swap backends without touching API.
