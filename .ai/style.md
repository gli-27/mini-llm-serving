# Code Style Guide

## Python
- **Version**: 3.11+
- **Formatter/Linter**: ruff (line-length=100)
- **Type Checking**: mypy --strict
- **Import Order**: stdlib → third-party → local (enforced by ruff `I` rules)

## Naming
- `snake_case` for functions, variables, modules
- `PascalCase` for classes
- `UPPER_SNAKE_CASE` for constants and env vars
- Prefix private helpers with `_`

## FastAPI Patterns
- Use `async def` for all endpoints
- Dependency injection via `Depends()`
- Pydantic v2 models for all request/response schemas
- Return explicit status codes
- Use `HTTPException` for error responses

## Error Handling
- Never swallow exceptions silently
- Use structured logging for all errors
- Return proper HTTP status codes (400, 401, 429, 500, 503)
- Include `request_id` in all error responses

## Testing
- pytest + pytest-asyncio
- Use `httpx.AsyncClient` for API tests
- Mock external services (Redis, model) in unit tests
- Aim for >80% coverage on core modules

## Docstrings
- Google-style docstrings for public functions/classes
- Include Args, Returns, Raises sections where applicable

## Git
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `chore:`, `refactor:`
- Small, focused commits
- PR-based workflow
