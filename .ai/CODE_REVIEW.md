# Code Review Feedback

No reviews yet.

## Patterns to Follow
- Type hints on all function signatures
- Pydantic models for all API request/response (no raw dicts)
- logging module, not print()
- Proper HTTP status codes, not just 500
- Config from env vars, never hardcode
