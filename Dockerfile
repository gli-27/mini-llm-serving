# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install only runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src /app/src

# Non-root user
RUN useradd --create-home appuser \
    && mkdir -p /home/appuser/.cache/huggingface/hub \
    && chown -R appuser:appuser /home/appuser/.cache
USER appuser

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "llm_serving.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
