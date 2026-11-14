# Multi-stage production container for Predictive-RL Autoscaler
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and scripts
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app
ENV TF_ENABLE_ONEDNN_OPTS=0

# Expose FastAPI REST & Prometheus metrics port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production server entrypoint
CMD ["uvicorn", "scripts.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

