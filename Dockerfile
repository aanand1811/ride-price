# ── Stage 1: Build / train ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY train.py .
COPY api/ api/

# Train model using synthetic data
RUN python train.py --synthetic --output models


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy globally installed packages and binaries from builder
COPY --from=builder /usr/local /usr/local

# Copy app and trained model
COPY --from=builder /app/api/ api/
COPY --from=builder /app/models/ models/

# Non-root user for security
RUN useradd -m appuser && \
    chown -R appuser /app

USER appuser

EXPOSE 8000

ENV MODEL_DIR=models

# Start API
CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]