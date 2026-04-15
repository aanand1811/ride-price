# ── Stage 1: Build / train ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy source
COPY train.py .
COPY api/ api/

# Train model using synthetic data (replace --synthetic with --data /data/rides.csv
# when mounting a real dataset volume)
RUN python train.py --synthetic --output models


# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy app and trained model
COPY --from=builder /app/api/ api/
COPY --from=builder /app/models/ models/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

ENV MODEL_DIR=models

# Uvicorn with 2 workers per CPU (adjust via K8s env vars)
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
