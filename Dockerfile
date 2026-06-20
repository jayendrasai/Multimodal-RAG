# syntax=docker/dockerfile:1

# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install build deps only in this stage — they won't exist in the final image
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .

# 1. Install CPU-only torch FIRST
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 2. Install requirements, forcing pip to check the CPU index first so it 
# doesn't replace the torch installation with the default CUDA versions.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd --gid 1001 appgroup \
    && useradd --uid 1001 --gid appgroup --shell /bin/sh --create-home appuser

# Runtime-only system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libmagic1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pull compiled packages from builder
COPY --from=builder /install /usr/local

WORKDIR /code

# Copy application source
COPY --chown=appuser:appgroup . .

USER appuser

EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--no-access-log", \
    "--workers", "1"]