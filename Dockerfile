# ==============================================================================
# RetinaCare AI - Production Container Specification
# Optimized for Hugging Face Spaces (CPU Basic: 2 vCPU, 16 GB RAM) & Container Runtimes
# ==============================================================================

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable immediate stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    PORT=7860 \
    HOST=0.0.0.0

# Install system dependencies:
# - libgl1 and libglib2.0-0: Required by OpenCV (cv2) in headless Linux environments
# - curl: For container health checks
# - git and git-lfs: For handling model weight checkpoints
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Setup non-root user for Hugging Face Spaces security sandbox (UID 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY --chown=user:user . /app

# Ensure proper permissions
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Expose standard Hugging Face Spaces port
EXPOSE 7860

# Container health monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Launch production Uvicorn ASGI server
CMD ["sh", "-c", "uvicorn backend.main:app --host ${HOST} --port ${PORT}"]
