# =============================================================================
# RoadSentinel Studio v3.0 — Production Container Image
# Multi-Architecture: linux/amd64, linux/arm64 (AWS Graviton 3/4)
# =============================================================================

FROM python:3.11-slim as base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8000 \
    HOST=0.0.0.0

# Install essential system dependencies for OpenCV and multimedia processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and assets
COPY src/ /app/src/
COPY saas_app.py /app/
COPY README.md /app/

# Create runtime directories for state, evidence, and uploads
RUN mkdir -p /app/artifacts /app/weights /app/videos /app/data /app/temp_uploads

# Expose standard Studio port
EXPOSE 8000

# Container Healthcheck verifying FastAPI Studio readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/system/health || exit 1

# Launch the unified RoadSentinel Studio server
CMD ["python", "saas_app.py"]
