# Multi-stage production container for VN-CoastSeg-Mask2Former
FROM python:3.11-slim-bullseye

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    APP_PORT=7860

WORKDIR /app

# Install system dependencies for geospatial libraries (GDAL, libgl1, libgomp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy repository source code and assets
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY checkpoints/ /app/checkpoints/

# Create non-root user for security (UID 1000 required for Hugging Face Spaces)
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose standard Gradio port
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Launch Gradio application
CMD ["python", "src/app.py"]
