# Silent Voice Bridge - Single Container Deployment
# Multi-stage build: Node (Frontend) + Python (Backend)

# ----------------------------
# STAGE 1: Frontend Build (Node)
# ----------------------------
FROM node:18-alpine as frontend-builder

WORKDIR /app/frontend

# Copy frontend dependency files
COPY src/web/frontend/package.json src/web/frontend/package-lock.json ./

# Install dependencies
RUN npm ci

# Copy frontend source code
COPY src/web/frontend/ .

# Build the React app (outputs to /app/frontend/dist)
RUN npm run build


# ----------------------------
# STAGE 2: Backend Base (Python)
# ----------------------------
# Use AMD64 platform for MediaPipe compatibility
FROM --platform=linux/amd64 python:3.14-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------
# STAGE 3: Python Builder
# ----------------------------
FROM base as backend-builder

WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ----------------------------
# STAGE 4: Final Runtime Image
# ----------------------------
FROM base

WORKDIR /app

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.14/site-packages /usr/local/lib/python3.14/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy Application Code
COPY src/ ./src/
COPY config/ ./config/
COPY models/checkpoints/best_model.pth ./models/checkpoints/best_model.pth

# Download MediaPipe Model (hand_landmarker.task) directly to avoid Git binary limits
# Using the official MediaPipe storage URL
RUN mkdir -p models/mediapipe && \
    wget -O models/mediapipe/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# Copy Built Frontend Assets from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./src/web/frontend/dist

# Create directories for data and model checkpoints
RUN mkdir -p data/raw data/processed models/checkpoints

# ---------------------------------------------------
# HUGGING FACE SPACES CONFIGURATION
# ---------------------------------------------------
# 1. Create a non-root user 'user' with UID 1000
RUN useradd -m -u 1000 user

# 2. Create local cache directories for various libraries
#    (Matplotlib, Torch, etc. try to write to /home/user/.cache)
RUN mkdir -p /app/.cache/matplotlib \
    && mkdir -p /app/.cache/torch \
    && mkdir -p /app/.cache/huggingface \
    && chown -R user:user /app

# 3. Switch to the new user
USER user

# Set environment variables for cache
ENV MPLCONFIGDIR=/app/.cache/matplotlib \
    TORCH_HOME=/app/.cache/torch \
    HF_HOME=/app/.cache/huggingface

# Set working directory
WORKDIR /app

# Health check (Using correct port 7860)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import cv2; import mediapipe" || exit 1

# Expose port 7860 (Hugging Face Spaces Default)
EXPOSE 7860

# Default command - Run FastAPI via Uvicorn on 7860
CMD ["uvicorn", "src.web.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
