FROM python:3.10-slim

# Prevent interactive prompts and disable Python bytecode / output buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DEMO_MODE=true

# Install system dependencies:
# - build-essential, gcc, g++: C compiler for webrtcvad & miniaudio extensions
# - tk: Tkinter GUI library support
# - libportaudio2, libasound2, libsndfile1: audio I/O libraries for sounddevice
# - ffmpeg: audio decoding for whisper and audio processing
# - curl: healthchecks and web utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    tk \
    libportaudio2 \
    libasound2 \
    libsndfile1 \
    ffmpeg \
    curl \
    swig \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip to ensure clean modern dependency resolution
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first to leverage Docker layer caching
COPY requirements-assistant.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-assistant.txt

# Install Playwright Chromium and its required OS dependencies
RUN playwright install --with-deps chromium

# Create mount points for persistent directories and logs
RUN mkdir -p /app/memory_store /app/models /app/logs

# Copy application source code (filtered by .dockerignore)
COPY . /app

# Application entry point
CMD ["python", "main.py"]

