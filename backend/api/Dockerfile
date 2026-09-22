# API service — CPU, small, scale-to-zero.
# Build context: tracker_app/backend
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Copy the shared modules the api imports from one level up:
#   storage.py, pipeline/ (for pipeline.errors.InvalidVideoError)
COPY storage.py /app/storage.py
COPY pipeline /app/pipeline
COPY api /app/api

# Ball-marker PNGs used by pipeline.bake_overlay (download_full endpoint).
# Staged into backend/api/assets so they're inside the build context.
ENV TRACKER_BALL_ASSETS_DIR=/app/api/assets

ENV PYTHONPATH=/app
ENV PORT=8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
