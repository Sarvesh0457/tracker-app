# Worker service — CUDA + GPU, scale-to-zero on L4.
# Build context: tracker_app/backend
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3-pip python3.11-venv \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY worker/requirements.txt /app/worker/requirements.txt
RUN pip install --no-cache-dir -r /app/worker/requirements.txt

# Shared backend modules.
COPY storage.py /app/storage.py
COPY pipeline /app/pipeline
COPY api /app/api
COPY worker /app/worker

# Bake the ONNX models into the image. Build context must contain them;
# see deploy/RUNBOOK.md step "stage models for worker build".
COPY worker/models /app/models

# Legacy tracker folders (white + red ball pipelines) — staged at build time.
COPY worker/legacy /app/legacy

ENV PYTHONPATH=/app
ENV MODELS_DIR=/app/models
ENV LEGACY_ROOT=/app/legacy
ENV PORT=8080

CMD ["uvicorn", "worker.main:app", "--host", "0.0.0.0", "--port", "8080"]
