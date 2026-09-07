FROM python:3.11-slim

# opencv-python (even "headless" use here) needs these shared libs at
# import time on a bare Debian slim image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Generous timeout/retries: pip's default 15s per-chunk read timeout is
# too tight for large wheels (torch, opencv) on a slow/constrained
# network - without this, a transient stall fails the whole build instead
# of just retrying.
ENV PIP_DEFAULT_TIMEOUT=120
RUN pip config set global.retries 5

# ultralytics (via requirements.txt below) depends on torch/torchvision
# with no pinned build variant, so a plain `pip install` on Linux resolves
# the default PyPI wheel - which pulls in the full NVIDIA CUDA/cuDNN
# stack (several GB) even though this app only ever runs on CPU
# (MODEL_DEVICE=cpu). Installing the CPU-only build from PyTorch's own
# index FIRST satisfies that dependency before requirements.txt gets to
# it, so pip never reaches for the CUDA variant.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# YOLO weights (yolov8n.pt / yolov8n-pose.pt) are intentionally NOT baked
# in here - ultralytics auto-downloads them on first use, same as the
# non-Docker dev setup (see README's "YOLO model weights" note). This
# keeps the image small and the Dockerfile simple; it does mean the first
# detection request needs network access.

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
