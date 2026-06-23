# Dockerfile for Wan2.2-S2V-14B on RunPod Serverless (NETWORK VOLUME VERSION)
# Model is NOT bundled in the image. It lives on a RunPod network volume mounted
# at /runpod-volume. The handler lazily downloads the model to MODEL_DIR on the
# first cold job (~40GB), which persists on the volume -> all later cold starts
# find it there (no re-download, small image that pushes reliably).
# Requires 80GB+ VRAM (H100, A100-80GB) + a network volume attached to the endpoint.

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone official Wan2.2 repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git . && \
    git checkout main

# Install Python dependencies
# Pin torch to 2.4.x + cu121 to match available flash-attn prebuilt wheel
# (diffusers>=0.31 needs torch>=2.4 for torch.xpu support)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    grep -v -E "^(torch|torchvision|torchaudio|flash_attn)" requirements.txt > requirements_fixed.txt && \
    pip install --no-cache-dir -r requirements_fixed.txt && \
    pip install --no-cache-dir runpod huggingface_hub[cli] decord einops librosa safetensors peft

# Install Flash Attention 2 (use pre-built wheel to avoid 2+ hour build time)
# Wheel from: https://github.com/mjun0812/flash-attention-prebuild-wheels
# Must match torch version (2.4) and CUDA version (cu121)
ARG FLASH_ATTN_WHEEL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.4/flash_attn-2.7.3%2Bcu121torch2.4-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir ${FLASH_ATTN_WHEEL}

# NO model bake here: the model is downloaded lazily by handler.py to the network
# volume (MODEL_DIR) on the first job, and persists across cold starts. This keeps
# the image small (~12GB deps only) so the registry push is reliable.

# =============================================================================
# VERIFY: All imports work (catch missing dependencies at build time)
# =============================================================================
# Note: wan module calls torch.cuda.current_device() at import time (upstream bug)
# so we can only verify non-CUDA imports during build
RUN python -c "\
from decord import VideoReader; \
from einops import rearrange; \
import librosa; \
from safetensors import safe_open; \
import runpod; \
print('Dependencies OK')"

# Note: Cannot verify generate.py or wan imports during build (requires CUDA)
# These will be verified at runtime when GPU is available

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables — MODEL_DIR points at the network volume (serverless
# mounts the volume at /runpod-volume). hf cache also on the volume.
ENV MODEL_DIR=/runpod-volume/Wan2.2-S2V-14B
ENV DEFAULT_SIZE=832*480
ENV DEFAULT_STEPS=30
ENV OFFLOAD_MODEL=False
ENV HF_HOME=/runpod-volume/hf_cache

# Expose port (optional, for health checks)
EXPOSE 8000

# Start handler
CMD ["python", "-u", "handler.py"]
