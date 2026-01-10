# Dockerfile for Wan2.2-S2V-14B on RunPod Serverless (CACHED VERSION)
# Uses RunPod model caching - no model bundled in image
# Requires 80GB+ VRAM (H100, A100-80GB)
#
# IMPORTANT: Set Model field in RunPod endpoint config to: Wan-AI/Wan2.2-S2V-14B
# This enables automatic model caching for faster cold starts

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
    pip install --no-cache-dir runpod huggingface_hub[cli] decord

# Install Flash Attention 2 (use pre-built wheel to avoid 2+ hour build time)
# Wheel from: https://github.com/mjun0812/flash-attention-prebuild-wheels
# Must match torch version (2.4) and CUDA version (cu121)
ARG FLASH_ATTN_WHEEL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.4/flash_attn-2.7.3%2Bcu121torch2.4-cp310-cp310-linux_x86_64.whl
RUN pip install --no-cache-dir ${FLASH_ATTN_WHEEL}

# NOTE: Model is NOT bundled - uses RunPod model caching instead
# Set Model field in endpoint config to: Wan-AI/Wan2.2-S2V-14B
# Model will be cached at: /runpod-volume/huggingface-cache/hub/models--Wan-AI--Wan2.2-S2V-14B/

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV MODEL_DIR=/models/Wan2.2-S2V-14B
ENV DEFAULT_SIZE=832*480
ENV DEFAULT_STEPS=30
ENV OFFLOAD_MODEL=True
ENV HF_HOME=/models/hf_cache

# Expose port (optional, for health checks)
EXPOSE 8000

# Start handler
CMD ["python", "-u", "handler.py"]
