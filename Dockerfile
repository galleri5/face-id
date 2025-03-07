FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin:$PATH"

# Install dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Create Python virtual environment
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install required packages
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir fastapi uvicorn websocket-client requests pydantic python-multipart

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git
WORKDIR /app/ComfyUI

# Install ComfyUI requirements
RUN pip install --no-cache-dir -r requirements.txt

# Install InstantID custom node
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git /app/ComfyUI/custom_nodes/ComfyUI_InstantID
WORKDIR /app/ComfyUI/custom_nodes/ComfyUI_InstantID

# Install InstantID requirements
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir insightface==0.7.3 onnxruntime==1.15.1 opencv-python

# Create required directories
RUN mkdir -p /app/ComfyUI/models/instantid
RUN mkdir -p /app/ComfyUI/models/checkpoints

# Download models
RUN wget -O /app/ComfyUI/models/instantid/diffusion_pytorch_model.safetensors https://huggingface.co/InstantX/InstantID/resolve/main/ControlNetModel/diffusion_pytorch_model.safetensors
RUN wget -O /app/ComfyUI/models/instantid/ip-instant.bin https://huggingface.co/InstantX/InstantID/resolve/main/ip-adapter.bin?download=true
RUN wget -O /app/ComfyUI/models/checkpoints/juggLight.safetensors https://huggingface.co/RunDiffusion/Juggernaut-XL-Lightning/resolve/main/Juggernaut_RunDiffusionPhoto2_Lightning_4Steps.safetensors?download=true

# Create API app directory and copy source code
WORKDIR /app
RUN mkdir -p /app/api
COPY main.py /app/api/main.py

# Create startup script to run both services
RUN echo '#!/bin/bash\n\
cd /app/ComfyUI\n\
python main.py --listen 0.0.0.0 --port 6000 > /app/comfyui.log 2>&1 &\n\
cd /app/api\n\
uvicorn main:app --host 0.0.0.0 --port 8000\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

# Expose ports
EXPOSE 6000 8000

# Set working directory for the container
WORKDIR /app

# Start both services
CMD ["/app/start.sh"]