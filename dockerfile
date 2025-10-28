FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Set CUDA related environment variables
ENV CUDA_VERSION=12.8.0
ENV NVIDIA_VISIBLE_DEVICES=1,2
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# environment
WORKDIR /app
ENV PYTHONUNBUFFED=1
ENV DEBIAN_FRONTEND=noninterative
ENV HF_HOME=/app/models-cache

# tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    cuda-toolkit-12-8 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of application code
COPY . .

# expose the port (Gradio default is 7860)
EXPOSE 7860

# command to run the app
CMD ["gradio", "app.py", "--share", "False", "--server_name", "0.0.0.0"]
