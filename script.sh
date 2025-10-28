#! /usr/bin/bash

sudo docker run \
    --gpus all \
    -p 7860:7860 \
    -v $(pwd)/outputs:/app/outputs \
    -v $HOME/.cache/huggingface:/app/models-cache \
    --name modif-ai-app-slim \
    modif-ai-slim