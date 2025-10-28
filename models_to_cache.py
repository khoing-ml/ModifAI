

import os
from dotenv import load_dotenv

# 1. Set visible GPUs (must be done BEFORE importing torch)
# This tells CUDA to only make GPUs 1 and 2 available.
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

# 2. Load the .env file
load_dotenv()

# 3. Get the HF_TOKEN from the environment
hf_token = os.getenv("HF_TOKEN")

import torch

print(f"Hugging Face Token Loaded: {hf_token is not None}")
print(f"CUDA_VISIBLE_DEVICES set to: {os.getenv('CUDA_VISIBLE_DEVICES')}")

if torch.cuda.is_available():
    print(f"Torch sees {torch.cuda.device_count()} GPU(s).")
    # This will print the name of your original GPU 1
    print(f"Device cuda:1,2 name: {torch.cuda.get_device_name(1)}")
else:
    print("Torch cannot see any CUDA-capable devices.")
from transformers import SamModel, SamProcessor
from diffusers import StableDiffusion3ControlNetInpaintingPipeline, ControlNetModel, UniPCMultistepScheduler

print("Pre-loading all models...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
HF_HOME = "models-cache"

print("Loading SAM...")
SamModel.from_pretrained("facebook/sam-vit-huge", cache_dir=HF_HOME)
SamProcessor.from_pretrained("facebook/sam-vit-huge", cache_dir=HF_HOME)

print("Loading ControlNet...")
ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16, cache_dir=HF_HOME)

print("Loading Inpainting Pipeline...")
StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    cache_dir=HF_HOME
)

print("All models have been downloaded.")
