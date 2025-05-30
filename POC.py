#!/bin/env python

from diffusers import DiffusionPipeline
import torch.nn as nn, torch, types

MODEL_DIR = "opendiffusionai/stablediffusionxl_t5"

pipe = DiffusionPipeline.from_pretrained(
    MODEL_DIR, custom_pipeline=MODEL_DIR, use_safetensors=True,
    torch_dtype=torch.bfloat16,
)

print("model initialized. Now moving to CUDA")
pipe.to("cuda")

print("Trying render now...")
images = pipe("a misty Tokyo alley at night",num_inference_steps=30).images
fname="save.png"
print(f"saving to {fname}")
images[0].save(fname)
