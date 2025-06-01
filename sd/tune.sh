
# Note: this uses the SD script, not SDXL script.

# one-time setup
#pip install "diffusers>=0.27" transformers accelerate torchvision safetensors
#

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# single-GPU (or Multi-GPU via accelerate config)
# Some stats for SD1.5 training, WITHOUT gradient checkpointing
#   With or without cached_txt, can only fit batchsize=14
#   TIME changes drastically however.
#   1.8 vs 8.2 sec/step
# With gradient checkpointing, however..
#   with cached_txt, it is still around 1.8 sec/step
#
# Batchsize 64 wont quite fit. But b=50 does
accelerate launch finetune_sd_t5.py \
  --pretrained_model  opendiffusionai/stablediffusion_t5 \
  --cached_txt \
  --train_data_dir    train-data \
  --output_dir        ./t5_sd1p5_finetuned \
  --resolution        512 \
  --batch_size        50 \
  --gradient_checkpointing  \
  --max_steps         20_000

# for 200k dataset, batch=14, approximately 15000 steps per epoch
#                   batch=32, approximately 6200 steps per epoch
#                   batch=64, approximately 3100 steps per epoch, but wont fit
