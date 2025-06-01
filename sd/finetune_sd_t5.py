#!/usr/bin/env python
# finetune_sd_t5.py
# See "tune.sh" for a nice wrapper for this.
# Rather than a "config file", just create variants of the wrapper script
# to hold different configurations.

"""
Fine-tune a custom T5 + SD 1.5 Diffusers pipeline.

Requirements:
  pip install "diffusers>=0.27" transformers accelerate torchvision safetensors
Example run:
  accelerate launch finetune_t5_sd1p5.py \
      --pretrained_model opendiffusionai/stablediffusion_t5 \
      --train_data_dir   /path/to/dataset \
      --output_dir       ./ft_model \
      --resolution       512 \
      --batch_size       4 \
      --max_steps        10_000
"""

import argparse, os, random, math
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torchvision.transforms as T
import safetensors.torch as st
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from PIL import Image

# --------------------------------------------------------------------------- #
# 1. CLI                                                                      #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model", required=True,  help="HF repo or local dir")
    p.add_argument("--train_data_dir",   required=True,  help="root with *.jpg + *.txt")
    p.add_argument("--output_dir",       required=True)
    p.add_argument("--resolution",  type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--max_steps",   type=int, default=10_000)
    p.add_argument("--save_steps",  type=int, default=1_000)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--cpu_offload", type=bool, default=True)
    p.add_argument('--cached_txt',      action='store_true',
                   help='Load *.txt_t5cache files instead of computing T5 embeddings')
    p.add_argument('--gradient_checkpointing', action='store_true')
    return p.parse_args()

# --------------------------------------------------------------------------- #
# 2. Dataset                                                                  #
# --------------------------------------------------------------------------- #
class CaptionImgDataset(Dataset):
    """Reads *.jpg + matching *.txt caption files anywhere under root."""
    def __init__(self, root: str, size: int, use_cache: bool = False):
        self.files = [p for p in Path(root).rglob("*.jpg")]
        self.size  = size
        self.use_cache = use_cache
        self.img_tfm = T.Compose([
            T.Lambda(lambda im: im.convert("RGB")),
            T.Resize(size, interpolation=Image.BICUBIC),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),        # SD convention
        ])

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        txt_path = img_path.with_suffix(".txt")
        caption  = txt_path.read_text(encoding="utf-8").strip()

        px = self.img_tfm(Image.open(img_path))

        if self.use_cache:
            # Tell caller to use cache file named like image.txt_t5cache
            # if it exists, we shouldnt bother reading txt caption file
            cache_path = img_path.with_suffix(".txt_t5cache")
            return {"pixel_values": px, "caption": caption, "cache_path": str(cache_path)}
        else:
            return {"pixel_values": px, "caption": caption}

def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    captions     = [e["caption"] for e in examples]
    batch = {"pixel_values": pixel_values, "captions": captions}
    if "cache_path" in examples[0]:
        cache_paths = [e["cache_path"] for e in examples]
        batch["cache_paths"] = cache_paths
    return batch


# --------------------------------------------------------------------------- #
# 3. Main                                                                     #
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)]
    )
    device = accelerator.device

    # ----- load pipeline --------------------------------------------------- #
    pipe = DiffusionPipeline.from_pretrained(
        args.pretrained_model,
        custom_pipeline=args.pretrained_model,      # uses your pipeline.py
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision=="bf16" else torch.float32,
    ).to(device)

    if args.cpu_offload == True:
        print("Enabling cpu offload")
        pipe.enable_model_cpu_offload()
    if args.gradient_checkpointing == True:
        print("Enabling gradient checkpointing in UNet")
        pipe.unet.enable_gradient_checkpointing()

    vae, unet, sched = pipe.vae.eval(), pipe.unet, pipe.scheduler

    # Freeze VAE (and T5) so only UNet is optimised; comment-out to train all.
    for p in vae.parameters():           p.requires_grad_(False)
    for p in pipe.text_encoder.parameters():   p.requires_grad_(False)
    for p in pipe.t5_projection.parameters():  p.requires_grad_(False)
    unet.train()

    # ----- data ------------------------------------------------------------ #
    ds = CaptionImgDataset(args.train_data_dir, args.resolution, use_cache=args.cached_txt)

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=4, pin_memory=True, collate_fn=collate_fn)

    # ----- optimiser & scheduler ------------------------------------------ #
    opt  = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    lr_s = get_scheduler("cosine", opt,
                         num_warmup_steps=500, num_training_steps=args.max_steps)

    unet, opt, dl, lr_s = accelerator.prepare(unet, opt, dl, lr_s)

    latent_scaling = 0.18215  # fixed for SD 1.5
    global_step    = 0

    pbar = tqdm(total=args.max_steps, desc="Training", unit="step")

    # ----- training loop --------------------------------------------------- #
    for epoch in range(math.ceil(args.max_steps / len(dl))):
        for batch in dl:
            with accelerator.accumulate(unet):
                # 3.1 images => latents
                pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * latent_scaling

                # 3.2 add noise
                noise = torch.randn_like(latents)
                bsz   = latents.size(0)
                timesteps = torch.randint(
                    0, sched.config.num_train_timesteps,
                    (bsz,), device=device, dtype=torch.long
                )
                noisy_latents = sched.add_noise(latents, noise, timesteps)

                # 3.3 encode text  (T5 frozen, no gradients needed) or load cached
                # See create_t5cache_gpu.py
                if args.cached_txt:
                    embeds = []
                    for cache_file in batch["cache_paths"]:
                        emb = st.load_file(cache_file)["emb"]
                        emb = emb.to(device, dtype=unet.dtype)  # ensure correct device & dtype
                        embeds.append(emb)
                    prompt_emb = torch.stack(embeds, dim=0)
                else:
                    with torch.no_grad():
                        prompt_emb = pipe.encode_prompt(
                            batch["captions"],
                            device=device,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False
                        ).to(device, dtype=unet.dtype)

                # 3.4 UNet forward & loss
                model_pred = unet(noisy_latents, timesteps,
                                  encoder_hidden_states=prompt_emb).sample
                loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                opt.step(); lr_s.step(); opt.zero_grad()

            # ----- logging & ckpt ----------------------------------------- #
            if accelerator.is_main_process:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                #if global_step % 100 == 0:
                #    print(f"step {global_step:>6}  loss {loss.item():.4f}")

            if (
                    accelerator.is_main_process
                    and args.save_steps
                    and global_step
                    and global_step % args.save_steps == 0
            ):
                ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.unwrap_model(unet).save_pretrained(ckpt, safe_serialization=True)
                pipe.save_pretrained(ckpt, safe_serialization=True)

            global_step += 1
            if global_step >= args.max_steps:
                break
        if global_step >= args.max_steps:
            break

    # ----- final save ------------------------------------------------------ #
    if accelerator.is_main_process:
        pipe.save_pretrained(args.output_dir, safe_serialization=True)
        print(f"finished:model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
