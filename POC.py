from diffusers import StableDiffusionXLPipeline
from transformers import T5Tokenizer, T5EncoderModel
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)
from typing import Optional

import torch.nn as nn, torch, types

T5_NAME  = "mcmonkey/google_t5-v1_1-xxl_encoderonly"
SDXL_DIR = "stabilityai/stable-diffusion-xl-base-1.0"



class T5SDXLPipeline(StableDiffusionXLPipeline):
    def __init__(
        self,
        vae,
        text_encoder,
        text_encoder_2,
        tokenizer,
        tokenizer_2,
        unet,
        scheduler,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2,
            unet, scheduler,
        )
        # ----- build T5 + projection -----
        self.tokenizer      = T5Tokenizer.from_pretrained(T5_NAME)
        self.t5_encoder     = T5EncoderModel.from_pretrained(T5_NAME,
                                torch_dtype=self.unet.dtype)
        self.t5_projection  = nn.Linear(4096, 2048)   # trainable

        # drop CLIP encoders to save VRAM
        self.text_encoder = self.text_encoder_2 = None
        self.tokenizer_2  = None


    # ------------------------------------------------------------------
    #  Encode a text prompt (T5-XXL + 4096→2048 projection)
    #  Returns exactly four tensors in the order SDXL’s __call__ expects.
    # ------------------------------------------------------------------
    def encode_prompt(
        self,
        prompt,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: str | None = None,
        **_,
    ):
        """
        Returns
        -------
        prompt_embeds                : Tensor [B, T, 2048]
        negative_prompt_embeds       : Tensor [B, T, 2048] | None
        pooled_prompt_embeds         : Tensor [B, 1280]
        negative_pooled_prompt_embeds: Tensor [B, 1280]    | None
        where B = batch * num_images_per_prompt
        """

        # --- helper to tokenize on the pipeline’s device ----------------
        def _tok(text: str):
            tok_out = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).to(self.device)
            return tok_out.input_ids, tok_out.attention_mask

        # ---------- positive stream -------------------------------------
        ids, mask = _tok(prompt)
        h_pos = self.t5_encoder(ids, attention_mask=mask).last_hidden_state     # [b, T, 4096]
        tok_pos = self.t5_projection(h_pos)                                     # [b, T, 2048]
        pool_pos = tok_pos.mean(dim=1)[:, :1280]                                # [b, 1280]

        # expand for multiple images per prompt
        tok_pos   = tok_pos.repeat_interleave(num_images_per_prompt, 0)
        pool_pos  = pool_pos.repeat_interleave(num_images_per_prompt, 0)

        # ---------- negative / CFG stream --------------------------------
        if do_classifier_free_guidance:
            neg_text = "" if negative_prompt is None else negative_prompt
            ids_n, mask_n = _tok(neg_text)
            h_neg = self.t5_encoder(ids_n, attention_mask=mask_n).last_hidden_state
            tok_neg = self.t5_projection(h_neg)
            pool_neg = tok_neg.mean(dim=1)[:, :1280]

            tok_neg  = tok_neg.repeat_interleave(num_images_per_prompt, 0)
            pool_neg = pool_neg.repeat_interleave(num_images_per_prompt, 0)
        else:
            tok_neg = pool_neg = None

        # ----------------- final ordered return --------------------------
        # 1) positive token embeddings
        # 2) negative token embeddings (or None)
        # 3) positive pooled embeddings
        # 4) negative pooled embeddings (or None)
        return tok_pos, tok_neg, pool_pos, pool_neg
    @classmethod
    def from_pretrained_t5sdxl(cls, *model_args, **model_kwargs):
        pipe = super().from_pretrained(*model_args, **model_kwargs)
        return cls(**pipe.components)   # rebuild into our subclass


# -- now do something with it --
pipe = T5SDXLPipeline.from_pretrained(
    SDXL_DIR,
    torch_dtype=torch.bfloat16,
).to("cuda")

pipe.t5_encoder.to(pipe.device, dtype=pipe.unet.dtype)
pipe.t5_projection.to(pipe.device, dtype=pipe.unet.dtype)


images = pipe("sad girl in snow").images
images[0].save("test.png")
