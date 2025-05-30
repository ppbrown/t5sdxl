# t5sdxl
Experiments in giving SDXL a T5 text encoder front end

I have heard a few experiments in adding adaptors, or translators, for T5 text encoder in front of SDXL. But I have yet to hear of completely replacing CLIP with it.

So.. I'm going to experiment with doing that.

# How to use

Grab POC.py
Run with

    pip install torch diffusers transformers accelerate sentencepiece
    python POC.py


# Improvements

Prior versions did a full conversion of the base SDXL model on the fly, before then trying to do something with it.

In contrast, THIS version draws from a mostly converted model, living at
https://huggingface.co/opendiffusionai/stablediffusionxl_t5

But... that is still a raw conversion. 
Now that the rendering code is somewhat clean, 
I plan to try finetuning it to create something actually comprehensible.

Once I can acquire updated training software that recognizes the model, that is.
