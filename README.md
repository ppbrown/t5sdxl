# t5sdxl
Experiments in giving SDXL a T5 text encoder front end

I have heard a few experiments in adding adaptors, or translators, for T5 text encoder in front of SDXL. But I have yet to hear of completely replacing CLIP with it.

So.. I'm going to experiment with doing that.

Right now there is only POC.py
Run with

    pip install torch diffusers transformers accelerate sentencepiece
    python POC.py

This will dynamically assemble a varient of SDXL, that replaces the text input CLIPs, with T5.
And then run a silly prompt. Which generates effectively garbage output.

Eventually I plan to try finetuning it to create something actually comprehensible.

