"""rimworld-sovereign-agent: a sovereign small-LM agent that learns RimWorld at the
embedding layer (extended tokenizer + semantic IDs + mmproj vision + discrete-diffusion
action planning) and improves through self-play.

The package reuses the `vocab-extend-qlora` research code (importable as ``src``) for the
tokenizer-extension / RQ-VAE / QLoRA machinery, and adds the game-specific layers:
knowledge extraction, vision, game interaction, self-play training, and evaluation.
"""

__version__ = "0.1.0"
