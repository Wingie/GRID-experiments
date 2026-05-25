"""Encode the 4-frame screenshot history into visual tokens for the LLM.

Pipeline (project spec §3 vision + §8): each frame -> SigLIP-SO400M features (frozen) ->
the trained mmproj MLP -> a block of ~256 LLM-embedding-space tokens. The four blocks are
concatenated with frame-separator markers to give ~20 s of visual history (gotcha #4).

Live/GPU-only. The :class:`StateEncoder` is constructed lazily so importing this module
(e.g. for typing) does not require torch/timm.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import cfg_get, get_logger

log = get_logger("state_encoder")

FRAME_SEP_TOKEN = "<FRAME_SEP>"
VISION_START_TOKEN = "<VISION_START>"
VISION_END_TOKEN = "<VISION_END>"


def vision_special_tokens() -> list[str]:
    return [VISION_START_TOKEN, VISION_END_TOKEN, FRAME_SEP_TOKEN]


class StateEncoder:
    """Turn a list of frame paths into a ``[num_frames, tokens_per_frame, d_model]`` tensor."""

    def __init__(self, cfg, mmproj_ckpt: str | Path | None = None):
        import torch

        from rimworld_agent.vision.train_mmproj import MMProj, load_siglip

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.siglip, self.preprocess, vis_dim = load_siglip(
            cfg_get(cfg, "vision.siglip_model", "vit_so400m_patch14_siglip_384")
        )
        self.siglip.eval().to(self.device)
        self.mmproj = MMProj(
            vis_dim=vis_dim,
            d_model=cfg_get(cfg, "vision.d_model", 1536),
            tokens_per_frame=cfg_get(cfg, "vision.tokens_per_frame", 256),
            hidden=cfg_get(cfg, "vision.mmproj_hidden", 2048),
        ).to(self.device)
        if mmproj_ckpt and Path(mmproj_ckpt).exists():
            self.mmproj.load_state_dict(torch.load(mmproj_ckpt, map_location=self.device))
        self.mmproj.eval()

    @property
    def num_frames(self) -> int:
        return 4

    def encode(self, frame_paths: list[str | Path], root: str | Path = ".") -> "object":
        """Return visual tokens ``[num_frames, tokens_per_frame, d_model]`` for the frames."""
        from PIL import Image

        torch = self.torch
        imgs = []
        for p in frame_paths[: self.num_frames]:
            full = Path(root) / p
            imgs.append(self.preprocess(Image.open(full).convert("RGB")))
        batch = torch.stack(imgs).to(self.device)
        with torch.no_grad():
            feats = self.siglip(batch)  # [F, vis_dim] or [F, patches, vis_dim]
            tokens = self.mmproj(feats)  # [F, tokens_per_frame, d_model]
        return tokens
