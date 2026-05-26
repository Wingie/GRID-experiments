"""Dataset of (screenshot, paired game state) records for mmproj training.

Each example yields the preprocessed image plus the indices into the SID-embedding table of
the entities present in the paired state (the alignment targets). When ``vision.contrastive``
is enabled, a second image from the same / a different state is added with a ``same_state``
flag for the contrastive loss. Built at training time; GPU-adjacent.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rimworld_agent.utils import cfg_get, get_logger

log = get_logger("vision_dataset")


def _build_torch_dataset():
    from torch.utils.data import Dataset

    class _ScreenshotStateDataset(Dataset):
        def __init__(self, cfg, preprocess, sid_embeddings):
            import torch

            self.torch = torch
            self.preprocess = preprocess
            self.root = Path(cfg_get(cfg, "paths.screenshots_dir", "data/screenshots"))
            self.contrastive = cfg_get(cfg, "vision.contrastive", True)
            self.records = sorted(self.root.glob("*.json"))
            # def_name -> RSID-table row index, supplied alongside the RSID embeddings.
            self.sid_index = json.loads(Path(cfg_get(cfg, "vision.rsid_index_path")).read_text())
            if not self.records:
                log.warning("no screenshot/state pairs under %s", self.root)

        def __len__(self) -> int:
            return len(self.records)

        def _present_idx(self, state: dict) -> list[int]:
            names = {c.get("def", "") for c in state.get("visible", [])} or set(state.get("visible_defs", []))
            idx = [self.sid_index[n] for n in names if n in self.sid_index]
            return idx or [0]

        def __getitem__(self, i: int):
            from PIL import Image

            rec = self.records[i]
            state = json.loads(rec.read_text())
            img = self.preprocess(Image.open(rec.with_suffix(".png")).convert("RGB"))
            item = {"image": img, "present_sid_idx": self.torch.tensor(self._present_idx(state))}
            if self.contrastive and len(self.records) > 1:
                same = random.random() < 0.5
                j = i if same else random.randrange(len(self.records))
                rec_b = self.records[j]
                item["pair_image"] = self.preprocess(Image.open(rec_b.with_suffix(".png")).convert("RGB"))
                item["same_state"] = self.torch.tensor(1.0 if same else 0.0)
            return item

    return _ScreenshotStateDataset


def ScreenshotStateDataset(cfg, preprocess, sid_embeddings):  # noqa: N802 (factory, returns Dataset)
    return _build_torch_dataset()(cfg, preprocess, sid_embeddings)
