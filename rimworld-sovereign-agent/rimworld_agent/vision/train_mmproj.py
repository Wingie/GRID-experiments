"""Train the mmproj vision projection on RimWorld screenshots (project spec §7b).

Architecture: SigLIP-SO400M (frozen) -> 2-layer MLP -> LLM embedding space. Only the MLP
is trained. Two losses:
  * alignment   — visual features of a screenshot whose state contains entity ``e`` are
                  pulled toward ``e``'s SID-token embedding (so "seeing a solar panel" lands
                  near ``<SID:3-3-2>``);
  * contrastive — screenshots of the same colony state are pulled together, different
                  states pushed apart.

Game screenshots are ~1920x1080 but SigLIP wants 384/768; we resize (gotcha #10) — prefer
SigLIP-SO400M at 384 and accept small UI text, or 768 variant when available.

GPU-only; written-but-unverified in this environment.
"""

from __future__ import annotations

from pathlib import Path

from rimworld_agent.utils import cfg_get, ensure_dir, get_logger, write_json

log = get_logger("train_mmproj")


def load_siglip(model_name: str = "vit_so400m_patch14_siglip_384"):
    """Load a frozen SigLIP image encoder via timm. Returns (model, preprocess, feat_dim)."""
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    preprocess = timm.data.create_transform(**cfg)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, cfg["input_size"][1], cfg["input_size"][2])
        feat = model(dummy)
    feat_dim = feat.shape[-1]
    return model, preprocess, feat_dim


class MMProj:
    """2-layer MLP mapping SigLIP features -> ``tokens_per_frame`` LLM-space tokens.

    Implemented as a thin wrapper so the class is importable without torch present (the
    actual ``nn.Module`` is built on first construction).
    """

    def __new__(cls, vis_dim: int, d_model: int, tokens_per_frame: int = 256, hidden: int = 2048):
        import torch
        from torch import nn

        class _MMProj(nn.Module):
            def __init__(self):
                super().__init__()
                self.tokens_per_frame = tokens_per_frame
                self.d_model = d_model
                self.net = nn.Sequential(
                    nn.Linear(vis_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, tokens_per_frame * d_model),
                )

            def forward(self, feats):  # feats: [B, vis_dim] (pooled) -> [B, T, d_model]
                if feats.dim() == 3:  # [B, patches, vis_dim] -> pool patches
                    feats = feats.mean(dim=1)
                out = self.net(feats)
                return out.view(feats.size(0), self.tokens_per_frame, self.d_model)

        return _MMProj()


def alignment_loss(visual_tokens, sid_embeddings, present_sid_idx):
    """Pull pooled visual tokens toward the mean SID embedding of entities present on screen."""
    import torch
    import torch.nn.functional as F

    pooled = visual_tokens.mean(dim=1)  # [B, d_model]
    targets = torch.stack([sid_embeddings[idx].mean(0) for idx in present_sid_idx])  # [B, d_model]
    return 1.0 - F.cosine_similarity(pooled, targets, dim=-1).mean()


def contrastive_loss(pooled_a, pooled_b, same_state_mask, temperature: float = 0.07):
    import torch
    import torch.nn.functional as F

    a = F.normalize(pooled_a, dim=-1)
    b = F.normalize(pooled_b, dim=-1)
    logits = a @ b.t() / temperature
    target = same_state_mask.float()
    return F.binary_cross_entropy_with_logits(logits, target)


def train_mmproj(cfg) -> dict:
    """Train the mmproj MLP on the collected screenshot/state dataset. GPU-only."""
    import torch
    from torch.utils.data import DataLoader

    from rimworld_agent.vision.dataset import ScreenshotStateDataset  # built at runtime

    device = "cuda" if torch.cuda.is_available() else "cpu"
    siglip, preprocess, vis_dim = load_siglip(cfg_get(cfg, "vision.siglip_model", "vit_so400m_patch14_siglip_384"))
    siglip.to(device).eval()
    mmproj = MMProj(
        vis_dim=vis_dim,
        d_model=cfg_get(cfg, "vision.d_model", 1536),
        tokens_per_frame=cfg_get(cfg, "vision.tokens_per_frame", 256),
        hidden=cfg_get(cfg, "vision.mmproj_hidden", 2048),
    ).to(device)

    sid_embeddings = torch.load(cfg_get(cfg, "vision.sid_embedding_path")).to(device)
    ds = ScreenshotStateDataset(cfg, preprocess, sid_embeddings)
    dl = DataLoader(ds, batch_size=cfg_get(cfg, "vision.batch_size", 16), shuffle=True)
    opt = torch.optim.AdamW(mmproj.parameters(), lr=cfg_get(cfg, "vision.lr", 1e-4))

    epochs = cfg_get(cfg, "vision.epochs", 5)
    align_w = cfg_get(cfg, "vision.align_weight", 1.0)
    contrast_w = cfg_get(cfg, "vision.contrast_weight", 0.5)
    history = []
    for epoch in range(epochs):
        for batch in dl:
            imgs = batch["image"].to(device)
            with torch.no_grad():
                feats = siglip(imgs)
            tokens = mmproj(feats)
            loss = align_w * alignment_loss(tokens, sid_embeddings, batch["present_sid_idx"])
            if "pair_image" in batch:
                with torch.no_grad():
                    feats_b = siglip(batch["pair_image"].to(device))
                tokens_b = mmproj(feats_b)
                loss = loss + contrast_w * contrastive_loss(
                    tokens.mean(1), tokens_b.mean(1), batch["same_state"].to(device)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
        history.append(float(loss.item()))
        log.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, history[-1])

    out = ensure_dir(Path(cfg_get(cfg, "paths.model_out_dir", "data/models"))) / "mmproj.pt"
    torch.save(mmproj.state_dict(), out)
    report = {"epochs": epochs, "final_loss": history[-1] if history else None, "ckpt": str(out)}
    write_json(report, Path(cfg_get(cfg, "paths.results_dir", "results")) / "mmproj_report.json")
    return report


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        train_mmproj(cfg)

    _run()


if __name__ == "__main__":
    main()
