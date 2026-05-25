"""A small, standalone Residual-Quantized VAE (RQ-VAE) for learning hierarchical
discrete semantic IDs from code-entity embeddings.

This follows the TIGER recipe (Rajput et al., 2023): an encoder maps an entity
embedding into a latent, which is quantised by a stack of ``L`` codebooks operating on
successive residuals. The result is an ``L``-tuple of discrete codes
``[c1, c2, c3]`` where ``c1`` captures a coarse module-level cluster, ``c2`` a finer
class-level cluster, and ``c3`` a method-level role. The model is tiny (<5M params) and
trains in minutes on CPU.

Design notes mirroring the working RQ implementation in the host GRID repo
(``src/modules/clustering/residual_quantization.py``):
  - residuals are (optionally) L2-normalised before each quantisation level;
  - the per-item code sequence has shape ``[N, L]``;
Codebooks here are EMA-updated (van den Oord et al.) rather than k-means, with explicit
dead-code re-initialisation to avoid codebook collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.utils import get_logger

log = get_logger("rqvae")


@dataclass
class RQVAEConfig:
    input_dim: int
    levels: int = 3
    codebook_size: int = 64
    latent_dim: int = 256
    hidden_dim: int = 512
    commitment_weight: float = 0.25
    reconstruction_weight: float = 1.0
    ema_decay: float = 0.99
    dead_code_threshold: int = 256
    normalize_residuals: bool = True
    epsilon: float = 1e-5

    @classmethod
    def from_cfg(cls, cfg: dict) -> "RQVAEConfig":
        from src.utils import cfg_get

        return cls(
            input_dim=cfg_get(cfg, "semantic_ids.embedding_dim", 256),
            levels=cfg_get(cfg, "semantic_ids.rqvae.levels", 3),
            codebook_size=cfg_get(cfg, "semantic_ids.rqvae.codebook_size", 64),
            latent_dim=cfg_get(cfg, "semantic_ids.rqvae.latent_dim", 256),
            hidden_dim=cfg_get(cfg, "semantic_ids.rqvae.hidden_dim", 512),
            commitment_weight=cfg_get(cfg, "semantic_ids.rqvae.commitment_weight", 0.25),
            reconstruction_weight=cfg_get(cfg, "semantic_ids.rqvae.reconstruction_weight", 1.0),
            ema_decay=cfg_get(cfg, "semantic_ids.rqvae.ema_decay", 0.99),
            dead_code_threshold=cfg_get(cfg, "semantic_ids.rqvae.dead_code_threshold", 256),
            normalize_residuals=cfg_get(cfg, "semantic_ids.rqvae.normalize_residuals", True),
        )


class EMAVectorQuantizer(nn.Module):
    """A single VQ layer with EMA codebook updates and dead-code re-initialisation.

    The codebook is a buffer (not an ``nn.Parameter``): it is updated by exponential
    moving averages of assigned inputs during training, not by gradient descent. Only
    the commitment loss carries gradient back to the encoder.
    """

    def __init__(self, num_codes: int, dim: int, decay: float, dead_threshold: int, eps: float):
        super().__init__()
        self.num_codes = num_codes
        self.dim = dim
        self.decay = decay
        self.dead_threshold = dead_threshold
        self.eps = eps

        codebook = torch.randn(num_codes, dim)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(num_codes))
        self.register_buffer("ema_w", codebook.clone())
        self.register_buffer("steps_since_used", torch.zeros(num_codes, dtype=torch.long))
        self._initialized = False

    @torch.no_grad()
    def _init_codebook(self, x: torch.Tensor) -> None:
        """Seed the codebook from the first real batch (avoids dead start)."""
        n = x.size(0)
        if n >= self.num_codes:
            idx = torch.randperm(n, device=x.device)[: self.num_codes]
            self.codebook.copy_(x[idx])
        else:
            reps = (self.num_codes + n - 1) // n
            self.codebook.copy_(x.repeat(reps, 1)[: self.num_codes])
        self.ema_w.copy_(self.codebook)
        self.cluster_size.fill_(1.0)
        self._initialized = True

    def _assign(self, x: torch.Tensor) -> torch.Tensor:
        # squared euclidean distance to each code -> argmin
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )
        return dist.argmin(dim=1)

    @torch.no_grad()
    def _ema_update(self, x: torch.Tensor, idx: torch.Tensor) -> None:
        onehot = F.one_hot(idx, self.num_codes).type_as(x)  # [B, K]
        counts = onehot.sum(0)  # [K]
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        dw = onehot.t() @ x  # [K, dim]
        self.ema_w.mul_(self.decay).add_(dw, alpha=1 - self.decay)
        # Laplace-smoothed normalisation
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.eps) / (n + self.num_codes * self.eps) * n
        self.codebook.copy_(self.ema_w / cluster_size.unsqueeze(1))

        # dead-code tracking: reset counter for used codes, increment others
        used = counts > 0
        self.steps_since_used[used] = 0
        self.steps_since_used[~used] += 1
        self._reinit_dead(x)

    @torch.no_grad()
    def _reinit_dead(self, x: torch.Tensor) -> None:
        dead = self.steps_since_used >= self.dead_threshold
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return
        idx = torch.randint(0, x.size(0), (n_dead,), device=x.device)
        self.codebook[dead] = x[idx]
        self.ema_w[dead] = x[idx]
        self.cluster_size[dead] = 1.0
        self.steps_since_used[dead] = 0
        log.debug("Re-initialised %d dead codes", n_dead)

    def forward(self, x: torch.Tensor, update: bool = True):
        """Return ``(quantized_st, indices, commitment_loss)``.

        ``quantized_st`` uses the straight-through estimator so gradients reach the
        encoder; ``indices`` are the discrete codes; commitment loss pulls the encoder
        output toward the chosen code.
        """
        if self.training and update and not self._initialized:
            self._init_codebook(x)

        idx = self._assign(x)
        quantized = self.codebook[idx]
        commitment = F.mse_loss(x, quantized.detach())
        quantized_st = x + (quantized - x).detach()

        if self.training and update:
            self._ema_update(x, idx)
        return quantized_st, idx, commitment


class ResidualVQVAE(nn.Module):
    def __init__(self, config: RQVAEConfig):
        super().__init__()
        self.config = config
        c = config
        self.encoder = nn.Sequential(
            nn.Linear(c.input_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Linear(c.hidden_dim, c.latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(c.latent_dim, c.hidden_dim),
            nn.ReLU(),
            nn.Linear(c.hidden_dim, c.input_dim),
        )
        self.quantizers = nn.ModuleList(
            EMAVectorQuantizer(c.codebook_size, c.latent_dim, c.ema_decay, c.dead_code_threshold, c.epsilon)
            for _ in range(c.levels)
        )

    def quantize_latent(self, z: torch.Tensor, update: bool = True):
        """Residual quantisation loop. Returns (z_q, indices[B,L], commitment_loss)."""
        residual = z
        z_q = torch.zeros_like(z)
        indices = []
        commitment = z.new_zeros(())
        for q in self.quantizers:
            r = F.normalize(residual, dim=-1) if self.config.normalize_residuals else residual
            quantized, idx, commit = q(r, update=update)
            # de-normalise contribution back into residual space is approximate; we
            # accumulate in the (normalised) code space and reconstruct from the sum.
            z_q = z_q + quantized
            residual = residual - quantized
            indices.append(idx)
            commitment = commitment + commit
        return z_q, torch.stack(indices, dim=1), commitment / len(self.quantizers)

    def forward(self, x: torch.Tensor, update: bool = True):
        z = self.encoder(x)
        z_q, indices, commitment = self.quantize_latent(z, update=update)
        recon = self.decoder(z_q)
        recon_loss = F.mse_loss(recon, x)
        loss = (
            self.config.reconstruction_weight * recon_loss
            + self.config.commitment_weight * commitment
        )
        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "commitment_loss": commitment,
            "indices": indices,
            "recon": recon,
        }

    @torch.no_grad()
    def encode_to_ids(self, x: torch.Tensor) -> torch.Tensor:
        """Map embeddings ``[N, input_dim]`` to discrete code sequences ``[N, L]``."""
        self.eval()
        z = self.encoder(x)
        _, indices, _ = self.quantize_latent(z, update=False)
        return indices

    @torch.no_grad()
    def codebook_vectors(self) -> torch.Tensor:
        """Stacked code embeddings ``[L, K, latent_dim]`` (for tokenizer init)."""
        return torch.stack([q.codebook.detach() for q in self.quantizers], dim=0)

    def codebook_usage(self) -> list[float]:
        """Fraction of codes used per level (collapse diagnostic)."""
        return [
            float((q.cluster_size > q.eps).float().mean().item()) for q in self.quantizers
        ]


def train_rqvae(embeddings: np.ndarray, cfg: dict) -> ResidualVQVAE:
    """Train the RQ-VAE on entity embeddings and return the fitted model."""
    from src.utils import cfg_get, set_seed

    set_seed(cfg_get(cfg, "seed", 42))
    config = RQVAEConfig.from_cfg(cfg)
    config.input_dim = embeddings.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualVQVAE(config).to(device)
    model.train()

    x = torch.from_numpy(np.asarray(embeddings, dtype=np.float32)).to(device)
    n = x.size(0)
    steps = cfg_get(cfg, "semantic_ids.rqvae.train_steps", 2000)
    bs = min(cfg_get(cfg, "semantic_ids.rqvae.batch_size", 256), n)
    lr = cfg_get(cfg, "semantic_ids.rqvae.lr", 1e-3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    layer_wise = cfg_get(cfg, "semantic_ids.rqvae.train_layer_wise", False)
    for step in range(steps):
        idx = torch.randint(0, n, (bs,), device=device)
        batch = x[idx]
        if layer_wise:
            active = min(step // max(1, steps // config.levels) + 1, config.levels)
            for li, q in enumerate(model.quantizers):
                q._update_gate = li < active  # informational; forward always updates
        out = model(batch, update=True)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            log.info(
                "step %d/%d loss=%.4f recon=%.4f commit=%.4f usage=%s",
                step,
                steps,
                out["loss"].item(),
                out["recon_loss"].item(),
                out["commitment_loss"].item(),
                [round(u, 2) for u in model.codebook_usage()],
            )
    return model


def save_rqvae(model: ResidualVQVAE, path: str) -> None:
    from src.utils import ensure_dir

    ensure_dir(__import__("pathlib").Path(path).parent)
    torch.save({"state_dict": model.state_dict(), "config": model.config.__dict__}, path)


def load_rqvae(path: str, map_location: str = "cpu") -> ResidualVQVAE:
    ckpt = torch.load(path, map_location=map_location)
    model = ResidualVQVAE(RQVAEConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
