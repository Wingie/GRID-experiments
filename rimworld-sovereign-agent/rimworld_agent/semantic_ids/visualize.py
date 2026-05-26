"""Visualise and compare READ vs WRITE semantic-ID clusters.

Projects the READ (taxonomy) and WRITE (workflow) embeddings to 2-D side by side, coloured by
their level-1 code, so you can eyeball whether READ clusters capture *kinds* of entity while
WRITE clusters capture *workflows* (project spec §9f). Requires the ``viz`` extra.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rimworld_agent.utils import cfg_get, ensure_dir, get_logger

log = get_logger("visualize")


def project_2d(embeddings: np.ndarray, method: str = "umap", seed: int = 42) -> np.ndarray:
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
        except ImportError:
            log.warning("umap-learn missing; falling back to t-SNE")
            method = "tsne"
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, len(embeddings) // 4))
        return TSNE(n_components=2, random_state=seed, perplexity=perplexity).fit_transform(embeddings)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(embeddings)


def _scatter(ax, coords, labels, title):
    import matplotlib.pyplot as plt

    uniq = sorted(set(labels), key=str)
    cmap = plt.get_cmap("tab20")
    for i, lab in enumerate(uniq):
        mask = np.array([x == lab for x in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, color=cmap(i % 20), label=str(lab))
    ax.set_title(title)


def plot_read_vs_write(result, out_path: str | Path, method: str = "pca") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    read_labels = [a["read_codes"][0] for a in result.assignments.values()]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    _scatter(axes[0], project_2d(result.read_embeddings, method), read_labels, "READ (taxonomy) — L1 code")

    if result.write_rqvae is not None and result.cooccurrence is not None and result.cooccurrence.def_names:
        from rimworld_agent.semantic_ids.rqvae_write import build_write_embeddings

        write_emb = build_write_embeddings(result.cooccurrence, result.read_embeddings.shape[1])
        write_codes = result.write_rqvae.encode_to_ids(
            __import__("torch").from_numpy(write_emb)
        ).cpu().numpy()
        _scatter(axes[1], project_2d(write_emb, method), list(write_codes[:, 0]), "WRITE (workflow) — L1 code")
    else:
        axes[1].set_title("WRITE — no gameplay yet")

    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote READ vs WRITE cluster plot -> %s", out_path)
    return out_path


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    from rimworld_agent.semantic_ids.assign_ids import run_pipeline

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        result = run_pipeline(cfg)
        out = cfg_get(cfg, "paths.results_dir", "results") + "/sid_clusters_read_vs_write.png"
        plot_read_vs_write(result, out, method=cfg_get(cfg, "semantic_ids.viz_method", "umap"))

    _run()


if __name__ == "__main__":
    main()
