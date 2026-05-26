"""Project entity embeddings to 2-D and colour them by their level-1 semantic-ID code (or
by entity-graph category) to inspect whether the SID hierarchy is meaningful.

Requires the ``viz`` extra (matplotlib + scikit-learn / umap-learn). Writes a PNG to
``results/``.
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


def plot_clusters(
    embeddings: np.ndarray,
    labels: list,
    out_path: str | Path,
    title: str = "Semantic-ID clusters",
    method: str = "umap",
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    coords = project_2d(embeddings, method)
    uniq = sorted(set(labels), key=str)
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, lab in enumerate(uniq):
        mask = np.array([l == lab for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1], s=10, color=cmap(i % 20), label=str(lab))
    ax.set_title(title)
    if len(uniq) <= 25:
        ax.legend(markerscale=2, fontsize=7, loc="best", ncol=2)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    log.info("wrote cluster plot -> %s", out_path)
    return out_path


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    from rimworld_agent.semantic_ids.assign_ids import run_pipeline

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        result = run_pipeline(cfg)
        color_by = cfg_get(cfg, "semantic_ids.viz_color_by", "level1")
        if color_by == "category":
            labels = [result.graph.category_of(e.def_name) for e in result.entities]
        else:
            labels = [a["codes"][0] for a in result.assignments.values()]
        out = cfg_get(cfg, "paths.results_dir", "results") + "/sid_clusters.png"
        plot_clusters(result.embeddings, labels, out, method=cfg_get(cfg, "semantic_ids.viz_method", "umap"))

    _run()


if __name__ == "__main__":
    main()
