"""Mine entity co-occurrence from recorded gameplay episodes for the WRITE RQ-VAE.

The WRITE codebook captures *workflow* similarity — which entities are used together — so it
needs gameplay data (project spec §7b-2, gotcha #11: it CANNOT be trained before bootstrap
episodes exist). We extract the def_names an agent acted on each step and accumulate:
  * window co-occurrence — entities acted on within the same K-action window (symmetric);
  * sequential pairs     — entity acted on at step t, then another at step t+1 (directed).

Output is a co-occurrence matrix keyed by def_name, consumed by `rqvae_write`. Offline-usable
over real recordings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from rimworld_agent.utils import cfg_get, get_logger

log = get_logger("collect_cooccurrence")

# Action params that name an entity def.
_ENTITY_PARAMS = ("def_name", "project_def", "plant_def")


def entities_in_step(step) -> list[str]:
    """def_names the agent acted on in a step (from action params)."""
    names: list[str] = []
    for action in step.actions:
        for key in _ENTITY_PARAMS:
            val = action.params.get(key)
            if isinstance(val, str) and val:
                names.append(val)
    return names


@dataclass
class CooccurrenceData:
    index: dict[str, int]  # def_name -> row index
    matrix: np.ndarray  # symmetric window co-occurrence counts [N, N]
    sequential: np.ndarray  # directed sequential counts [N, N] (row=from, col=to)
    def_names: list[str] = field(default_factory=list)

    def ppmi(self) -> np.ndarray:
        """Positive pointwise mutual information of the symmetric matrix (workflow affinity)."""
        m = self.matrix.astype(np.float64)
        total = m.sum()
        if total <= 0:
            return m.astype(np.float32)
        row = m.sum(axis=1, keepdims=True)
        col = m.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((m * total) / np.clip(row * col, 1e-12, None))
        pmi[~np.isfinite(pmi)] = 0.0
        return np.maximum(pmi, 0.0).astype(np.float32)


def collect_cooccurrence(episode_dir: str | Path, window: int = 5) -> CooccurrenceData:
    """Scan recorded episodes and build the co-occurrence + sequential matrices."""
    from rimworld_agent.game.episode_recorder import load_episode

    episodes = [load_episode(p) for p in sorted(Path(episode_dir).glob("*/episode.json"))]
    # Build the vocabulary of acted-on entities.
    vocab: dict[str, int] = {}
    per_step: list[list[str]] = []
    for ep in episodes:
        for step in ep.steps:
            names = entities_in_step(step)
            per_step.append(names)
            for n in names:
                vocab.setdefault(n, len(vocab))
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=np.float32)
    sequential = np.zeros((n, n), dtype=np.float32)

    # Re-walk per episode to respect step ordering for sequential pairs.
    cursor = 0
    for ep in episodes:
        steps = ep.steps
        ep_step_names = per_step[cursor : cursor + len(steps)]
        cursor += len(steps)
        for i, names in enumerate(ep_step_names):
            ids = [vocab[x] for x in names]
            # symmetric co-occurrence within this step
            for a in ids:
                for b in ids:
                    if a != b:
                        matrix[a, b] += 1.0
            # window co-occurrence across nearby steps
            for w in range(1, window):
                if i + w < len(ep_step_names):
                    for a in ids:
                        for b in (vocab[x] for x in ep_step_names[i + w]):
                            if a != b:
                                matrix[a, b] += 1.0 / w
                            sequential[a, b] += 1.0 / w
    log.info("co-occurrence: %d acted-on entities over %d episodes", n, len(episodes))
    return CooccurrenceData(index=vocab, matrix=matrix, sequential=sequential, def_names=list(vocab))


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    from rimworld_agent.utils import ensure_dir, write_json

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        data = collect_cooccurrence(
            cfg_get(cfg, "paths.episodes_dir", "data/episodes"),
            window=cfg_get(cfg, "semantic_ids.write.cooccurrence_window", 5),
        )
        out = cfg_get(cfg, "paths.cooccurrence_file", "results/cooccurrence.json")
        ensure_dir(Path(out).parent)
        write_json({"def_names": data.def_names, "num_entities": len(data.def_names)}, out)
        log.info("wrote co-occurrence summary -> %s", out)

    _run()


if __name__ == "__main__":
    main()
