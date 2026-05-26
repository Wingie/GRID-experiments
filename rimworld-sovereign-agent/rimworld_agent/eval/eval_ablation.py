"""Dual vs single RQ-VAE ablation (project spec §9f — the key experiment).

Compares four semantic-ID configurations on the same downstream metrics:
  1. ``none``          — no SIDs (baseline);
  2. ``single``        — one shared RQ-VAE (one SID per entity, used in reasoning AND actions);
  3. ``dual``          — separate READ (reasoning) + WRITE (actions) RQ-VAEs (this repo's default);
  4. ``multitask``     — one codebook trained on both objectives (Spotify-style bi-encoder).

Each configuration is trained + evaluated separately (GPU); this module *aggregates* the
per-config metric JSONs into one comparison table. The aggregation is pure and testable; the
training runs are not.

Run each config first (e.g. ``sid_mode=dual``), writing
``results/ablation/<mode>/{eval_gameplay,eval_planning,eval_compression}.json``, then:
    python -m rimworld_agent.eval.eval_ablation
"""

from __future__ import annotations

import json
from pathlib import Path

from rimworld_agent.utils import cfg_get, get_logger, write_json

log = get_logger("eval_ablation")

MODES = ["none", "single", "dual", "multitask"]
_METRIC_FILES = ["eval_gameplay", "eval_planning", "eval_compression"]


def collect_mode(mode_dir: Path) -> dict:
    out: dict = {}
    for name in _METRIC_FILES:
        p = mode_dir / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text())
    return out


def build_table(ablation_dir: str | Path) -> dict:
    """Aggregate per-mode metrics into a single comparison dict."""
    ablation_dir = Path(ablation_dir)
    table: dict[str, dict] = {}
    for mode in MODES:
        mode_dir = ablation_dir / mode
        if not mode_dir.exists():
            continue
        m = collect_mode(mode_dir)
        gameplay = m.get("eval_gameplay", {})
        planning = m.get("eval_planning", {})
        compression = m.get("eval_compression", {})
        table[mode] = {
            "quest_completion": gameplay.get("mean_quest_completion"),
            "mean_reward": gameplay.get("mean_reward"),
            "action_validity": planning.get("mean_action_validity"),
            "rsid_usage": planning.get("rsid_usage_rate"),
            "wsid_usage": planning.get("wsid_usage_rate"),
            "sid_leakage": planning.get("sid_leakage_rate"),
            "token_reduction": compression.get("reduction_fraction"),
        }
    return table


def main() -> None:
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base="1.3", config_path="../../configs", config_name="base.yaml")
    def _run(cfg: DictConfig) -> None:
        ablation_dir = Path(cfg_get(cfg, "paths.results_dir", "results")) / "ablation"
        table = build_table(ablation_dir)
        write_json(table, ablation_dir / "comparison.json")
        for mode, row in table.items():
            log.info("%-9s %s", mode, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    _run()


if __name__ == "__main__":
    main()
