"""Quality metrics specific to the semantic-ID contribution.

CPU-computable from embeddings + assignments:
  - cluster_coherence        : intra- vs inter-cluster cosine similarity per level.
  - hierarchical_consistency : do methods of the same class share L1 / L2 prefixes?
  - novel_entity_generalization : hold out entities, refit RQ-VAE, check that held-out
                                  entities land near same-prefix neighbours above chance.

Model-dependent (GPU; computed when a model_path + CUDA are available):
  - sid_prediction_accuracy  : given a body, predict its SID (top-1 per level).
  - sid_to_signature         : given a SID, generate a plausible signature (BLEU/edit).

Usage:
    python -m src.eval_semantic_ids configs/experiments/semid_qlora.yaml \
        [model_path=data/models/semid_qlora]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from src.semantic_ids.assign_ids import run_pipeline
from src.utils import cfg_get, get_logger, load_config, write_json

log = get_logger("eval_semantic_ids")


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-8, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-8, None)
    return a @ b.T


def cluster_coherence(embeddings: np.ndarray, code_lists: list[list[int]], levels: int) -> dict:
    """Mean intra-cluster vs inter-cluster cosine similarity at each level."""
    sim = _cosine(embeddings, embeddings)
    n = sim.shape[0]
    tri = np.triu_indices(n, k=1)
    out = {}
    for level in range(levels):
        labels = np.array([c[level] for c in code_lists])
        same = labels[tri[0]] == labels[tri[1]]
        intra = sim[tri][same]
        inter = sim[tri][~same]
        out[f"level_{level + 1}"] = {
            "intra_mean": round(float(intra.mean()) if intra.size else 0.0, 4),
            "inter_mean": round(float(inter.mean()) if inter.size else 0.0, 4),
            "separation": round(float((intra.mean() if intra.size else 0) - (inter.mean() if inter.size else 0)), 4),
        }
    return out


def hierarchical_consistency(assignments: dict[str, dict]) -> dict:
    """For each class with >=2 methods, fraction of methods sharing the modal L1 and
    (L1,L2) prefix. Averaged across classes.
    """
    by_class: dict[str, list[list[int]]] = defaultdict(list)
    for a in assignments.values():
        if a.get("parent"):
            by_class[f"{a['file_path']}::{a['parent']}"].append(a["codes"])
    l1_rates, l2_rates = [], []
    for codes in by_class.values():
        if len(codes) < 2:
            continue
        l1 = Counter(c[0] for c in codes)
        l1_rates.append(l1.most_common(1)[0][1] / len(codes))
        if len(codes[0]) >= 2:
            l2 = Counter((c[0], c[1]) for c in codes)
            l2_rates.append(l2.most_common(1)[0][1] / len(codes))
    return {
        "num_multi_method_classes": len(l1_rates),
        "l1_prefix_consistency": round(float(np.mean(l1_rates)) if l1_rates else 0.0, 4),
        "l2_prefix_consistency": round(float(np.mean(l2_rates)) if l2_rates else 0.0, 4),
    }


def novel_entity_generalization(entities, embeddings: np.ndarray, cfg: dict) -> dict:
    """Hold out 10% of entities, refit the RQ-VAE on the rest, assign IDs to the
    held-out set, and check that each held-out entity's nearest training neighbour
    shares its L1 code more often than chance (1/K).
    """
    import torch

    from src.semantic_ids.rqvae import train_rqvae

    rng = np.random.default_rng(cfg_get(cfg, "seed", 42))
    n = len(entities)
    perm = rng.permutation(n)
    k = max(1, int(0.1 * n))
    held_idx, train_idx = perm[:k], perm[k:]
    if len(train_idx) < cfg_get(cfg, "semantic_ids.rqvae.codebook_size", 64):
        return {"skipped": "too few entities for a held-out refit"}

    model = train_rqvae(embeddings[train_idx], cfg)
    # Encode directly (order-preserving) rather than via the uid-keyed assignment dict,
    # which de-duplicates entities that share a uid.
    train_c1 = model.encode_to_ids(torch.from_numpy(embeddings[train_idx]))[:, 0].cpu().numpy()
    held_c1 = model.encode_to_ids(torch.from_numpy(embeddings[held_idx]))[:, 0].cpu().numpy()
    sim = _cosine(embeddings[held_idx], embeddings[train_idx])
    nn = sim.argmax(axis=1)
    match = float((held_c1 == train_c1[nn]).mean())
    chance = 1.0 / cfg_get(cfg, "semantic_ids.rqvae.codebook_size", 64)
    return {
        "num_held_out": int(k),
        "nn_l1_match_rate": round(match, 4),
        "chance_rate": round(chance, 4),
        "lift_over_chance": round(match / chance, 2) if chance else 0.0,
    }


# --------------------------------------------------------------------------- #
# Model-dependent metrics
# --------------------------------------------------------------------------- #
def _model_metrics(cfg: dict, entities, assignments, vocab) -> dict | None:
    model_path = cfg_get(cfg, "model_path")
    if not model_path:
        return None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    if not __import__("torch").cuda.is_available():
        log.warning("No CUDA; skipping model-dependent SID metrics.")
        return None

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto").eval()

    sample = entities[: cfg_get(cfg, "eval.num_completion_samples", 200)]
    level_correct = [0] * vocab.levels
    sig_bleu = []
    from src.eval_completion import _bleu

    for e in sample:
        a = assignments[e.uid]
        # entity -> SID
        prompt = f"<PREDICT_SID> {e.body}\n"
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=2 * vocab.levels + 2, do_sample=False)
        gen = tok.decode(out[0][ids.input_ids.shape[1] :])
        pred_codes = [lc[1] for lc in (vocab.token_to_level_code(t) for t in _split_tokens(gen)) if lc]
        for li in range(vocab.levels):
            if li < len(pred_codes) and pred_codes[li] == a["codes"][li]:
                level_correct[li] += 1
        # SID -> signature
        sprompt = f"<SID_TO_SIG>{a['sid_inline']} "
        sids = tok(sprompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            sout = model.generate(**sids, max_new_tokens=64, do_sample=False)
        sgen = tok.decode(sout[0][sids.input_ids.shape[1] :], skip_special_tokens=True)
        sig_bleu.append(_bleu(sgen, e.signature))

    n = max(1, len(sample))
    return {
        "sid_prediction_top1_per_level": [round(c / n, 4) for c in level_correct],
        "sid_to_signature_bleu": round(sum(sig_bleu) / n, 4),
        "num_samples": len(sample),
    }


def _split_tokens(text: str) -> list[str]:
    """Pull ``<...>`` special tokens out of a decoded string in order."""
    import re

    return re.findall(r"<[^>]+>", text)


def evaluate(cfg: dict) -> dict:
    entities, embeddings, model, vocab, assignments = run_pipeline(cfg)
    code_lists = [assignments[e.uid]["codes"] for e in entities]
    result = {
        "config_name": cfg.get("name"),
        "levels": vocab.levels,
        "codebook_size": vocab.codebook_size,
        "codebook_usage": model.codebook_usage(),
        "cluster_coherence": cluster_coherence(embeddings, code_lists, vocab.levels),
        "hierarchical_consistency": hierarchical_consistency(assignments),
        "novel_entity_generalization": novel_entity_generalization(entities, embeddings, cfg),
    }
    model_metrics = _model_metrics(cfg, entities, assignments, vocab)
    if model_metrics:
        result["model_metrics"] = model_metrics
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=str)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    summary = evaluate(cfg)
    out = Path(cfg_get(cfg, "paths.results_dir", "results")) / f"semantic_ids_{cfg.get('name','run')}.json"
    write_json(summary, out)
    log.info("Semantic-ID eval -> %s", out)


if __name__ == "__main__":
    main()
