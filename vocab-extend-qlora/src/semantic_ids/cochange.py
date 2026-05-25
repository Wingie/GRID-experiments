"""Optional git co-change signal for the semantic-ID layer.

Functions that change together in the same commit are usually semantically related,
even when their names/text are dissimilar. We mine a file-level co-change graph from the
repository's git history and use it to *smooth* entity embeddings toward their
co-changing neighbours before quantisation, so the RQ-VAE is more likely to place
co-evolving entities in the same cluster.

This needs real history: a ``--depth 1`` shallow clone yields a single commit and the
graph is empty (the smoothing is then a no-op). Enable a full clone via
``scripts/clone_target.sh <url> full`` and set ``semantic_ids.cochange.enabled=true``.

The git-log *parsing* and the smoothing *math* are factored into pure functions so they
can be unit-tested without a live repository.
"""

from __future__ import annotations

import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.semantic_ids.extract_entities import CodeEntity
from src.utils import cfg_get, get_logger

log = get_logger("cochange")

_COMMIT_MARKER = "__COMMIT__"


def parse_git_log(text: str) -> list[set[str]]:
    """Parse ``git log --pretty=format:__COMMIT__ --name-only`` output into a list of
    per-commit changed-file sets.
    """
    commits: list[set[str]] = []
    current: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == _COMMIT_MARKER:
            if current:
                commits.append(current)
            current = set()
        elif stripped:
            current.add(stripped)
    if current:
        commits.append(current)
    return commits


def get_commits_from_repo(repo_root: str, max_commits: int) -> list[set[str]]:
    """Return per-commit changed-file sets (absolute paths), or ``[]`` if git is
    unavailable / the clone is shallow.
    """
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        log_text = subprocess.check_output(
            [
                "git", "-C", repo_root, "log", f"-n{max_commits}", "--no-merges",
                "--pretty=format:" + _COMMIT_MARKER, "--name-only",
            ],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8", "replace")
    except Exception as exc:  # not a repo, git missing, etc.
        log.warning("co-change: could not read git history (%s)", exc)
        return []
    commits = parse_git_log(log_text)
    return [{str(Path(toplevel) / f) for f in files} for files in commits]


def build_cochange_graph(commits: list[set[str]], max_files_per_commit: int = 50) -> dict[str, dict[str, float]]:
    """Build a row-normalised file-level co-change affinity graph.

    Commits touching more than ``max_files_per_commit`` files (sweeping refactors,
    formatting passes) are skipped as noise.
    """
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for files in commits:
        if len(files) < 2 or len(files) > max_files_per_commit:
            continue
        fl = sorted(files)
        for i in range(len(fl)):
            for j in range(i + 1, len(fl)):
                counts[fl[i]][fl[j]] += 1.0
                counts[fl[j]][fl[i]] += 1.0
    graph: dict[str, dict[str, float]] = {}
    for f, nbrs in counts.items():
        total = sum(nbrs.values())
        graph[f] = {g: w / total for g, w in nbrs.items()} if total else {}
    return graph


def cochange_smoothing(
    entities: list[CodeEntity],
    embeddings: np.ndarray,
    graph: dict[str, dict[str, float]],
    alpha: float,
    repo_root: str,
) -> np.ndarray:
    """Blend each entity embedding with the mean embedding of entities in co-changing
    files: ``e' = normalize((1 - alpha) * e + alpha * neighbour_mean)``.
    """
    abspaths = [str(Path(repo_root) / e.file_path) for e in entities]
    file_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, ap in enumerate(abspaths):
        file_to_idx[ap].append(i)
    file_mean = {f: embeddings[idxs].mean(axis=0) for f, idxs in file_to_idx.items()}

    out = embeddings.copy()
    for i, ap in enumerate(abspaths):
        nbrs = graph.get(ap)
        if not nbrs:
            continue
        acc = np.zeros(embeddings.shape[1], dtype=np.float32)
        wsum = 0.0
        for g, w in nbrs.items():
            if g in file_mean:
                acc += w * file_mean[g]
                wsum += w
        if wsum > 0:
            v = (1 - alpha) * embeddings[i] + alpha * (acc / wsum)
            norm = np.linalg.norm(v)
            if norm > 0:
                out[i] = v / norm
    return out


def maybe_apply_cochange(entities: list[CodeEntity], embeddings: np.ndarray, cfg: dict):
    """Apply co-change smoothing when enabled and history is available. Returns
    ``(embeddings, info)`` and never raises — degrades to a no-op.
    """
    if not cfg_get(cfg, "semantic_ids.cochange.enabled", False):
        return embeddings, {"enabled": False}
    repo = cfg_get(cfg, "paths.target_repo")
    commits = get_commits_from_repo(repo, cfg_get(cfg, "semantic_ids.cochange.max_commits", 2000))
    if len(commits) < 2:
        log.warning("co-change: <2 commits of history (shallow clone?); skipping smoothing.")
        return embeddings, {"enabled": True, "applied": False, "num_commits": len(commits)}
    graph = build_cochange_graph(commits, cfg_get(cfg, "semantic_ids.cochange.max_files_per_commit", 50))
    alpha = cfg_get(cfg, "semantic_ids.cochange.alpha", 0.3)
    smoothed = cochange_smoothing(entities, embeddings, graph, alpha, repo)
    info = {
        "enabled": True,
        "applied": True,
        "num_commits": len(commits),
        "num_files_in_graph": len(graph),
        "alpha": alpha,
    }
    log.info("co-change smoothing applied (commits=%d, files=%d, alpha=%.2f)", len(commits), len(graph), alpha)
    return smoothed, info
