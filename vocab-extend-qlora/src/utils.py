"""Shared helpers: config loading with inheritance, seeding, logging, repo walking,
and JSON result IO. Every script imports from here so paths and conventions stay
consistent across the pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml

# Map source-file extension -> (language name, tree-sitter package module).
# The language name is what we record in results; the module is imported lazily by
# the entity extractor, so a missing grammar only disables that one language.
LANG_EXTS: dict[str, tuple[str, str]] = {
    "py": ("python", "tree_sitter_python"),
    "ts": ("typescript", "tree_sitter_typescript"),
    "tsx": ("tsx", "tree_sitter_typescript"),
    "js": ("javascript", "tree_sitter_javascript"),
    "jsx": ("javascript", "tree_sitter_javascript"),
    "rs": ("rust", "tree_sitter_rust"),
    "go": ("go", "tree_sitter_go"),
    "java": ("java", "tree_sitter_java"),
    "rb": ("ruby", "tree_sitter_ruby"),
    "sol": ("solidity", "tree_sitter_solidity"),
}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_with_inheritance(path: Path, _seen: set[Path] | None = None) -> dict:
    """Load a YAML config, resolving an optional ``inherit: [..]`` list of parent
    configs (paths relative to the ``configs/`` directory). Parents are merged in
    order, then the current file is layered on top.
    """
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"Circular config inheritance involving {path}")
    _seen.add(path)

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    parents = raw.pop("inherit", []) or []
    configs_root = path.parent
    # `inherit` entries are resolved relative to the configs root (the dir holding
    # base.yaml). We find it by walking up until we see base.yaml, else use parent dir.
    root = configs_root
    while root != root.parent and not (root / "base.yaml").exists():
        root = root.parent
    if not (root / "base.yaml").exists():
        root = configs_root

    merged: dict = {}
    for parent in parents:
        parent_cfg = _load_with_inheritance(root / parent, _seen=set(_seen))
        _deep_update(merged, parent_cfg)
    _deep_update(merged, raw)
    return merged


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI ``key=value`` overrides."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply dotted ``a.b.c=value`` overrides (e.g. from argv) to a config dict."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _coerce(value)
    return cfg


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> dict:
    """Load an experiment/model config with inheritance and optional CLI overrides."""
    cfg = _load_with_inheritance(Path(path))
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg


def cfg_get(cfg: dict, dotted_key: str, default: Any = None) -> Any:
    """Read a nested value via a dotted path, returning ``default`` if absent."""
    node: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# --------------------------------------------------------------------------- #
# Reproducibility / logging
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Repo walking
# --------------------------------------------------------------------------- #
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", "target"}


@dataclass
class RepoFile:
    path: Path           # absolute path on disk
    rel_path: str        # path relative to the repo root (stable identifier)
    ext: str             # extension without the dot
    language: str        # canonical language name from LANG_EXTS
    text: str            # file contents


def iter_repo_files(repo_root: str | Path, languages: Iterable[str]) -> Iterator[RepoFile]:
    """Yield source files under ``repo_root`` whose extension is in ``languages``.

    Skips common vendored / build directories and unreadable (binary) files.
    """
    repo_root = Path(repo_root)
    allowed = {ext.lstrip(".") for ext in languages}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            ext = fname.rsplit(".", 1)[-1] if "." in fname else ""
            if ext not in allowed:
                continue
            fpath = Path(dirpath) / fname
            try:
                text = fpath.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            language = LANG_EXTS.get(ext, (ext, ""))[0]
            yield RepoFile(
                path=fpath,
                rel_path=str(fpath.relative_to(repo_root)),
                ext=ext,
                language=language,
                text=text,
            )


# --------------------------------------------------------------------------- #
# JSON IO
# --------------------------------------------------------------------------- #
def write_json(obj: Any, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return path


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays / torch tensors
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy / torch scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- #
# Tokenizer loading (with an offline fallback for CPU-only / no-download envs)
# --------------------------------------------------------------------------- #
import re as _re

_TOKEN_SPLIT = _re.compile(r"[A-Za-z0-9]+|[^A-Za-z0-9\s]|\s+")
_CAMEL_SPLIT = _re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")


class FallbackTokenizer:
    """Deterministic, dependency-free stand-in used only for *counting* sub-tokens
    when the real HF tokenizer cannot be downloaded (offline / CI). It approximates
    subword behaviour by splitting on word boundaries + camelCase and chunking long
    runs into ~4-character pieces. Not suitable for training — only for mining,
    analysis, and compression estimation.
    """

    name_or_path = "fallback"

    def __init__(self, chunk: int = 4):
        self.chunk = chunk

    def tokenize(self, text: str) -> list[str]:
        pieces: list[str] = []
        for tok in _TOKEN_SPLIT.findall(text):
            if tok.isspace():
                continue
            subs = _CAMEL_SPLIT.findall(tok) or [tok]
            for sub in subs:
                for i in range(0, len(sub), self.chunk):
                    pieces.append(sub[i : i + self.chunk])
        return pieces

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [hash(p) % 50000 for p in self.tokenize(text)]

    def __len__(self) -> int:  # nominal vocab size
        return 50000


def load_hf_tokenizer(model_id: str, allow_fallback: bool = True, **kwargs):
    """Load a HuggingFace tokenizer, falling back to :class:`FallbackTokenizer` when
    the model cannot be fetched (no network / no cache) and ``allow_fallback`` is set.

    Returns ``(tokenizer, is_real)``.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
        return tok, True
    except Exception as exc:  # network, auth, or transformers missing
        if not allow_fallback:
            raise
        get_logger("utils").warning(
            "Could not load tokenizer %s (%s); using offline FallbackTokenizer.",
            model_id,
            type(exc).__name__,
        )
        return FallbackTokenizer(), False


def count_subtokens(tokenizer, text: str) -> int:
    """Number of sub-tokens the tokenizer fragments ``text`` into."""
    return len(tokenizer.tokenize(text))
