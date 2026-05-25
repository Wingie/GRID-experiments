"""Shared helpers: the Hydra<->vocab-extend-qlora config bridge, logging, JSON IO, and a
locator that makes the sibling ``vocab-extend-qlora`` package importable as ``src``.

Why the bridge exists
---------------------
We configure runs with Hydra/OmegaConf, but we reuse functions from ``vocab-extend-qlora``
(token mining, tokenizer extension, RQ-VAE, QLoRA, GGUF export). Those functions read
their config with ``src.utils.cfg_get`` which does ``isinstance(node, dict)`` checks — an
OmegaConf ``DictConfig`` is *not* a ``dict`` and would silently yield defaults. So every
call into vocab-extend-qlora must pass a *plain* dict produced by :func:`to_veq_cfg`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(f"rsa.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


log = get_logger("utils")


# --------------------------------------------------------------------------- #
# vocab-extend-qlora (veq) locator
# --------------------------------------------------------------------------- #
def ensure_veq_importable(explicit_path: str | os.PathLike | None = None) -> bool:
    """Make the sibling ``vocab-extend-qlora`` checkout importable as the ``src`` package.

    Resolution order:
      1. an already-installed ``src`` package (``pip install -e ../vocab-extend-qlora``);
      2. ``explicit_path`` if given;
      3. ``$VEQ_PATH`` environment variable;
      4. a sibling ``../vocab-extend-qlora`` directory next to this repo.

    Returns ``True`` if ``import src`` will resolve to a vocab-extend-qlora checkout.
    Note: when the current working directory contains its own top-level ``src`` package
    (e.g. running from the GRID repo root), that one shadows the sibling — run the
    pipeline from this repo's directory, where there is no local ``src``.
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.environ.get("VEQ_PATH"):
        candidates.append(Path(os.environ["VEQ_PATH"]))
    here = Path(__file__).resolve()
    # repo root is .../rimworld-sovereign-agent; sibling is .../vocab-extend-qlora
    candidates.append(here.parents[1].parent / "vocab-extend-qlora")

    for cand in candidates:
        if (cand / "src" / "__init__.py").exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            return True

    # maybe it is pip-installed already
    try:
        import importlib

        importlib.import_module("src.utils")
        return True
    except Exception:
        log.warning(
            "vocab-extend-qlora not found. Install it with `pip install -e "
            "../vocab-extend-qlora` or set $VEQ_PATH. Reuse stages will fail until then."
        )
        return False


# --------------------------------------------------------------------------- #
# Hydra <-> veq config bridge
# --------------------------------------------------------------------------- #
def to_veq_cfg(cfg: Any) -> dict:
    """Convert an OmegaConf ``DictConfig`` (or any mapping) into a plain ``dict`` that
    vocab-extend-qlora's ``cfg_get`` can read.

    vocab-extend-qlora expects keys under the dotted namespaces it defines
    (``mining.*``, ``semantic_ids.*``, ``extend.*``, ``data.*``, ``qlora.*``, ``eval.*``,
    ``paths.*``, ``model.*``, ``seed``). Our Hydra configs use those same names by design,
    so this is a structural conversion, not a remapping.
    """
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            container = OmegaConf.to_container(cfg, resolve=True)
            assert isinstance(container, dict)
            return container
    except ImportError:  # omegaconf not installed; assume already a dict
        pass
    if isinstance(cfg, dict):
        return cfg
    raise TypeError(f"Cannot convert {type(cfg)!r} to a veq cfg dict")


def cfg_get(cfg: Any, dotted_key: str, default: Any = None) -> Any:
    """Read a nested value via dotted path from a dict or OmegaConf config.

    Works on both plain dicts and OmegaConf ``DictConfig`` (unlike veq's ``cfg_get``,
    which is dict-only). Use this for *our* code; use :func:`to_veq_cfg` before calling
    into vocab-extend-qlora.
    """
    node: Any = cfg
    for part in dotted_key.split("."):
        try:
            node = node[part]
        except (KeyError, TypeError, AttributeError):
            try:  # OmegaConf attribute access
                node = getattr(node, part)
            except AttributeError:
                return default
    return node


# --------------------------------------------------------------------------- #
# JSON IO
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):  # numpy arrays / torch tensors
        return obj.tolist()
    if hasattr(obj, "item"):  # numpy / torch scalars
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


def write_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=indent, default=_json_default)
    return p


def read_json(path: str | os.PathLike) -> Any:
    with open(path) as fh:
        return json.load(fh)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
