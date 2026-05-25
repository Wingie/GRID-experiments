"""CPU-only tests for the semantic-ID layer: entity extraction, SID vocabulary
formatting/parsing, RQ-VAE shapes + dead-code reinit, and config inheritance.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.semantic_ids.assign_ids import SemanticIDVocab, code_width, sid_token  # noqa: E402
from src.semantic_ids.extract_entities import extract_from_file  # noqa: E402
from src.utils import RepoFile, load_config  # noqa: E402

CONFIGS = pathlib.Path(__file__).parents[1] / "configs"

SAMPLE = '''\
class Config:
    """A config object."""

    def load(self, path):
        return open(path).read()

    def validate(self):
        return True


def top_level(x):
    return x + 1
'''


def _repo_file(text: str) -> RepoFile:
    return RepoFile(
        path=pathlib.Path("sample.py"),
        rel_path="sample.py",
        ext="py",
        language="python",
        text=text,
    )


def test_extract_python_entities():
    entities = extract_from_file(_repo_file(SAMPLE))
    by_name = {e.name: e for e in entities}
    assert "Config" in by_name and by_name["Config"].kind == "class"
    assert by_name["Config"].docstring == "A config object."
    assert set(by_name["Config"].children) == {"load", "validate"}
    assert by_name["load"].kind == "method" and by_name["load"].parent == "Config"
    assert by_name["top_level"].kind == "function" and by_name["top_level"].parent is None


def test_sid_token_formatting():
    assert code_width(64) == 2
    assert code_width(256) == 3
    assert sid_token(1, 7, 2) == "<SID_L1_07>"
    assert sid_token(3, 63, 2) == "<SID_L3_63>"


def test_sid_vocab_roundtrip():
    vocab = SemanticIDVocab(levels=3, codebook_size=64)
    assert len(vocab.per_level_tokens()) == 3 * 64
    # per_level ordering: index = level0 * K + code
    tokens = vocab.per_level_tokens()
    assert tokens[0] == "<SID_L1_00>"
    assert tokens[64] == "<SID_L2_00>"
    # structural + task included
    specials = vocab.special_tokens()
    assert "<SID_START>" in specials and "<PREDICT_SID>" in specials
    # parse round-trip
    assert vocab.token_to_level_code("<SID_L2_05>") == (1, 5)
    assert vocab.token_to_level_code("<SID_START>") is None
    seq = vocab.format_sequence([1, 2, 3])
    assert seq.startswith("<SID_START>") and seq.endswith("<SID_END>") and "<SID_SEP>" in seq
    assert vocab.format_inline([1, 2, 3]) == "<SID_L1_01><SID_L2_02><SID_L3_03>"


def test_config_inheritance():
    cfg = load_config(CONFIGS / "experiments" / "semid_qlora.yaml")
    assert cfg["name"] == "semid_qlora"
    assert cfg["semantic_ids"]["enabled"] is True
    assert cfg["extend"]["init_method"] == "codebook"
    assert cfg["model"]["id"].startswith("Qwen/")  # merged from model config
    assert cfg["qlora"]["lora_r"] == 32             # inherited from base


def test_config_override():
    cfg = load_config(CONFIGS / "experiments" / "baseline.yaml", ["mining.top_n=256", "seed=7"])
    assert cfg["mining"]["top_n"] == 256
    assert cfg["seed"] == 7


# --------------------------------------------------------------------------- #
# RQ-VAE (needs torch)
# --------------------------------------------------------------------------- #
def test_rqvae_shapes_and_training():
    torch = pytest.importorskip("torch")
    import numpy as np

    from src.semantic_ids.rqvae import RQVAEConfig, ResidualVQVAE, train_rqvae

    cfg = {
        "seed": 0,
        "semantic_ids": {
            "embedding_dim": 32,
            "rqvae": {
                "levels": 3,
                "codebook_size": 16,
                "latent_dim": 24,
                "hidden_dim": 48,
                "train_steps": 50,
                "batch_size": 32,
                "dead_code_threshold": 5,
            },
        },
    }
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((128, 32)).astype("float32")
    model = train_rqvae(emb, cfg)

    ids = model.encode_to_ids(torch.from_numpy(emb))
    assert ids.shape == (128, 3)
    assert int(ids.min()) >= 0 and int(ids.max()) < 16

    cb = model.codebook_vectors()
    assert cb.shape == (3, 16, 24)

    usage = model.codebook_usage()
    assert len(usage) == 3 and all(0.0 <= u <= 1.0 for u in usage)

    # forward returns the expected loss components
    out = model(torch.from_numpy(emb[:16]), update=False)
    assert "loss" in out and "indices" in out
    assert out["indices"].shape == (16, 3)

    # config dataclass picks up overrides
    rc = RQVAEConfig.from_cfg(cfg)
    assert rc.levels == 3 and rc.codebook_size == 16
    assert isinstance(model, ResidualVQVAE)


def test_rqvae_dead_code_reinit_runs():
    torch = pytest.importorskip("torch")
    import numpy as np

    from src.semantic_ids.rqvae import EMAVectorQuantizer

    q = EMAVectorQuantizer(num_codes=8, dim=4, decay=0.9, dead_threshold=2, eps=1e-5)
    q.train()
    x = torch.from_numpy(np.random.default_rng(0).standard_normal((16, 4)).astype("float32"))
    for _ in range(10):
        quantized, idx, commit = q(x, update=True)
    assert quantized.shape == x.shape
    assert idx.shape == (16,)
    assert commit.item() >= 0.0
