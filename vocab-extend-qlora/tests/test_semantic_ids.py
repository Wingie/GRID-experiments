"""CPU-only tests for the semantic-ID layer: entity extraction, SID vocabulary
formatting/parsing, RQ-VAE shapes + dead-code reinit, and config inheritance.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from src.semantic_ids.assign_ids import SemanticIDVocab, code_width, sid_token  # noqa: E402
from src.semantic_ids.extract_entities import CodeEntity, extract_from_file  # noqa: E402
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


def test_parse_git_log():
    from src.semantic_ids.cochange import parse_git_log

    text = "__COMMIT__\na.py\nb.py\n__COMMIT__\nb.py\nc.py\n"
    commits = parse_git_log(text)
    assert commits == [{"a.py", "b.py"}, {"b.py", "c.py"}]


def test_build_cochange_graph_is_row_normalised():
    from src.semantic_ids.cochange import build_cochange_graph

    commits = [{"a.py", "b.py"}, {"a.py", "b.py"}, {"a.py", "c.py"}]
    graph = build_cochange_graph(commits)
    # a co-changes with b twice and c once -> normalised to 2/3, 1/3
    assert abs(graph["a.py"]["b.py"] - 2 / 3) < 1e-6
    assert abs(graph["a.py"]["c.py"] - 1 / 3) < 1e-6
    assert abs(sum(graph["a.py"].values()) - 1.0) < 1e-6


def test_build_cochange_graph_skips_huge_commits():
    from src.semantic_ids.cochange import build_cochange_graph

    huge = {f"f{i}.py" for i in range(60)}
    graph = build_cochange_graph([huge], max_files_per_commit=50)
    assert graph == {}


def test_cochange_smoothing_pulls_toward_neighbours():
    import numpy as np

    from src.semantic_ids.cochange import cochange_smoothing

    # two entities in two files that always co-change; orthogonal start embeddings
    e1 = CodeEntity(name="A", kind="function", file_path="a.py", body="", signature="")
    e2 = CodeEntity(name="B", kind="function", file_path="b.py", body="", signature="")
    emb = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    graph = {"a.py": {"b.py": 1.0}, "b.py": {"a.py": 1.0}}
    out = cochange_smoothing([e1, e2], emb, graph, alpha=0.5, repo_root=".")
    # smoothed embeddings move toward each other (cosine similarity increases)
    before = float(emb[0] @ emb[1])
    after = float(out[0] @ out[1])
    assert after > before
    # output stays unit-normalised and same shape
    assert out.shape == emb.shape
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-5


def test_cochange_disabled_is_noop():
    import numpy as np

    from src.semantic_ids.cochange import maybe_apply_cochange

    e = CodeEntity(name="A", kind="function", file_path="a.py", body="", signature="")
    emb = np.random.default_rng(0).standard_normal((1, 4)).astype("float32")
    out, info = maybe_apply_cochange([e], emb, {"semantic_ids": {"cochange": {"enabled": False}}})
    assert info == {"enabled": False}
    assert np.allclose(out, emb)


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
