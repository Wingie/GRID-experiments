"""Tests for knowledge extraction + semantic-ID machinery.

The XML-parsing and entity-graph tests are pure (lxml only). The SID-vocab test needs the
sibling vocab-extend-qlora checkout importable; the RQ-VAE test additionally needs torch and
is skipped cleanly when either is absent.
"""

import numpy as np
import pytest

from rimworld_agent.knowledge.extract_defs import extract_defs
from rimworld_agent.semantic_ids.build_entity_graph import build_entity_graph

FIXTURE_XML = """<?xml version="1.0" encoding="utf-8"?>
<Defs>
  <ThingDef Name="ApparelBase" Abstract="True">
    <statBases><MaxHitPoints>80</MaxHitPoints></statBases>
    <thingCategories><li>Apparel</li></thingCategories>
  </ThingDef>
  <ThingDef ParentName="ApparelBase">
    <defName>Apparel_Parka</defName>
    <label>parka</label>
    <description>A heavy jacket that keeps people warm.</description>
    <thingClass>Apparel</thingClass>
  </ThingDef>
  <ResearchProjectDef>
    <defName>Electricity</defName>
    <label>electricity</label>
    <techLevel>Industrial</techLevel>
  </ResearchProjectDef>
</Defs>
"""


@pytest.fixture
def xml_dir(tmp_path):
    (tmp_path / "Apparel.xml").write_text(FIXTURE_XML)
    return tmp_path


def test_extract_defs_inheritance(xml_dir):
    entities = extract_defs(xml_dir)
    by_name = {e.def_name: e for e in entities}
    # Abstract base is not emitted; two concrete defs are.
    assert set(by_name) == {"Apparel_Parka", "Electricity"}

    parka = by_name["Apparel_Parka"]
    assert parka.def_type == "ThingDef"
    assert parka.label == "parka"
    assert parka.parent_def == "ApparelBase"
    # Inherited fields from the abstract parent are merged in.
    assert parka.fields["statBases"]["MaxHitPoints"] == "80"
    assert parka.fields["thingCategories"] == ["Apparel"]


def test_entity_graph_categories(xml_dir):
    graph = build_entity_graph(extract_defs(xml_dir))
    assert graph.category_of("Apparel_Parka") == "Items"
    assert graph.subcategory_of("Apparel_Parka") == "Apparel"
    assert graph.category_of("Electricity") == "Research"
    tree = graph.as_tree()
    assert "Items" in tree and "Research" in tree


def test_text_blob_nonempty(xml_dir):
    entities = extract_defs(xml_dir)
    assert all(e.text_blob().strip() for e in entities)


def test_sid_vocab_format_and_parse():
    pytest.importorskip("torch")  # veq's assign_ids module pulls numpy; vocab itself is light
    from rimworld_agent.semantic_ids.assign_ids import _vocab_cls

    Vocab = _vocab_cls()
    v = Vocab(levels=3, codebook_size=64)
    assert len(v.per_level_tokens()) == 192
    assert len(v.special_tokens()) == 192 + 3 + 5
    assert v.token_to_level_code("<SID_L1_02>") == (0, 2)
    seq = v.format_sequence([2, 1, 5])
    assert seq.startswith("<SID_START>") and seq.endswith("<SID_END>")
    assert "<SID_L1_02>" in seq and "<SID_L3_05>" in seq


def test_rqvae_shapes_and_codebook():
    torch = pytest.importorskip("torch")
    from rimworld_agent.utils import ensure_veq_importable

    if not ensure_veq_importable():
        pytest.skip("vocab-extend-qlora not importable")
    from src.semantic_ids.rqvae import train_rqvae  # type: ignore

    rng = np.random.default_rng(0)
    emb = rng.standard_normal((200, 16)).astype("float32")
    cfg = {
        "seed": 0,
        "semantic_ids": {
            "embedding_dim": 16,
            "rqvae": {
                "levels": 3,
                "codebook_size": 8,
                "latent_dim": 16,
                "hidden_dim": 32,
                "train_steps": 50,
                "batch_size": 64,
                "lr": 1e-3,
            },
        },
    }
    model = train_rqvae(emb, cfg)
    codes = model.encode_to_ids(torch.from_numpy(emb))
    assert tuple(codes.shape) == (200, 3)
    assert codes.min().item() >= 0 and codes.max().item() < 8
    cb = model.codebook_vectors()
    assert tuple(cb.shape) == (3, 8, 16)
