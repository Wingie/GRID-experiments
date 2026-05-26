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


def test_dual_sid_vocab_format_and_parse():
    from rimworld_agent.semantic_ids.sid_vocab import SIDVocab

    rv = SIDVocab(prefix="RSID", levels=3, codebook_size=64)
    wv = SIDVocab(prefix="WSID", levels=3, codebook_size=64)
    # Each family contributes 192 per-level tokens + 3 structural; they do NOT collide.
    assert len(rv.per_level_tokens()) == 192 and len(wv.per_level_tokens()) == 192
    assert set(rv.special_tokens()).isdisjoint(set(wv.special_tokens()))
    assert rv.token_to_level_code("<RSID_L1_02>") == (0, 2)
    assert wv.token_to_level_code("<WSID_L3_07>") == (2, 7)
    assert rv.token_to_level_code("<WSID_L1_02>") is None  # wrong family
    rseq = rv.format_sequence([2, 1, 5])
    assert rseq.startswith("<RSID_START>") and "<RSID_L1_02>" in rseq and rseq.endswith("<RSID_END>")
    wseq = wv.format_inline([5, 2, 3])
    assert wseq == "<WSID_L1_05><WSID_L2_02><WSID_L3_03>"


def test_cooccurrence_and_write_embeddings(tmp_path):
    from rimworld_agent.game.action_space import Action
    from rimworld_agent.game.episode_recorder import EpisodeRecorder, GameState
    from rimworld_agent.semantic_ids.collect_cooccurrence import collect_cooccurrence
    from rimworld_agent.semantic_ids.rqvae_write import build_write_embeddings

    # Two episodes where SolarGenerator + Battery are built together (workflow co-occurrence).
    for ep_i in range(2):
        rec = EpisodeRecorder(episode_id=f"ep_{ep_i}", root=tmp_path)
        rec.record(GameState(), "", [
            Action("order_build", {"def_name": "SolarGenerator", "x": 1, "y": 1}),
            Action("order_build", {"def_name": "Battery", "x": 2, "y": 1}),
        ], {"total": 1.0})
        rec.record(GameState(), "", [Action("order_research", {"project_def": "Electricity"})], {"total": 1.0})
        rec.save()

    cooc = collect_cooccurrence(tmp_path, window=5)
    assert set(cooc.def_names) == {"SolarGenerator", "Battery", "Electricity"}
    si, bi = cooc.index["SolarGenerator"], cooc.index["Battery"]
    assert cooc.matrix[si, bi] > 0  # they co-occur within a step
    emb = build_write_embeddings(cooc, dim=16)
    assert emb.shape == (3, 16)


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
