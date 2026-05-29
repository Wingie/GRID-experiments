"""Offline tests for the response-only kiosk backend.

Confirms the action space is strictly response-only (no asking back), the catalog loads,
the four respond actions work, the no-question-mark safeguard fires, and the reward shaper
rewards grounded citations / penalises asking back.
"""

import json
import textwrap

import pytest

from rimworld_agent.games.base import Action, get_backend
from rimworld_agent.games.kiosk import (
    KIOSK_ACTION_SPACE,
    Catalog,
    CatalogItem,
    KioskBackend,
    _strip_questions,
    score_response,
)


@pytest.fixture
def catalog_file(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({
        "items": [
            {"item_id": "rifle_01", "name": "Bolt-Action Rifle", "category": "weapons",
             "price": 1200.0, "stock": 4, "rsid": "<RSID_L1_01><RSID_L2_03><RSID_L3_00>",
             "description": "Reliable long-range bolt action."},
            {"item_id": "parka_01", "name": "Wool Parka", "category": "apparel",
             "price": 250.0, "stock": 12, "rsid": "<RSID_L1_02><RSID_L2_04><RSID_L3_07>",
             "description": "Warm wool jacket."},
        ],
    }))
    return path


@pytest.fixture
def kiosk(catalog_file, tmp_path):
    questions = tmp_path / "questions.txt"
    questions.write_text(textwrap.dedent("""
        How much is the rifle?
        Do you have any parkas in stock?
        Recommend a warm jacket.
    """).strip())
    return KioskBackend(catalog_path=catalog_file, questions_path=questions, data_root=tmp_path)


def test_action_space_is_response_only():
    # No action solicits info from the customer.
    for name in KIOSK_ACTION_SPACE:
        assert "ask" not in name and "request" not in name and "clarif" not in name
    assert "answer" in KIOSK_ACTION_SPACE
    assert KIOSK_ACTION_SPACE["wait"]["type"] == "noop"


def test_catalog_load_and_lookup(catalog_file):
    cat = Catalog.load(catalog_file)
    assert set(cat.items) == {"rifle_01", "parka_01"}
    assert cat.lookup("rifle_01").price == 1200.0
    assert cat.by_category("weapons")[0].item_id == "rifle_01"
    assert cat.search("warm")[0].item_id == "parka_01"


def test_observe_pulls_next_question(kiosk):
    obs = kiosk.observe()
    assert obs.metadata["question"].startswith("How much")
    assert obs.metadata["queue_depth"] == 2  # 3 questions total, 1 in flight


def test_lookup_price_grounded(kiosk):
    kiosk.observe()
    r = kiosk.execute(Action("lookup_price", {"item_id": "rifle_01"}))
    assert r.ok and "Bolt-Action Rifle" in r.result["text"] and r.result["cited_items"] == ["rifle_01"]


def test_inventory_query_unknown_item(kiosk):
    kiosk.observe()
    r = kiosk.execute(Action("inventory_query", {"item_id": "nope"}))
    assert r.ok and r.result["cited_items"] == [] and "Not stocked" in r.result["text"]


def test_recommend_appends_rationale(kiosk):
    kiosk.observe()
    r = kiosk.execute(Action("recommend", {"item_id": "parka_01", "rationale": "Wool insulates well."}))
    assert r.ok and "Wool Parka" in r.result["text"] and "insulates" in r.result["text"]


def test_response_strips_questions(kiosk):
    # Even if the model emits a clarifying question, the backend strips it before publishing.
    kiosk.observe()
    r = kiosk.execute(Action("answer", {"text": "We have one. Anything else?", "cited_items": ["rifle_01"]}))
    assert "?" not in r.result["text"]
    assert _strip_questions("Got it?") == "Got it."


def test_register_and_observe_via_factory(catalog_file, tmp_path):
    b = get_backend("kiosk", catalog_path=catalog_file, data_root=tmp_path)
    assert b.name == "kiosk"
    assert "answer" in b.action_space()


def test_score_response_rewards_grounded_and_punishes_asking():
    cat = Catalog(items={"x": CatalogItem("x", "Widget", price=1.0)})
    good = {"text": "Widget is 1.00.", "cited_items": ["x"]}
    asked_back = {"text": "Which size do you want?", "cited_items": ["x"]}
    empty = {"text": "", "cited_items": []}
    sg = score_response("how much?", good, cat)
    sa = score_response("how much?", asked_back, cat)
    se = score_response("how much?", empty, cat)
    assert sg["total"] > sa["total"]
    assert sa["asked_back_penalty"] == -2.0
    assert se["empty_penalty"] == -1.0
