"""Offline tests for the EVE backend: action space, SDE type parsing, reward.

Network is never hit — the ESI client isn't exercised. The backend factory and the
``ExecutionResult`` for an unauthenticated action are validated.
"""

import textwrap

import pytest

from rimworld_agent.games.base import Action, get_backend
from rimworld_agent.games.eve import EVE_ACTION_SPACE, EVEState, compute_reward_eve, extract_sde_types


def test_eve_action_space_covers_expected_endpoints():
    for required in ("skill_train", "order_buy", "order_sell", "set_destination", "send_mail", "wait"):
        assert required in EVE_ACTION_SPACE
    assert EVE_ACTION_SPACE["skill_train"]["auth"] is True
    assert EVE_ACTION_SPACE["wait"]["type"] == "noop"


def test_eve_backend_factory_and_unauth_execute(monkeypatch):
    monkeypatch.delenv("EVE_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("EVE_CHARACTER_ID", raising=False)
    b = get_backend("eve")
    assert b.name == "eve"
    # wait is a noop and always succeeds.
    assert b.execute(Action("wait")).ok
    # auth-required action without a token surfaces a clear error, no network call.
    r = b.execute(Action("set_destination", {"destination_id": 30000142}))
    assert not r.ok and r.error and "auth" in r.error


def test_compute_reward_eve():
    prev = EVEState(wallet_isk=1_000_000, skill_queue=[{}], market_orders=[{}, {}])
    curr = EVEState(wallet_isk=2_500_000, skill_queue=[{}, {}], market_orders=[{}])
    r = compute_reward_eve(prev, curr, [Action("order_sell")])
    assert pytest.approx(r["isk_delta"], rel=1e-6) == 1_500_000 * 1e-6
    assert r["skills_queued"] == 1.0
    assert r["orders_filled"] == 2.0  # one fewer order now -> assume filled
    assert r["total"] > 0


def test_extract_sde_types(tmp_path):
    pytest.importorskip("yaml")
    (tmp_path / "fsd").mkdir()
    (tmp_path / "fsd" / "types.yaml").write_text(textwrap.dedent("""
        587:
          name: {en: Rifter}
          description: {en: A nimble Minmatar frigate.}
          groupID: 25
          marketGroupID: 71
          published: true
        34:
          name: {en: Tritanium}
          description: {en: The most common mineral type.}
          groupID: 18
          marketGroupID: 1857
          published: true
    """).strip())
    types = extract_sde_types(tmp_path)
    by_id = {t.type_id: t for t in types}
    assert by_id[587].name == "Rifter"
    assert by_id[587].group_id == 25
    assert by_id[34].name == "Tritanium"
    assert by_id[587].text_blob().startswith("EVE type 587 Rifter")
