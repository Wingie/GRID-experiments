"""Offline unit tests for the action vocabulary + token (de)serialisation."""

from rimworld_agent.game.action_space import (
    ACTION_SPACE,
    Action,
    action_special_tokens,
    format_action,
    format_actions,
    is_valid_action,
    parse_action,
    parse_actions,
    required_params,
)


def test_known_actions_present():
    assert is_valid_action("order_build")
    assert is_valid_action("speed_3x")
    assert not is_valid_action("teleport_colonist")
    assert "wait" in ACTION_SPACE


def test_required_params():
    assert required_params("order_build") == ["def_name", "x", "y"]
    assert required_params("speed_pause") == []


def test_structural_validity():
    ok, _ = Action("order_build", {"def_name": "Bed", "x": 15, "y": 22}).is_valid()
    assert ok
    bad, msg = Action("order_build", {"def_name": "Bed"}).is_valid()
    assert not bad and "missing" in msg
    unknown, msg = Action("fly").is_valid()
    assert not unknown and "unknown" in msg


def test_format_then_parse_roundtrip():
    a = Action("order_build", {"def_name": "SolarGenerator", "x": 42, "y": 18}, "need power")
    block = format_action(a)
    assert "<ACT:order_build>" in block and "<PARAM:def_name=SolarGenerator>" in block
    parsed = parse_action(block)
    assert parsed.action == "order_build"
    assert parsed.params == {"def_name": "SolarGenerator", "x": 42, "y": 18}
    assert parsed.reason == "need power"


def test_param_type_coercion():
    a = Action("increase_priority", {"pawn": "Doc", "work_type": "Construction", "delta": 1})
    parsed = parse_action(format_action(a))
    assert parsed.params["delta"] == 1 and isinstance(parsed.params["delta"], int)
    assert parsed.params["pawn"] == "Doc"


def test_parse_multiple_actions_and_cap():
    actions = [Action("speed_3x", reason=str(i)) for i in range(7)]
    text = format_actions(actions)  # capped at 5
    assert len(parse_actions(text)) == 5


def test_action_special_tokens_cover_vocab():
    toks = action_special_tokens()
    for name in ACTION_SPACE:
        assert f"<ACT:{name}>" in toks
    assert "<ACTION_START>" in toks and "<REASON>" in toks
