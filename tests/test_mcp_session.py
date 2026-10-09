"""The MCP session layer: plain-Python logic behind the server tools."""

from __future__ import annotations

from typing import Any

import pytest

from hospital_sim_mcp.session import EpisodeSession, SessionError

CONFIG = "example_hospital_dynamic"


def play_to_end(session: EpisodeSession, limit: int = 5000) -> dict[str, Any]:
    last: dict[str, Any] = {}
    for _ in range(limit):
        actions = session.valid_actions()
        if not actions:
            break
        starts = [a for a in actions if a["kind"] == "start"]
        last = session.step((starts or actions)[0]["action"])
        if last["episode_over"]:
            break
    return last


def test_configs_listed_and_unknown_rejected() -> None:
    session = EpisodeSession()
    assert CONFIG in session.available_configs()
    with pytest.raises(SessionError, match="Unknown config"):
        session.reset("../../etc/passwd")


def test_tools_need_reset_first() -> None:
    with pytest.raises(SessionError, match="reset"):
        EpisodeSession().state()


def test_full_episode_reaches_metrics() -> None:
    session = EpisodeSession()
    session.reset(CONFIG, seed=1)
    play_to_end(session)
    metrics = session.metrics()
    assert metrics["episode_over"]
    assert metrics["episode"]["patients_finished"] > 0
    with pytest.raises(SessionError, match="over"):
        session.step(0)


def test_invalid_action_is_recoverable() -> None:
    session = EpisodeSession()
    session.reset(CONFIG, seed=1)
    result = session.step(10**6)
    assert result["ok"] is False and result["valid_actions"]
    # The episode is still alive and a valid action still works.
    assert session.step(result["valid_actions"][0]["action"])["ok"] is True


def test_same_seed_same_first_state() -> None:
    a, b = EpisodeSession(), EpisodeSession()
    assert a.reset(CONFIG, seed=3)["state"] == b.reset(CONFIG, seed=3)["state"]
    assert a.valid_actions() == b.valid_actions()


def test_descriptions_do_not_leak_hidden_information() -> None:
    session = EpisodeSession()
    session.reset(CONFIG, seed=2)
    text = session.state() + " ".join(a["description"] for a in session.valid_actions())
    for forbidden in ("hidden", "realized", "no_show", "quantile"):
        assert forbidden not in text
