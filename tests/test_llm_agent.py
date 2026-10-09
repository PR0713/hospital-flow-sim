"""LLMAgent against a fake client built from real SDK message types (no network)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import numpy as np
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.eval import evaluate
from hospital_sim.generators import load_scenario
from hospital_sim_agents import BudgetExceededError, LLMAgent

SCENARIO = load_scenario(Path(__file__).parent.parent / "configs" / "example_hospital_dynamic.yaml")
CONFIG = EnvConfig(SCENARIO, split="eval")


def reply(*blocks: Any) -> Message:
    return Message(
        id="msg_test",
        content=list(blocks),
        model="claude-opus-5-5",
        role="assistant",
        stop_reason="tool_use",
        type="message",
        usage=Usage(input_tokens=1000, output_tokens=50),
    )


def tool_call(action: Any) -> ToolUseBlock:
    return ToolUseBlock(
        id="toolu_test",
        name="choose_action",
        input={"action": action, "reason": "test"},
        type="tool_use",
    )


class FakeClient:
    """``messages.create`` answers with whatever ``choose(prompt)`` returns."""

    def __init__(self, choose: Any) -> None:
        self.choose, self.prompts = choose, []
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        prompt = kwargs["messages"][0]["content"]
        self.prompts.append(prompt)
        result = self.choose(prompt)
        if isinstance(result, Exception):
            raise result
        return result


def first_listed(prompt: str) -> Message:
    return reply(tool_call(int(re.search(r"^(\d+):", prompt, re.M).group(1))))  # type: ignore[union-attr]


def one_decision(agent: LLMAgent) -> int:
    env = HospitalEnv(CONFIG)
    obs, _ = env.reset(seed=0)
    return agent.act(obs, env.action_masks())


def test_valid_choice_is_used_and_tokens_counted() -> None:
    agent = LLMAgent(CONFIG, client=FakeClient(first_listed))
    action = one_decision(agent)
    assert action == 0
    assert agent.stats.fallbacks == 0
    assert agent.stats.input_tokens == 1000
    assert agent.stats.cost_usd("claude-opus-5-5") == pytest.approx(0.005)


def test_invalid_choice_falls_back_and_is_counted() -> None:
    agent = LLMAgent(CONFIG, client=FakeClient(lambda p: reply(tool_call(10**6))))
    env = HospitalEnv(CONFIG)
    obs, _ = env.reset(seed=0)
    action = agent.act(obs, env.action_masks())
    assert env.action_masks()[action]
    assert agent.stats.invalid_choices == 1


def test_non_integer_choice_is_invalid() -> None:
    agent = LLMAgent(CONFIG, client=FakeClient(lambda p: reply(tool_call("0"))))
    one_decision(agent)
    assert agent.stats.invalid_choices == 1


def test_missing_tool_call_falls_back() -> None:
    text = reply(TextBlock(type="text", text="I think we should start the first one."))
    agent = LLMAgent(CONFIG, client=FakeClient(lambda p: text))
    one_decision(agent)
    assert agent.stats.no_tool_call == 1


def test_api_error_falls_back() -> None:
    error = anthropic.APIConnectionError(request=httpx2.Request("POST", "http://test"))
    agent = LLMAgent(CONFIG, client=FakeClient(lambda p: error))
    one_decision(agent)
    assert agent.stats.api_errors == 1


def test_call_budget_stops_before_spending() -> None:
    client = FakeClient(first_listed)
    agent = LLMAgent(CONFIG, client=client, max_calls=1)
    env = HospitalEnv(CONFIG)
    obs, _ = env.reset(seed=0)
    agent.act(obs, env.action_masks())
    with pytest.raises(BudgetExceededError):
        agent.act(obs, env.action_masks())
    assert len(client.prompts) == 1


def test_prompt_contains_only_public_information() -> None:
    client = FakeClient(first_listed)
    one_decision(LLMAgent(CONFIG, client=client))
    text = client.prompts[0]
    assert "Allowed actions" in text
    for forbidden in ("hidden", "realized", "no_show", "quantile"):
        assert forbidden not in text


def test_runs_through_the_evaluation_harness() -> None:
    agent = LLMAgent(CONFIG, client=FakeClient(first_listed), max_calls=10_000)
    result = evaluate(SCENARIO, [agent.as_env_agent()], n_episodes=2, split="eval")
    records = result.records[agent.name]
    assert len(records) == 2 and all(r.values["makespan"] > 0 for r in records)
    assert agent.stats.decisions == agent.stats.calls > 0
    assert agent.stats.fallbacks == 0
    assert np.isfinite(agent.stats.cost_usd("claude-opus-5-5"))
