"""A single-LLM scheduling policy: one tool-call decision per environment step.

``LLMAgent.act(observation, mask)`` has the same signature as every other
policy, so the evaluation harness scores it against the baselines on identical
instances. Each decision is a stateless request: the full situation is in the
prompt, and the model answers by calling ``choose_action``.

Failures never crash an episode. An invalid choice, a missing tool call or an
API error falls back to the longest-waiting startable operation, and each is
counted in ``stats`` so a result can be reported with its fallback rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import anthropic
import numpy as np
from anthropic.types import ToolParam, ToolUseBlock
from numpy.typing import NDArray

from hospital_sim.env import EnvConfig, HospitalEnv
from hospital_sim.env.observation import OP_FEATURES
from hospital_sim.eval import EnvAgent
from hospital_sim_mcp.describe import describe_actions, describe_state

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens (input, output), from the Claude API model table.
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}

SYSTEM_PROMPT = """\
You are the scheduler of a small hospital unit. At each decision you see the \
current state and the list of actions allowed right now, and you must pick one.

Goal: minimise the weighted time patients spend in the system (flow time). \
Patients with a higher weight, urgent patients and high-acuity patients matter more. \
A patient who leaves unseen is heavily penalised.

How to decide:
- Starting an operation uses a whole team at once, so it only appears when that team is free.
- Do not wait while a useful operation can be started, unless holding resources for a more \
important patient who is about to arrive is clearly better.
- Durations are uncertain; use the expected values shown. Everything you need is in the \
state: do not assume anything else.

Respond only by calling choose_action with one action number from the list."""

CHOOSE_ACTION_TOOL: ToolParam = {
    "name": "choose_action",
    "description": "Choose the action to take now. The number must come from the list given.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "integer", "description": "An action number from the list."},
            "reason": {"type": "string", "description": "One short sentence."},
        },
        "required": ["action", "reason"],
        "additionalProperties": False,
    },
}


class BudgetExceededError(RuntimeError):
    """Raised before a request that would exceed ``max_calls``."""


@dataclass
class AgentStats:
    calls: int = 0
    decisions: int = 0
    invalid_choices: int = 0
    no_tool_call: int = 0
    api_errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def fallbacks(self) -> int:
        return self.invalid_choices + self.no_tool_call + self.api_errors

    def cost_usd(self, model: str) -> float:
        price_in, price_out = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * price_in + self.output_tokens * price_out) / 1e6


class LLMAgent:
    def __init__(
        self,
        config: EnvConfig,
        *,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: Literal["low", "medium", "high"] = "low",
        max_calls: int = 300,
        name: str | None = None,
    ) -> None:
        # Used only for init-time metadata (names, time scale, action layout).
        self.env = HospitalEnv(config)
        self.client = client if client is not None else anthropic.Anthropic()
        self.model, self.effort, self.max_calls = model, effort, max_calls
        self.name = name or f"llm_{model}"
        self.stats = AgentStats()
        self._waited = OP_FEATURES.index("waited")

    def as_env_agent(self) -> EnvAgent:
        return EnvAgent(self.name, self.act, self.env.config)

    # --- policy --------------------------------------------------------------

    def act(self, observation: dict[str, Any], mask: NDArray[np.bool_]) -> int:
        self.stats.decisions += 1
        actions = describe_actions(self.env, observation, mask)
        valid = {a["action"] for a in actions}
        chosen = self._ask(observation, actions)
        if chosen is None:
            return self._fallback(observation, actions)
        if chosen not in valid:
            self.stats.invalid_choices += 1
            return self._fallback(observation, actions)
        return chosen

    def _ask(self, observation: dict[str, Any], actions: list[dict[str, Any]]) -> int | None:
        if self.stats.calls >= self.max_calls:
            raise BudgetExceededError(f"reached max_calls={self.max_calls}")
        self.stats.calls += 1
        listing = "\n".join(f"{a['action']}: {a['description']}" for a in actions)
        prompt = f"{describe_state(self.env, observation)}\n\nAllowed actions:\n{listing}"
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=[CHOOSE_ACTION_TOOL],
                tool_choice={"type": "auto"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError:
            self.stats.api_errors += 1
            return None
        usage = response.usage
        self.stats.input_tokens += usage.input_tokens
        self.stats.output_tokens += usage.output_tokens
        self.stats.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
        for block in response.content:
            if isinstance(block, ToolUseBlock) and block.name == "choose_action":
                value = block.input.get("action")
                return value if isinstance(value, int) else -1
        self.stats.no_tool_call += 1
        return None

    def _fallback(self, observation: dict[str, Any], actions: list[dict[str, Any]]) -> int:
        """Longest-waiting startable operation, else the first allowed action."""
        n_rules = len(self.env.config.allocation_rules)
        starts = [a["action"] for a in actions if a["kind"] == "start"]
        if not starts:
            return int(actions[0]["action"])
        waited = observation["ops"][:, self._waited]
        return int(max(starts, key=lambda a: waited[a // n_rules]))
