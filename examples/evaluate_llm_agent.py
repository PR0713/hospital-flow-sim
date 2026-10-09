"""Score the single-LLM scheduling agent against the baseline rules.

    .venv/bin/python examples/evaluate_llm_agent.py --dry-run              # free, scripted stand-in
    .venv/bin/python examples/evaluate_llm_agent.py --model claude-haiku-5-5 --episodes 5 --yes

Real runs call the Anthropic API (needs ANTHROPIC_API_KEY or `ant auth login`) and
cost money, so they print an estimate first and need --yes.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from anthropic.types import Message, ToolUseBlock, Usage

from hospital_sim.baselines import Policy
from hospital_sim.env import EnvConfig
from hospital_sim.eval import evaluate
from hospital_sim.generators import load_scenario
from hospital_sim_agents import BudgetExceededError, LLMAgent
from hospital_sim_agents.llm_agent import DEFAULT_MODEL, PRICES

CONFIGS = Path(__file__).parent.parent / "configs"
# Measured with random play on example_hospital_dynamic (13-114 per episode).
DECISIONS_PER_EPISODE = 75
# Rough assumptions for the estimate only; the real numbers are printed afterwards.
TOKENS_IN, TOKENS_OUT = 1500, 300


class ScriptedClient:
    """Stand-in model for --dry-run: always picks the first allowed action."""

    def __init__(self) -> None:
        self.messages = self

    def create(self, **kwargs: Any) -> Message:
        prompt = kwargs["messages"][0]["content"]
        action = int(re.search(r"^(\d+):", prompt, re.M).group(1))  # type: ignore[union-attr]
        call = ToolUseBlock(
            id="toolu_dry", name="choose_action", input={"action": action}, type="tool_use"
        )
        return Message(
            id="msg_dry",
            content=[call],
            model="scripted",
            role="assistant",
            stop_reason="tool_use",
            type="message",
            usage=Usage(input_tokens=0, output_tokens=0),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", default="example_hospital_dynamic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high"])
    parser.add_argument(
        "--max-calls", type=int, default=None, help="hard stop (default: 2x estimate)"
    )
    parser.add_argument("--split", default="eval", choices=["eval", "test"])
    parser.add_argument("--dry-run", action="store_true", help="scripted model, no API calls")
    parser.add_argument("--yes", action="store_true", help="confirm spending real API money")
    args = parser.parse_args()

    estimate_calls = args.episodes * DECISIONS_PER_EPISODE
    price_in, price_out = PRICES.get(args.model, (0.0, 0.0))
    estimate_usd = estimate_calls * (TOKENS_IN * price_in + TOKENS_OUT * price_out) / 1e6
    if not args.dry_run:
        print(f"Estimate: ~{estimate_calls} calls, ~${estimate_usd:.2f} on {args.model}")
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            sys.exit("No ANTHROPIC_API_KEY set (or run `ant auth login`). Use --dry-run to test.")
        if not args.yes:
            sys.exit("Re-run with --yes to spend that. Nothing was called.")

    scenario = load_scenario(CONFIGS / f"{args.config}.yaml")
    config = EnvConfig(scenario, split=args.split)
    agent = LLMAgent(
        config,
        client=ScriptedClient() if args.dry_run else None,
        model=args.model,
        effort=args.effort,
        max_calls=args.max_calls or 2 * estimate_calls,
        name="llm_scripted" if args.dry_run else f"llm_{args.model}",
    )
    policies = [
        Policy("fifo"),
        Policy("urgent_first"),
        Policy("weight_aware"),
        agent.as_env_agent(),
    ]
    try:
        result = evaluate(scenario, policies, n_episodes=args.episodes, split=args.split)
    except BudgetExceededError as exc:
        sys.exit(f"Stopped: {exc}. Spent so far: {agent.stats}")

    print(result.table(("weighted_mean_flow_time", "mean_flow_time", "makespan", "left")))
    print()
    for baseline in ("urgent_first+flexible", "weight_aware+flexible"):
        diff, half = result.paired_difference("weighted_mean_flow_time", agent.name, baseline)
        verdict = "better" if diff + half < 0 else "not shown to be better"
        print(f"{agent.name} minus {baseline}: {diff:+.1f} ± {half:.1f}  ({verdict})")
    s = agent.stats
    print(
        f"\ncalls {s.calls}, fallbacks {s.fallbacks} "
        f"(invalid {s.invalid_choices}, no tool {s.no_tool_call}, api {s.api_errors}), "
        f"tokens in/out {s.input_tokens}/{s.output_tokens}, cost ${s.cost_usd(args.model):.2f}"
    )


if __name__ == "__main__":
    main()
