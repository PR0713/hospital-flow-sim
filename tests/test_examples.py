"""The scripts in examples/ must keep working: they are the entry point for
someone adding their own agent."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

EXAMPLES = Path(__file__).parent.parent / "examples"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_agent_example_runs() -> None:
    returns = load("random_agent").main(episodes=2)
    assert len(returns) == 2 and all(r < 0 for r in returns)


def test_agent_template_runs_and_is_compared_on_common_instances() -> None:
    result = load("evaluate_my_agent").main(n_episodes=4)
    assert "my_agent" in result.records
    assert all(len(records) == 4 for records in result.records.values())
