"""MCP server exposing one hospital episode as tools.

Run over stdio:  .venv/bin/python -m hospital_sim_mcp.server
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from hospital_sim_mcp.session import EpisodeSession, SessionError

mcp = FastMCP("hospital-flow-sim")
session = EpisodeSession()


def _guard(fn: Any) -> Any:
    """Return caller mistakes as data so the model can correct itself."""
    try:
        return fn()
    except SessionError as exc:
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def list_configs() -> list[str]:
    """List the hospital scenarios that reset() accepts."""
    return session.available_configs()


@mcp.tool()
def reset(config: str = "example_hospital_dynamic", seed: int = 0, split: str = "train") -> Any:
    """Start a new episode. The same config, seed and split always give the same patients.
    split is one of train, eval, test. Returns the first state."""
    return _guard(lambda: session.reset(config, seed, split))


@mcp.tool()
def get_state() -> Any:
    """Describe the hospital now: clock, patients, idle staff per role, running operations."""
    return _guard(session.state)


@mcp.tool()
def list_valid_actions() -> Any:
    """List every action allowed right now. Pass one action number to step().
    Choose only from this list; other numbers are rejected."""
    return _guard(session.valid_actions)


@mcp.tool()
def step(action: int) -> Any:
    """Take one action from list_valid_actions(). Returns the reward (negative = cost),
    whether the episode is over, and the new state."""
    return _guard(lambda: session.step(action))


@mcp.tool()
def get_metrics() -> Any:
    """Totals so far. When the episode is over this includes patients finished, flow time,
    waiting time and patients who left unseen."""
    return _guard(session.metrics)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
