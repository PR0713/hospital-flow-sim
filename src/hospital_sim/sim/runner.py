"""Run a simulator to completion under a simple dispatching function."""

from __future__ import annotations

from collections.abc import Callable

from hospital_sim.sim.simulator import Simulator

# Given the simulator and the operations that can start now (never empty),
# return the operation to start, or ``None`` to wait for the next event.
Dispatcher = Callable[[Simulator, list[int]], int | None]


def fifo(sim: Simulator, startable: list[int]) -> int:
    """Start the operation that has been ready the longest."""
    return startable[0]


def run(sim: Simulator, dispatch: Dispatcher = fifo) -> Simulator:
    """Drive ``sim`` until every patient is finished. Teams are first-idle."""
    while sim.run_until_decision():
        op = dispatch(sim, sim.startable_ops())
        if op is not None:
            sim.start(op)
        elif not sim.advance():
            raise RuntimeError("dispatcher chose to wait but no event is pending")
    return sim
