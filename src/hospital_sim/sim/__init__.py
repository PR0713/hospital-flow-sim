"""Event-driven core: simulator, resource manager, durations, metrics."""

from hospital_sim.sim.durations import (
    Distribution,
    DurationModel,
    StandardDurationModel,
    StartContext,
    make_distribution,
)
from hospital_sim.sim.invariants import InvariantViolation, check_invariants
from hospital_sim.sim.metrics import EpisodeMetrics, compute_metrics
from hospital_sim.sim.resources import AllocationError, ResourceManager, Team
from hospital_sim.sim.runner import Dispatcher, fifo, run
from hospital_sim.sim.simulator import (
    DeadlockError,
    OpState,
    PatientStatus,
    SimulationError,
    Simulator,
)

__all__ = [
    "AllocationError",
    "DeadlockError",
    "Dispatcher",
    "Distribution",
    "DurationModel",
    "EpisodeMetrics",
    "InvariantViolation",
    "OpState",
    "PatientStatus",
    "ResourceManager",
    "SimulationError",
    "Simulator",
    "Team",
    "StandardDurationModel",
    "StartContext",
    "check_invariants",
    "compute_metrics",
    "fifo",
    "make_distribution",
    "run",
]
