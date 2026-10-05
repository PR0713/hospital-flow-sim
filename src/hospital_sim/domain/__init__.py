"""Domain model: entities, DAG utilities, team matching and validation."""

from hospital_sim.domain.dag import CycleError
from hospital_sim.domain.matching import find_team
from hospital_sim.domain.model import (
    ArrivalKind,
    ArrivalSpec,
    BreakdownSpec,
    Calendar,
    ComplexitySpec,
    DurationKind,
    DurationSpec,
    HiddenRealization,
    Operation,
    OperationType,
    PathwayStep,
    PathwayTemplate,
    Patient,
    ProblemInstance,
    RequirementSpec,
    Resource,
    Role,
    RoleKind,
    Scenario,
)
from hospital_sim.domain.validation import ScenarioError, validate_scenario

__all__ = [
    "ArrivalKind",
    "ArrivalSpec",
    "BreakdownSpec",
    "Calendar",
    "ComplexitySpec",
    "CycleError",
    "DurationKind",
    "DurationSpec",
    "HiddenRealization",
    "Operation",
    "OperationType",
    "PathwayStep",
    "PathwayTemplate",
    "Patient",
    "ProblemInstance",
    "RequirementSpec",
    "Resource",
    "Role",
    "RoleKind",
    "Scenario",
    "ScenarioError",
    "find_team",
    "validate_scenario",
]
