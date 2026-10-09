"""Turn the environment's public observation into text an LLM can act on.

Shared by the MCP server and the in-process LLM agent. Reads only the
observation arrays and init-time environment metadata, never hidden state.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from hospital_sim.env import HospitalEnv
from hospital_sim.env.observation import OP_FEATURES, PATIENT_FEATURES

_OP = {name: i for i, name in enumerate(OP_FEATURES)}
_PATIENT = {name: i for i, name in enumerate(PATIENT_FEATURES)}


def describe_op(env: HospitalEnv, obs: dict[str, Any], slot: int) -> str:
    scale = env.time_scale
    op, p = obs["ops"][slot], obs["patients"][int(obs["op_patient"][slot])]
    name = env.config.scenario.operation_types[int(obs["op_type"][slot])].name
    flags = []
    if p[_PATIENT["urgent"]] > 0:
        flags.append("URGENT")
    if p[_PATIENT["acuity"]] > 0:
        flags.append(f"acuity {p[_PATIENT['acuity']]:g}")
    flags.append(f"weight {p[_PATIENT['weight']]:g}")
    detail = (
        f"waited {op[_OP['waited']] * scale:.1f}, expected duration "
        f"{op[_OP['expected_duration']] * scale:.1f}"
    )
    if op[_OP["is_running"]] > 0:
        detail = (
            f"running {op[_OP['elapsed']] * scale:.1f}, "
            f"expected remaining {op[_OP['expected_remaining']] * scale:.1f}"
        )
    patient = int(obs["op_patient"][slot])
    in_system = p[_PATIENT["time_in_system"]] * scale
    return f"[{name}] patient {patient} ({', '.join(flags)}; in system {in_system:.1f}) - {detail}"


def describe_state(env: HospitalEnv, obs: dict[str, Any], over: bool = False) -> str:
    scale = env.time_scale
    unit = env.config.scenario.time_unit
    g = obs["global"]
    lines = [
        f"Clock: {g[0] * scale:.1f} {unit}s. Patients in system: {int(g[1])}. "
        f"Operations: {int(g[2])} ready, {int(g[3])} startable, {int(g[4])} running."
    ]
    if over:
        lines.append("The episode is over.")
        return "\n".join(lines)
    for role, (idle, members, demand) in zip(env.config.scenario.roles, obs["roles"], strict=True):
        lines.append(
            f"Role {role.name}: {int(idle)}/{int(members)} idle, {demand:.0f} wanted by ready ops."
        )
    for slot in range(env.max_ops):
        if obs["ops"][slot, _OP["is_running"]] > 0:
            lines.append(f"Running: {describe_op(env, obs, slot)}")
    return "\n".join(lines)


def describe_actions(
    env: HospitalEnv, obs: dict[str, Any], mask: NDArray[np.bool_]
) -> list[dict[str, Any]]:
    n_rules = len(env.config.allocation_rules)
    actions: list[dict[str, Any]] = []
    for raw in np.flatnonzero(mask):
        index = int(raw)
        if index == env.wait_action:
            actions.append(
                {"action": index, "kind": "wait", "description": "Wait until the next event."}
            )
        else:
            description = "Start " + describe_op(env, obs, index // n_rules)
            actions.append({"action": index, "kind": "start", "description": description})
    return actions
