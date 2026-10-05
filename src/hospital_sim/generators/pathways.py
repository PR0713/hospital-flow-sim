"""Random pathway (DAG) generator."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from hospital_sim.domain.dag import transitive_reduction
from hospital_sim.domain.model import PathwayStep, PathwayTemplate


def random_pathway(
    name: str,
    op_types: Sequence[str],
    n_ops: int,
    density: float,
    rng: np.random.Generator,
) -> PathwayTemplate:
    """Sample a pathway with ``n_ops`` steps and controllable precedence density.

    Steps are laid out in a fixed order and each forward pair ``(i, j)``, i < j,
    becomes a precedence constraint with probability ``density``. So
    ``density=0`` gives fully parallel steps and ``density=1`` a strict chain.
    Operation types are drawn uniformly with replacement. Redundant edges are
    removed (transitive reduction), which does not change the partial order.
    """
    if n_ops < 1:
        raise ValueError("n_ops must be >= 1")
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")
    if not op_types:
        raise ValueError("op_types must not be empty")

    ids = [f"s{i}" for i in range(n_ops)]
    types = rng.choice(len(op_types), size=n_ops)
    edge = rng.random((n_ops, n_ops)) < density
    preds = transitive_reduction(
        {ids[j]: [ids[i] for i in range(j) if edge[i, j]] for j in range(n_ops)}
    )
    return PathwayTemplate(
        name=name,
        steps=tuple(
            PathwayStep(id=ids[j], op_type=op_types[int(types[j])], predecessors=preds[ids[j]])
            for j in range(n_ops)
        ),
    )
