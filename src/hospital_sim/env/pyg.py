"""PyTorch Geometric adapter for ``HospitalEnv.graph()``.

Optional: needs ``pip install -e ".[gnn]"`` (torch and torch_geometric).
Both are imported only when ``to_hetero_data`` is called, so the core
package never requires them.

Node types and features (column names from ``hospital_sim.env.observation``):

* ``op`` - ``OP_FEATURES``; also ``op_type`` (operation type index, long);
* ``patient`` - ``PATIENT_FEATURES``;
* ``resource`` - ``RESOURCE_FEATURES``;
* ``role`` - ``ROLE_FEATURES``.

Edge types (``edge_index`` is ``[2, E]``, source first):

* ``(op, precedes, op)`` - predecessor to successor, hidden operations bridged;
* ``(op, requires, role)`` - ``edge_attr`` ``[E, 1]`` is the quantity;
* ``(op, of, patient)``;
* ``(resource, fills, role)`` - role membership;
* ``(op, uses, resource)`` - resource currently working on the operation.

Graph-level features (``GLOBAL_FEATURES``) are in ``data.global_x``
(shape ``[1, G]``). Reverse edges are not added; use
``torch_geometric.transforms.ToUndirected`` if a model needs them.
Operation and patient rows are those of the padded observation (row ``i``
here is row ``i`` there), so an action's slot indexes ``data["op"]``.
"""

from __future__ import annotations

from typing import Any

EDGE_TYPES = (
    ("op", "precedes", "op"),
    ("op", "requires", "role"),
    ("op", "of", "patient"),
    ("resource", "fills", "role"),
    ("op", "uses", "resource"),
)


def to_hetero_data(graph: dict[str, Any]) -> Any:
    """``env.graph()`` as a ``torch_geometric.data.HeteroData``."""
    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ImportError(
            "to_hetero_data needs torch and torch_geometric: pip install -e '.[gnn]'"
        ) from exc

    data = HeteroData()
    for node_type, features in graph["nodes"].items():
        x = torch.as_tensor(features, dtype=torch.float32)
        data[node_type].x = x
        data[node_type].num_nodes = x.shape[0]
    data["op"].op_type = torch.as_tensor(graph["op_type"], dtype=torch.long)
    for src, relation, dst in EDGE_TYPES:
        index = graph["edges"][f"{src}__{relation}__{dst}"]
        data[src, relation, dst].edge_index = torch.as_tensor(index, dtype=torch.long).reshape(
            2, -1
        )
    quantities = graph["edge_attr"]["op__requires__role"]
    data["op", "requires", "role"].edge_attr = torch.as_tensor(
        quantities, dtype=torch.float32
    ).reshape(-1, 1)
    data.global_x = torch.as_tensor(graph["global"], dtype=torch.float32).reshape(1, -1)
    return data
